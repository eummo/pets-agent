import type {
  AgentConversationMessage,
  ConversationLogger,
  InboundMessage,
  MessageGateway,
  OutboundMessage,
  UserRole
} from "./index.js";
import type {
  AgentRequest,
  AgentRuntime,
  AgentRuntimeFactory,
  AgentStreamEvent,
  ProgressReporter
} from "../agent/index.js";
import type { AuthorizationService } from "../auth/index.js";
import type {
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackStore
} from "../persistence/index.js";
import type { KnowledgeWorkspace, KnowledgeWorkspaceResolver } from "../workspace/index.js";
import type { IntentDetectionService, UserIntent } from "../intent/index.js";
import { handleCommandWithoutWorkspace, isNewConversationCommand } from "./conversationCommands.js";
import { responseForDeniedIntent } from "./intentAuthorization.js";
import { fallbackIntentFor } from "./intentHeuristics.js";
import { RequestAuthorizationGate } from "./requestAuthorizationGate.js";
import { RuntimeCache } from "./runtimeCache.js";
import { formatInternalError, formatSafeRuntimeError } from "./runtimeErrorFormatter.js";
import { formatUnknownError } from "./unknownRecord.js";
import { progressEventForAgentStreamEvent } from "./streamProgressMapper.js";

export type OrchestratorDependencies = {
  readonly workspaceResolver: KnowledgeWorkspaceResolver;
  readonly authorization: AuthorizationService;
  readonly runtimeFactory: AgentRuntimeFactory;
  readonly initialRuntimes?: Record<string, AgentRuntime>;
  readonly sessionStore?: ConversationSessionStore;
  readonly historyStore?: ConversationHistoryStore;
  readonly conversationLogger?: ConversationLogger;
  readonly eventLogger?: ConversationLogger;
  readonly progressReporter?: ProgressReporter;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly intentDetection?: IntentDetectionService;
};

export class AgentOrchestrator implements MessageGateway {
  private readonly runtimeCache: RuntimeCache;
  private readonly requestAuthorizationGate: RequestAuthorizationGate;
  private readonly lastRoleForSession = new Map<string, string>();

  public constructor(private readonly dependencies: OrchestratorDependencies) {
    this.runtimeCache = new RuntimeCache(
      dependencies.initialRuntimes ?? {},
      dependencies.runtimeFactory
    );
    this.requestAuthorizationGate = new RequestAuthorizationGate({
      authorization: dependencies.authorization,
      detectIntent: (userMessage, role, history) => this.detectIntent(userMessage, role, history)
    });
  }

  public async handle(message: InboundMessage): Promise<OutboundMessage> {
    const commandResponse = handleCommandWithoutWorkspace(message);
    if (commandResponse !== undefined) {
      await this.logConversation(message, commandResponse.text);
      return commandResponse;
    }

    const workspace = await this.resolveWorkspace(message);
    if (workspace === undefined) {
      const response = { text: "No matching knowledge base or source repository was found." };
      await this.logConversation(message, response.text);
      return response;
    }
    await this.logEvent("workspace.resolved", message, {
      workspaceId: workspace.id,
      workspaceKind: workspace.kind,
      workspacePath: workspace.path
    });

    if (isNewConversationCommand(message)) {
      const response = await this.startNewConversation(message, workspace.path);
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    const authorization = await this.authorizeRequest(message, workspace);
    await this.logEvent("role.resolved", message, { role: authorization.role });
    if (authorization.status === "denied") {
      if (authorization.deniedAt === "intent") {
        await this.logIntent(message, workspace, authorization.role, authorization.intent);
        await this.saveFeedback(message, workspace.path, authorization.intent, authorization.role);
        await this.logEvent("permission.denied", message, {
          role: authorization.role,
          action: authorization.requiredAction,
          intentType: authorization.intent.type,
          workspacePath: workspace.path
        });
      }
      const response = { text: authorization.responseText };
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }
    await this.logIntent(message, workspace, authorization.role, authorization.intent);

    const role = authorization.role;
    const runtime = await this.resolveRuntime(role);
    if (runtime === undefined) {
      const response = { text: `No runtime configured for role: ${role}` };
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }
    await this.logEvent("runtime.selected", message, { role, runtime: runtime.name });

    return this.executeRuntime(message, workspace.path, role, runtime);
  }

  private async resolveWorkspace(message: InboundMessage) {
    const workspaces = await this.dependencies.workspaceResolver.resolve(message);
    return workspaces[0];
  }

  private async authorizeRequest(message: InboundMessage, workspace: KnowledgeWorkspace) {
    const sessionKey = this.createSessionKey(message, workspace.path);
    const history = await this.dependencies.historyStore?.get(sessionKey);
    const recentHistory = history?.slice(-4);

    return this.requestAuthorizationGate.evaluate({
      message,
      workspace,
      ...(recentHistory !== undefined ? { history: recentHistory } : {})
    });
  }

  private async logIntent(
    message: InboundMessage,
    workspace: KnowledgeWorkspace,
    role: UserRole,
    intent: UserIntent
  ): Promise<void> {
    await this.logEvent("intent.classified", message, {
      role,
      intentType: intent.type,
      workspacePath: workspace.path
    });
  }

  private async resolveRuntime(role: string): Promise<AgentRuntime | undefined> {
    return this.runtimeCache.resolve(role);
  }

  private async executeRuntime(
    message: InboundMessage,
    workspacePath: string,
    role: string,
    runtime: AgentRuntime
  ): Promise<OutboundMessage> {
    const sessionKey = this.createSessionKey(message, workspacePath);
    let sessionId = await this.dependencies.sessionStore?.get(sessionKey);
    let priorHistory: readonly AgentConversationMessage[] | undefined;

    // When the role changes, the old sessionId belongs to a different runtime
    // instance that cannot be reused. Dispose the old session and start a new
    // one, but carry the conversation history so the new role can see context.
    const sessionKeyText = JSON.stringify(sessionKey);
    const previousRole = this.lastRoleForSession.get(sessionKeyText);
    if (previousRole !== undefined && previousRole !== role) {
      if (sessionId !== undefined) {
        const oldRuntime = await this.resolveRuntime(previousRole);
        if (oldRuntime !== undefined) {
          await oldRuntime.disposeSession(sessionId);
        }
        await this.dependencies.sessionStore?.delete(sessionKey);
      }
      priorHistory = await this.dependencies.historyStore?.get(sessionKey);
      sessionId = undefined;
    }

    const request: AgentRequest = {
      user: message.user,
      text: message.text,
      workspacePath,
      role,
      progress: (event) => this.publishProgress(message, event),
      stream: message.stream ?? ((event) => this.publishStreamEvent(message, event)),
      onCompact: async (summary) => {
        await this.dependencies.historyStore?.compact(sessionKey, summary);
        await this.logEvent("context.compacted", message, {
          workspacePath,
          summaryLength: summary.length
        });
      },
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(priorHistory !== undefined && priorHistory.length > 0 ? { history: priorHistory } : {}),
      ...(message.chatType !== undefined ? { chatType: message.chatType } : {}),
      ...(message.chatId !== undefined ? { chatId: message.chatId } : {})
    };

    try {
      const response = await runtime.run(request);

      this.lastRoleForSession.set(sessionKeyText, role);
      if (response.sessionId !== undefined) {
        await this.dependencies.sessionStore?.set(sessionKey, response.sessionId);
      }
      await this.dependencies.historyStore?.append(sessionKey, [
        { role: "user", content: message.text },
        { role: "assistant", content: response.text }
      ]);
      if (response.contextUsage !== undefined) {
        await this.logEvent("context.usage", message, {
          workspacePath,
          ...response.contextUsage
        });
      }
      await this.logConversation(message, response.text, workspacePath);

      return {
        text: response.text,
        ...(response.sessionId !== undefined ? { sessionId: response.sessionId } : {})
      };
    } catch (error) {
      const safeMessage = formatSafeRuntimeError(error);
      const internalDetail = formatInternalError(error);
      await this.logConversation(message, `Model call failed: ${internalDetail}`, workspacePath);
      return { text: safeMessage };
    }
  }

  private async detectIntent(
    userMessage: string,
    role: UserRole,
    history?: readonly AgentConversationMessage[]
  ): Promise<UserIntent> {
    const intentDetection = this.dependencies.intentDetection;
    if (intentDetection === undefined) {
      return fallbackIntentFor(userMessage);
    }

    try {
      return await intentDetection.detectIntent(userMessage, role, history);
    } catch (error) {
      void this.dependencies.eventLogger?.write({
        type: "intent.fallback",
        channel: "",
        messageId: "",
        userId: "system",
        reason: formatUnknownError(error),
        userMessage,
        role
      });
    }
    return fallbackIntentFor(userMessage);
  }

  private async saveFeedback(
    message: InboundMessage,
    workspacePath: string,
    intent: UserIntent,
    role: UserRole
  ): Promise<void> {
    if (this.dependencies.feedbackStore === undefined) return;

    const sessionKey = this.createSessionKey(message, workspacePath);
    const history = await this.dependencies.historyStore?.get(sessionKey);
    const contextParts = (history ?? []).map((m) => `${m.role}: ${m.content}`);
    // Include the current user message since it hasn't been appended to history yet
    contextParts.push(`user: ${message.text}`);
    // Include the denial response so admin sees the full conversation
    const denialResponse = responseForDeniedIntent(intent);
    contextParts.push(`assistant: ${denialResponse}`);
    const conversationContext = contextParts.join("\n");

    await this.dependencies.feedbackStore.save({
      userId: message.user.id,
      channel: message.channel,
      messageId: message.id,
      workspacePath,
      intentType: intent.type,
      roleName: role,
      userMessage: message.text,
      conversationContext,
      status: "pending"
    });
  }

  private async startNewConversation(
    message: InboundMessage,
    workspacePath: string
  ): Promise<OutboundMessage> {
    const sessionKey = this.createSessionKey(message, workspacePath);
    const sessionId = await this.dependencies.sessionStore?.get(sessionKey);

    if (sessionId !== undefined) {
      const role = await this.dependencies.authorization.roleFor(message.user);
      const runtime = await this.resolveRuntime(role);
      if (runtime !== undefined) {
        await runtime.disposeSession(sessionId);
      }
      await this.dependencies.sessionStore?.delete(sessionKey);
    }
    await this.dependencies.historyStore?.archive(sessionKey);

    return {
      text: "New conversation started."
    };
  }

  private createSessionKey(message: InboundMessage, workspacePath: string): ConversationSessionKey {
    return {
      channel: message.channel,
      userId: message.user.id,
      workspacePath,
      ...(message.chatId !== undefined ? { chatId: message.chatId } : {})
    };
  }

  private async logConversation(
    message: InboundMessage,
    output: string,
    workspacePath?: string
  ): Promise<void> {
    await this.dependencies.conversationLogger?.write({
      type: "conversation.turn",
      channel: message.channel,
      messageId: message.id,
      userId: message.user.id,
      chatId: message.chatId,
      input: message.text,
      output,
      workspacePath
    });
  }

  private async logEvent(
    type: string,
    message: InboundMessage,
    data: Record<string, unknown>
  ): Promise<void> {
    await this.dependencies.eventLogger?.write({
      type,
      channel: message.channel,
      messageId: message.id,
      userId: message.user.id,
      chatId: message.chatId,
      ...data
    });
  }

  private async publishProgress(
    message: InboundMessage,
    event: Parameters<ProgressReporter["publish"]>[1]
  ): Promise<void> {
    await this.dependencies.progressReporter?.publish(message.user, event);
  }

  private publishStreamEvent(message: InboundMessage, event: AgentStreamEvent): void {
    void this.publishProgress(message, progressEventForAgentStreamEvent(event));
  }
}
