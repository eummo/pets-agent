import type {
  AgentConversationMessage,
  AgentRequest,
  AgentRuntime,
  AgentRuntimeFactory,
  AgentStreamEvent,
  AuthorizationService,
  ConversationHistoryStore,
  ConversationLogger,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackStore,
  InboundMessage,
  KnowledgeWorkspace,
  KnowledgeWorkspaceResolver,
  MessageGateway,
  OutboundMessage,
  ProgressReporter,
  UserIntent,
  UserRole
} from "./contracts.js";
import { handleCommandWithoutWorkspace, isNewConversationCommand } from "./conversationCommands.js";
import { actionForIntent, responseForDeniedIntent } from "./intentAuthorization.js";
import { fallbackIntentFor } from "./intentHeuristics.js";
import { parseIntentResponse } from "../agent/intentAgentRuntime.js";
import { RuntimeCache } from "./runtimeCache.js";
import { formatInternalError, formatSafeRuntimeError } from "./runtimeErrorFormatter.js";
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
};

export class AgentOrchestrator implements MessageGateway {
  private readonly runtimeCache: RuntimeCache;

  public constructor(private readonly dependencies: OrchestratorDependencies) {
    this.runtimeCache = new RuntimeCache(dependencies.initialRuntimes ?? {}, dependencies.runtimeFactory);
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
    await this.logEvent("workspace.resolved", message, { workspaceId: workspace.id, workspaceKind: workspace.kind, workspacePath: workspace.path });

    if (isNewConversationCommand(message)) {
      const response = await this.startNewConversation(message, workspace.path);
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    const readDecision = await this.dependencies.authorization.can(message.user, "read", workspace);
    if (!readDecision.allowed) {
      const response = { text: readDecision.reason ?? "You do not have permission to access this workspace." };
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    const role = await this.dependencies.authorization.roleFor(message.user);
    await this.logEvent("role.resolved", message, { role });
    const intentCheck = await this.checkIntentAuthorization(message, workspace, role);
    if (intentCheck !== undefined) {
      return intentCheck;
    }

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

  private async checkIntentAuthorization(
    message: InboundMessage,
    workspace: { readonly path: string },
    role: UserRole,
  ): Promise<OutboundMessage | undefined> {
    const sessionKey = this.createSessionKey(message, workspace.path);
    const history = await this.dependencies.historyStore?.get(sessionKey);
    const recentHistory = history?.slice(-4);

    const intent = await this.detectIntent(message.text, role, recentHistory);
    await this.logEvent("intent.classified", message, { role, intentType: intent.type, workspacePath: workspace.path });
    const requiredAction = actionForIntent(intent);

    if (requiredAction !== undefined) {
      const intentDecision = await this.dependencies.authorization.can(message.user, requiredAction, workspace as KnowledgeWorkspace);
      if (!intentDecision.allowed) {
        await this.saveFeedback(message, workspace.path, intent, role);
        await this.logEvent("permission.denied", message, { role, action: requiredAction, intentType: intent.type, workspacePath: workspace.path });
        const response = { text: responseForDeniedIntent(intent) };
        await this.logConversation(message, response.text, workspace.path);
        return response;
      }
    }

    return undefined;
  }

  private async resolveRuntime(role: string): Promise<AgentRuntime | undefined> {
    return this.runtimeCache.resolve(role);
  }

  private async executeRuntime(
    message: InboundMessage,
    workspacePath: string,
    role: string,
    runtime: AgentRuntime,
  ): Promise<OutboundMessage> {
    const sessionKey = this.createSessionKey(message, workspacePath);
    const sessionId = await this.dependencies.sessionStore?.get(sessionKey);

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
          summaryLength: summary.length,
        });
      },
      ...(sessionId !== undefined ? { sessionId } : {}),
    };

    try {
      const response = await runtime.run(request);

      if (response.sessionId !== undefined) {
        await this.dependencies.sessionStore?.set(sessionKey, response.sessionId);
      }
      await this.dependencies.historyStore?.append(sessionKey, [
        { role: "user", content: message.text },
        { role: "assistant", content: response.text },
      ]);
      if (response.contextUsage !== undefined) {
        await this.logEvent("context.usage", message, {
          workspacePath,
          ...response.contextUsage,
        });
      }
      await this.logConversation(message, response.text, workspacePath);

      return {
        text: response.text,
        ...(response.sessionId !== undefined ? { sessionId: response.sessionId } : {}),
      };
    } catch (error) {
      const safeMessage = formatSafeRuntimeError(error);
      const internalDetail = formatInternalError(error);
      await this.logConversation(message, `Model call failed: ${internalDetail}`, workspacePath);
      return { text: safeMessage };
    }
  }

  private async detectIntent(userMessage: string, role: UserRole, history?: readonly AgentConversationMessage[]): Promise<UserIntent> {
    try {
      const intentRuntime = await this.runtimeCache.resolve("intent");
      if (intentRuntime !== undefined) {
        const request: AgentRequest = {
          user: { id: "system" },
          text: userMessage,
          workspacePath: "",
          role,
          ...(history !== undefined ? { history } : {}),
        };
        const response = await intentRuntime.run(request);
        return parseIntentResponse(response.text);
      }
    } catch {
      // Fall back to heuristic if intent runtime fails
    }
    return fallbackIntentFor(userMessage);
  }

  private async saveFeedback(
    message: InboundMessage,
    workspacePath: string,
    intent: UserIntent,
    role: UserRole,
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
      status: "pending",
    });
  }

  private async startNewConversation(message: InboundMessage, workspacePath: string): Promise<OutboundMessage> {
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
      ...(message.chatId !== undefined ? { chatId: message.chatId } : {}),
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
      input: message.text,
      output,
      workspacePath
    });
  }

  private async logEvent(
    type: string,
    message: InboundMessage,
    data: Record<string, unknown>,
  ): Promise<void> {
    await this.dependencies.eventLogger?.write({
      type,
      channel: message.channel,
      messageId: message.id,
      userId: message.user.id,
      ...data,
    });
  }

  private async publishProgress(message: InboundMessage, event: Parameters<ProgressReporter["publish"]>[1]): Promise<void> {
    await this.dependencies.progressReporter?.publish(message.user, event);
  }

  private publishStreamEvent(message: InboundMessage, event: AgentStreamEvent): void {
    void this.publishProgress(message, progressEventForAgentStreamEvent(event));
  }
}
