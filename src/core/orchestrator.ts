import type { JsonlLogger } from "../logging/jsonlLogger.js";
import type {
  AgentRequest,
  AgentRuntime,
  AgentStreamEvent,
  AuthorizationAction,
  AuthorizationService,
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackStore,
  InboundMessage,
  IntentDetectionService,
  KnowledgeWorkspaceResolver,
  MessageHandler,
  OutboundMessage,
  ProgressReporter,
  UserIntent,
  UserRole
} from "./ports.js";

export type OrchestratorDependencies = {
  readonly workspaceResolver: KnowledgeWorkspaceResolver;
  readonly authorization: AuthorizationService;
  readonly agentRuntimes: Record<string, AgentRuntime>;
  readonly sessionStore?: ConversationSessionStore;
  readonly historyStore?: ConversationHistoryStore;
  readonly conversationLogger?: JsonlLogger;
  readonly progressReporter?: ProgressReporter;
  readonly intentDetection?: IntentDetectionService | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
};

export class AgentOrchestrator implements MessageHandler {
  public constructor(private readonly dependencies: OrchestratorDependencies) {}

  public async handle(message: InboundMessage): Promise<OutboundMessage> {
    const commandResponse = this.handleCommandWithoutWorkspace(message);
    if (commandResponse !== undefined) {
      await this.logConversation(message, commandResponse.text);
      return commandResponse;
    }

    const workspaces = await this.dependencies.workspaceResolver.resolve(message);
    const workspace = workspaces[0];

    if (workspace === undefined) {
      const response = { text: "No matching knowledge base or source repository was found." };
      await this.logConversation(message, response.text);
      return response;
    }

    if (this.isNewConversationCommand(message)) {
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
    const intent = await this.detectIntent(message.text, role);
    const requiredAction = actionForIntent(intent);

    if (requiredAction !== undefined) {
      const intentDecision = await this.dependencies.authorization.can(message.user, requiredAction, workspace);
      if (!intentDecision.allowed) {
        await this.saveFeedback(message, workspace.path, intent, role);
        const response = { text: responseForDeniedIntent(intent) };
        await this.logConversation(message, response.text, workspace.path);
        return response;
      }
    }

    const runtime = this.dependencies.agentRuntimes[role];
    if (runtime === undefined) {
      const response = { text: `No runtime configured for role: ${role}` };
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    const sessionKey = this.createSessionKey(message, workspace.path);
    const sessionId = await this.dependencies.sessionStore?.get(sessionKey);

    const request: AgentRequest = {
      user: message.user,
      text: message.text,
      workspacePath: workspace.path,
      progress: (event) => this.publishProgress(message, event),
      stream: message.stream ?? ((event) => this.publishStreamEvent(message, event)),
    };
    if (sessionId !== undefined) {
      (request as { sessionId?: string }).sessionId = sessionId;
    }

    try {
      const response = await runtime.run(request);

      if (response.sessionId !== undefined) {
        await this.dependencies.sessionStore?.set(sessionKey, response.sessionId);
      }
      await this.dependencies.historyStore?.append(sessionKey, [
        { role: "user", content: message.text },
        { role: "assistant", content: response.text },
      ]);
      await this.logConversation(message, response.text, workspace.path);

      return {
        text: response.text,
        ...(response.sessionId !== undefined ? { sessionId: response.sessionId } : {}),
      };
    } catch (error) {
      const errorText = `Model call failed: ${formatRuntimeError(error)}`;
      await this.logConversation(message, errorText, workspace.path);
      return { text: errorText };
    }
  }

  private handleCommandWithoutWorkspace(message: InboundMessage): OutboundMessage | undefined {
    const normalizedText = message.text.trim().toLowerCase();

    if (normalizedText === "/help") {
      return {
        text: [
          "Available commands:",
          "/new - start a fresh conversation",
          "/help - show this help message"
        ].join("\n")
      };
    }

    return undefined;
  }

  private isNewConversationCommand(message: InboundMessage): boolean {
    return message.text.trim().toLowerCase() === "/new";
  }

  private async detectIntent(userMessage: string, role: UserRole): Promise<UserIntent> {
    if (this.dependencies.intentDetection === undefined) {
      return { type: "query" };
    }
    return this.dependencies.intentDetection.detectIntent(userMessage, role);
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
    const contextParts = (history ?? []).slice(-4).map((m) => `${m.role}: ${m.content}`);
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
      const runtime = this.dependencies.agentRuntimes[role];
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
      workspacePath
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

  private async publishProgress(message: InboundMessage, event: Parameters<ProgressReporter["publish"]>[1]): Promise<void> {
    await this.dependencies.progressReporter?.publish(message.user, event);
  }

  private publishStreamEvent(message: InboundMessage, event: AgentStreamEvent): void {
    const stage = event.type === "text_delta" ? "agent.text_delta"
      : event.type === "tool_use_start" ? "agent.tool_use_start"
      : event.type === "tool_use_result" ? "agent.tool_use_result"
      : event.type === "thinking" ? "agent.thinking"
      : event.type === "completed" ? "agent.completed"
      : "agent.error";

    void this.publishProgress(message, {
      stage,
      message: stage,
      data: event,
    });
  }
}

function actionForIntent(intent: UserIntent): AuthorizationAction | undefined {
  return intent.type === "mutate" || intent.type === "update_kb" ? "mutate" : undefined;
}

function responseForDeniedIntent(intent: UserIntent): string {
  if (intent.type === "update_kb") {
    return "感谢您的反馈！我已记录您希望更新知识库的请求。当前文档助手权限仅支持查看知识库，不支持修改内容。您的请求已保存，管理员将尽快审核处理。";
  }

  return "我已识别到这是修改请求，但你当前是文档助手权限，只能查看知识库，不能修改文件。您的请求已记录，管理员将尽快审核处理。";
}

function formatRuntimeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Unknown error.";
  }

  if (error.message.includes("invalid api key")) {
    return "Invalid API key. Check ANTHROPIC_API_KEY for the configured Anthropic-compatible endpoint.";
  }

  return error.message.split("\n")[0] ?? "Unknown error.";
}
