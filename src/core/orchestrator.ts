import type {
  AgentRequest,
  AgentRuntime,
  AgentRuntimeFactory,
  AgentStreamEvent,
  AuthorizationAction,
  AuthorizationService,
  ConversationHistoryStore,
  ConversationLogger,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackStore,
  InboundMessage,
  IntentDetectionService,
  KnowledgeWorkspace,
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
  readonly runtimeFactory?: AgentRuntimeFactory;
  readonly sessionStore?: ConversationSessionStore;
  readonly historyStore?: ConversationHistoryStore;
  readonly conversationLogger?: ConversationLogger;
  readonly progressReporter?: ProgressReporter;
  readonly intentDetection?: IntentDetectionService | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
};

export class AgentOrchestrator implements MessageHandler {
  private readonly runtimeCache: Map<string, AgentRuntime>;
  private readonly runtimeCacheOrder: string[];
  private readonly maxCacheSize: number;

  public constructor(private readonly dependencies: OrchestratorDependencies) {
    this.runtimeCache = new Map(Object.entries(dependencies.agentRuntimes));
    this.runtimeCacheOrder = Object.keys(dependencies.agentRuntimes);
    this.maxCacheSize = 16;
  }

  public async handle(message: InboundMessage): Promise<OutboundMessage> {
    const commandResponse = this.handleCommandWithoutWorkspace(message);
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
    const intent = await this.detectIntent(message.text, role);
    const requiredAction = actionForIntent(intent);

    if (requiredAction !== undefined) {
      const intentDecision = await this.dependencies.authorization.can(message.user, requiredAction, workspace as KnowledgeWorkspace);
      if (!intentDecision.allowed) {
        await this.saveFeedback(message, workspace.path, intent, role);
        const response = { text: responseForDeniedIntent(intent) };
        await this.logConversation(message, response.text, workspace.path);
        return response;
      }
    }

    return undefined;
  }

  private async resolveRuntime(role: string): Promise<AgentRuntime | undefined> {
    let runtime = this.runtimeCache.get(role);
    if (runtime === undefined && this.dependencies.runtimeFactory !== undefined) {
      const created = await this.dependencies.runtimeFactory.createRuntime(role);
      if (created !== undefined) {
        this.cacheRuntime(role, created);
        runtime = created;
      }
    }
    return runtime;
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
      progress: (event) => this.publishProgress(message, event),
      stream: message.stream ?? ((event) => this.publishStreamEvent(message, event)),
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

  private cacheRuntime(role: string, runtime: AgentRuntime): void {
    this.runtimeCache.set(role, runtime);
    this.runtimeCacheOrder.push(role);
    while (this.runtimeCacheOrder.length > this.maxCacheSize) {
      const oldest = this.runtimeCacheOrder.shift();
      if (oldest !== undefined) {
        this.runtimeCache.delete(oldest);
      }
    }
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
      const runtime = this.runtimeCache.get(role);
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

function formatSafeRuntimeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Model call failed due to an unexpected error. Please try again later.";
  }

  if (error.message.includes("invalid api key")) {
    return "Model call failed: API key is invalid or not configured. Contact an administrator.";
  }

  return "Model call failed. The service encountered an error processing your request. Please try again later.";
}

function formatInternalError(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }

  return error.message.split("\n")[0] ?? "Unknown error.";
}
