import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { classifyMessageIntent } from "./intent.js";
import type {
  AgentRequest,
  AgentRuntime,
  AuthorizationService,
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  InboundMessage,
  KnowledgeWorkspaceResolver,
  MessageHandler,
  OutboundMessage,
  ProgressReporter
} from "./ports.js";

export type OrchestratorDependencies = {
  readonly workspaceResolver: KnowledgeWorkspaceResolver;
  readonly authorization: AuthorizationService;
  readonly agentRuntime: AgentRuntime;
  readonly codeChangeRuntime?: AgentRuntime;
  readonly sessionStore?: ConversationSessionStore;
  readonly historyStore?: ConversationHistoryStore;
  readonly conversationLogger?: JsonlLogger;
  readonly progressReporter?: ProgressReporter;
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

    const decision = await this.dependencies.authorization.can(message.user, "read", workspace);
    if (!decision.allowed) {
      const response = { text: decision.reason ?? "You do not have permission to access this workspace." };
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    if (classifyMessageIntent(message.text) === "mutate") {
      const response = await this.handleMutation(message, workspace.path);
      await this.logConversation(message, response.text, workspace.path);
      return response;
    }

    const sessionKey = this.createSessionKey(message, workspace.path);
    const response = await this.runAgentSafely(message, workspace.path, sessionKey);
    await this.logConversation(message, response.text, workspace.path);

    return { text: response.text };
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

  private async startNewConversation(message: InboundMessage, workspacePath: string): Promise<OutboundMessage> {
    const sessionKey = this.createSessionKey(message, workspacePath);
    const sessionId = await this.dependencies.sessionStore?.get(sessionKey);

    if (sessionId !== undefined) {
      await this.dependencies.agentRuntime.disposeSession(sessionId);
      await this.dependencies.sessionStore?.delete(sessionKey);
    }
    await this.dependencies.historyStore?.archive(sessionKey);

    return {
      text: "New conversation started."
    };
  }

  private async handleMutation(
    message: InboundMessage,
    workspacePath: string
  ): Promise<{ readonly text: string }> {
    const mutationDecision = await this.dependencies.authorization.can(message.user, "mutate", {
      kind: "source-repository",
      id: "selected-workspace",
      path: workspacePath
    });

    if (!mutationDecision.allowed) {
      await this.publishProgress(message, {
        stage: "code_change.denied",
        message: "修改请求已识别，但当前用户没有直接修改权限。"
      });
      return {
        text: [
          "已识别到修改请求。",
          "你当前权限不足，不能直接修改文件。",
          mutationDecision.reason ?? "请联系开发者处理，或先让我生成修改建议。"
        ].join("\n")
      };
    }

    if (this.dependencies.codeChangeRuntime === undefined) {
      await this.publishProgress(message, {
        stage: "code_change.not_configured",
        message: "代码执行 runtime 未配置，没有修改文件。"
      });
      return {
        text: [
          "已识别到修改请求。",
          "你当前具备开发者权限，可以进入代码变更流程。",
          "不过代码执行 runtime 还没有配置，所以这次没有修改文件。"
        ].join("\n")
      };
    }

    try {
      return await this.dependencies.codeChangeRuntime.run({
        user: message.user,
        text: message.text,
        workspacePath,
        progress: (event) => this.publishProgress(message, event)
      });
    } catch (error) {
      await this.publishProgress(message, {
        stage: "code_change.failed",
        message: "代码变更流程失败。",
        data: { error: formatRuntimeError(error) }
      });
      return {
        text: `Code change failed: ${formatRuntimeError(error)}`
      };
    }
  }

  private async runAgentSafely(
    message: InboundMessage,
    workspacePath: string,
    sessionKey: ConversationSessionKey
  ): Promise<{ readonly text: string }> {
    try {
      const sessionId = await this.dependencies.sessionStore?.get(sessionKey);
      const history = (await this.dependencies.historyStore?.get(sessionKey)) ?? [];
      const request: AgentRequest = {
        user: message.user,
        text: message.text,
        workspacePath,
        ...(history.length === 0 ? {} : { history }),
        ...(sessionId === undefined ? {} : { sessionId })
      };
      const response = await this.dependencies.agentRuntime.run(request);

      if (response.sessionId !== undefined) {
        await this.dependencies.sessionStore?.set(sessionKey, response.sessionId);
      }
      await this.dependencies.historyStore?.append(sessionKey, [
        { role: "user", content: message.text },
        { role: "assistant", content: response.text }
      ]);

      return response;
    } catch (error) {
      return {
        text: `Model call failed: ${formatRuntimeError(error)}`
      };
    }
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
}

function formatRuntimeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Unknown error.";
  }

  if (error.message.includes("invalid api key")) {
    return "Invalid API key. Check LOCAL_LLM_API_KEY for the configured MiniMax Anthropic endpoint.";
  }

  return error.message.split("\n")[0] ?? "Unknown error.";
}
