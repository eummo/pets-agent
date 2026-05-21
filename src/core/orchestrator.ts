import type { JsonlLogger } from "../logging/jsonlLogger.js";
import type {
  AgentRequest,
  AgentRuntime,
  AgentStreamEvent,
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
  readonly agentRuntimes: Record<string, AgentRuntime>;
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

    // Role-based routing: select runtime by user role
    const role = await this.dependencies.authorization.roleFor(message.user);
    const effectiveRole = role === "viewer" ? "reviewer" : role;
    const runtime = this.dependencies.agentRuntimes[effectiveRole];
    if (runtime === undefined) {
      const response = { text: `No runtime configured for role: ${effectiveRole}` };
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

  private async startNewConversation(message: InboundMessage, workspacePath: string): Promise<OutboundMessage> {
    const sessionKey = this.createSessionKey(message, workspacePath);
    const sessionId = await this.dependencies.sessionStore?.get(sessionKey);

    if (sessionId !== undefined) {
      const role = await this.dependencies.authorization.roleFor(message.user);
      const effectiveRole = role === "viewer" ? "reviewer" : role;
      const runtime = this.dependencies.agentRuntimes[effectiveRole];
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

function formatRuntimeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Unknown error.";
  }

  if (error.message.includes("invalid api key")) {
    return "Invalid API key. Check LOCAL_LLM_API_KEY for the configured MiniMax Anthropic endpoint.";
  }

  return error.message.split("\n")[0] ?? "Unknown error.";
}
