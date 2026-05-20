import Anthropic from "@anthropic-ai/sdk";
import type {
  BetaManagedAgentsSession,
  BetaManagedAgentsSessionEvent,
  EventListParams,
  EventSendParams,
  SessionCreateParams
} from "@anthropic-ai/sdk/resources/beta/sessions";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { buildWorkspaceContext } from "./workspaceContext.js";

export type ManagedSessionsClient = {
  readonly beta: {
    readonly sessions: {
      create(params: SessionCreateParams): Promise<Pick<BetaManagedAgentsSession, "id">>;
      archive(sessionId: string): Promise<unknown>;
      readonly events: {
        send(sessionId: string, params: EventSendParams): Promise<unknown>;
        list(
          sessionId: string,
          params?: EventListParams
        ): AsyncIterable<BetaManagedAgentsSessionEvent>;
      };
    };
  };
};

export type AnthropicManagedSessionAgentRuntimeOptions = {
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly agentId: string;
  readonly environmentId: string;
  readonly rawLogger?: JsonlLogger;
  readonly client?: ManagedSessionsClient;
  readonly pollIntervalMs?: number;
  readonly maxPolls?: number;
};

export class AnthropicManagedSessionAgentRuntime implements AgentRuntime {
  public readonly name = "anthropic-managed-sessions";
  private readonly client: ManagedSessionsClient;
  private readonly agentId: string;
  private readonly environmentId: string;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly pollIntervalMs: number;
  private readonly maxPolls: number;

  public constructor(options: AnthropicManagedSessionAgentRuntimeOptions) {
    this.client =
      options.client ??
      new Anthropic({
        apiKey: options.apiKey,
        baseURL: options.baseUrl
      });
    this.agentId = options.agentId;
    this.environmentId = options.environmentId;
    this.rawLogger = options.rawLogger;
    this.pollIntervalMs = options.pollIntervalMs ?? 500;
    this.maxPolls = options.maxPolls ?? 60;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const sessionId = request.sessionId ?? (await this.createSession(request));
    const since = new Date();
    const workspaceContext = await buildWorkspaceContext({
      workspacePath: request.workspacePath,
      query: request.text
    });
    const userText = buildGroundedUserMessage(request, workspaceContext);

    await this.rawLogger?.write({
      type: "llm.session.request",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      request: {
        eventType: "user.message",
        text: userText
      }
    });

    await this.client.beta.sessions.events.send(sessionId, {
      events: [
        {
          type: "user.message",
          content: [
            {
              type: "text",
              text: userText
            }
          ]
        }
      ]
    });

    const text = await this.waitForTurnText(sessionId, since);

    await this.rawLogger?.write({
      type: "llm.session.response",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      extractedText: text
    });

    return { text, sessionId };
  }

  public async disposeSession(sessionId: string): Promise<void> {
    await this.client.beta.sessions.archive(sessionId);
    await this.rawLogger?.write({
      type: "llm.session.archived",
      runtime: this.name,
      sessionId
    });
  }

  private async createSession(request: AgentRequest): Promise<string> {
    const session = await this.client.beta.sessions.create({
      agent: this.agentId,
      environment_id: this.environmentId,
      metadata: {
        userId: request.user.id,
        workspacePath: request.workspacePath
      },
      title: `Workspace chat: ${request.user.id}`
    });

    await this.rawLogger?.write({
      type: "llm.session.created",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId: session.id
    });

    return session.id;
  }

  private async waitForTurnText(sessionId: string, since: Date): Promise<string> {
    const collected: string[] = [];

    for (let attempt = 0; attempt < this.maxPolls; attempt += 1) {
      for await (const event of this.client.beta.sessions.events.list(sessionId, {
        order: "asc",
        "created_at[gte]": since.toISOString()
      })) {
        if (event.type === "agent.message") {
          collected.push(...event.content.map((block) => block.text));
        }

        if (event.type === "session.status_idle") {
          return formatIdleResponse(event.stop_reason.type, collected);
        }

        if (event.type === "session.error") {
          throw new Error(`Session failed: ${event.error.message}`);
        }
      }

      await sleep(this.pollIntervalMs);
    }

    throw new Error("Session did not become idle before the polling limit.");
  }
}

function buildGroundedUserMessage(request: AgentRequest, workspaceContext: string): string {
  return [
    "You answer questions about the selected workspace or knowledge base.",
    "Answer concisely in the same language as the user.",
    "Treat phrases like current project, this project, system architecture, or business architecture as referring to the selected workspace content, not this assistant service.",
    "Use only the provided workspace context when answering questions.",
    "Do not infer product domain from the project name.",
    "Do not describe the assistant runtime, message channels, model provider, test page, or implementation unless the user explicitly asks how this assistant is built or tested.",
    "If the context is insufficient, say what is missing instead of guessing.",
    `Current workspace path: ${request.workspacePath}`,
    "",
    "Workspace context:",
    workspaceContext,
    "",
    "User question:",
    request.text
  ].join("\n");
}

function formatIdleResponse(stopReason: string, collected: readonly string[]): string {
  const text = collected.join("\n").trim();
  if (stopReason === "end_turn") {
    return text.length > 0 ? text : "Agent finished without returning text content.";
  }

  if (stopReason === "requires_action") {
    return text.length > 0 ? text : "Agent requires action before it can continue.";
  }

  if (stopReason === "retries_exhausted") {
    return text.length > 0 ? text : "Agent stopped because the retry budget was exhausted.";
  }

  return text.length > 0 ? text : `Agent stopped with reason: ${stopReason}.`;
}

async function sleep(milliseconds: number): Promise<void> {
  if (milliseconds <= 0) {
    return;
  }
  await new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}
