import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlock, Message, MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { buildWorkspaceContext } from "./workspaceContext.js";

type AnthropicCompatibleClient = {
  readonly messages: {
    create(params: MessageCreateParamsNonStreaming): Promise<Message>;
  };
};

export type AnthropicCompatibleAgentRuntimeOptions = {
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly modelId: string;
  readonly maxTokens?: number;
  readonly rawLogger?: JsonlLogger;
  readonly client?: AnthropicCompatibleClient;
};

export class AnthropicCompatibleAgentRuntime implements AgentRuntime {
  public readonly name = "anthropic-compatible";
  private readonly client: AnthropicCompatibleClient;
  private readonly modelId: string;
  private readonly maxTokens: number;
  private readonly rawLogger: JsonlLogger | undefined;

  public constructor(options: AnthropicCompatibleAgentRuntimeOptions) {
    this.client =
      options.client ??
      new Anthropic({
        apiKey: options.apiKey,
        baseURL: options.baseUrl
      });
    this.modelId = options.modelId;
    this.maxTokens = options.maxTokens ?? 1024;
    this.rawLogger = options.rawLogger;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const workspaceContext = await buildWorkspaceContext({
      workspacePath: request.workspacePath,
      query: request.text
    });
    const createParams: MessageCreateParamsNonStreaming = {
      model: this.modelId,
      max_tokens: this.maxTokens,
      system: [
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
        workspaceContext
      ].join("\n"),
      messages: [
        ...(request.history ?? []).map((message) => ({
          role: message.role,
          content: message.content
        })),
        {
          role: "user",
          content: request.text
        }
      ]
    };

    await this.rawLogger?.write({
      type: "llm.request",
      runtime: this.name,
      modelId: this.modelId,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      request: createParams
    });

    try {
      const message = await this.client.messages.create(createParams);
      const text = extractText(message.content);

      await this.rawLogger?.write({
        type: "llm.response",
        runtime: this.name,
        modelId: this.modelId,
        userId: request.user.id,
        response: message,
        extractedText: text
      });

      return { text };
    } catch (error) {
      await this.rawLogger?.write({
        type: "llm.error",
        runtime: this.name,
        modelId: this.modelId,
        userId: request.user.id,
        error: serializeError(error)
      });
      throw error;
    }
  }

  public disposeSession(): Promise<void> {
    return Promise.resolve();
  }
}

function extractText(blocks: readonly ContentBlock[]): string {
  const text = blocks
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  return text.length > 0 ? text : "Model did not return text content.";
}

function serializeError(error: unknown): Record<string, unknown> {
  if (!(error instanceof Error)) {
    return { message: String(error) };
  }

  return {
    name: error.name,
    message: error.message,
    stack: error.stack
  };
}
