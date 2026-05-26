import { query } from "@anthropic-ai/claude-agent-sdk";
import type { PermissionResult, SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport, StoredRoleConfig } from "../core/contracts.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import {
  autoAllowedToolsForRole,
  availableToolsForRole,
  canUseConfiguredTool,
  decideToolPermission,
  disallowedToolsForRole,
  type ToolPermissionDecider,
} from "./claudeToolPolicy.js";
import {
  forwardAssistantMessageEvents,
  forwardStreamEvent,
  forwardSystemMessageEvents,
  isAssistantMessage,
  isResultMessage,
  isSystemMessage,
} from "./claudeSdkMessageMapper.js";
import { buildWorkspacePrompt } from "./workspacePromptBuilder.js";

export type ClaudeSdkAgentRuntimeOptions = {
  readonly roleConfig: StoredRoleConfig;
  readonly contextConfig?: ContextConfig | undefined;
  readonly rawLogger?: JsonlLogger;
  readonly model?: string;
  readonly toolPermissionDecider?: ToolPermissionDecider;
};

export class ClaudeSdkAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly contextConfig: ContextConfig;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly model: string | undefined;
  private readonly toolPermissionDecider: ToolPermissionDecider | undefined;

  private static readonly DEFAULT_CONTEXT_CONFIG: ContextConfig = {
    autoCompactEnabled: true,
    autoCompactWindow: 150_000,
    workspaceMaxChars: 8_000,
    historyMaxMessages: 20,
  };

  public constructor(options: ClaudeSdkAgentRuntimeOptions) {
    this.name = `claude-sdk-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.contextConfig = options.contextConfig ?? ClaudeSdkAgentRuntime.DEFAULT_CONTEXT_CONFIG;
    this.rawLogger = options.rawLogger;
    this.model = options.model;
    this.toolPermissionDecider = options.toolPermissionDecider;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const prompt = await buildWorkspacePrompt(request, this.contextConfig.workspaceMaxChars);
    const queryOptions: Record<string, unknown> = {
      cwd: request.workspacePath,
      tools: availableToolsForRole(this.roleConfig),
      allowedTools: autoAllowedToolsForRole(this.roleConfig),
      disallowedTools: disallowedToolsForRole(this.roleConfig),
      permissionMode: this.roleConfig.permissionMode,
      allowDangerouslySkipPermissions: this.roleConfig.permissionMode === "bypassPermissions",
      systemPrompt: this.roleConfig.systemPrompt,
      includePartialMessages: true,
      canUseTool: (toolName: string, input: Record<string, unknown>) => this.canUseTool(toolName, input, request.workspacePath),
    };
    if (this.roleConfig.maxTurns !== undefined) {
      queryOptions["maxTurns"] = this.roleConfig.maxTurns;
    }
    if (this.model !== undefined) {
      queryOptions["model"] = this.model;
    }
    if (request.sessionId !== undefined) {
      queryOptions["resume"] = request.sessionId;
    }
    if (this.contextConfig.autoCompactEnabled) {
      queryOptions["settings"] = {
        autoCompactEnabled: true,
        autoCompactWindow: this.contextConfig.autoCompactWindow,
      };
    }
    if (request.onCompact !== undefined) {
      queryOptions["hooks"] = {
        PostCompact: [{
          hooks: [async (input: Record<string, unknown>) => {
            const summary = input["compact_summary"] as string;
            await request.onCompact?.(summary);
          }],
        }],
      };
    }

    const sdkOptions = buildSdkOptions(queryOptions);
    const startTime = Date.now();
    await this.rawLogger?.write({
      type: "llm.request",
      operation: "agent_runtime",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId: request.sessionId,
      prompt,
      options: serializeQueryOptions(queryOptions),
    });
    const stream = query({
      prompt,
      options: sdkOptions,
    });

    let finalText = "";
    let sessionId: string | undefined;
    let contextUsage: ContextUsageReport | undefined;
    let sdkResult: Record<string, unknown> | undefined;

    try {
      for await (const message of stream) {
        if (isAssistantMessage(message)) {
          const assistantMsg = message as Extract<SDKMessage, { type: "assistant" }>;
          sessionId = assistantMsg.session_id;
          await this.logToolEvents(assistantMsg, request, sessionId);
          forwardAssistantMessageEvents(assistantMsg, request, this.roleConfig);
        } else if (isResultMessage(message)) {
          const resultMsg = message as Extract<SDKMessage, { type: "result" }>;
          const resultData = resultMsg as unknown as Record<string, unknown>;
          sdkResult = resultData;
          sessionId = resultData["session_id"] as string | undefined;
          const subtype = resultData["subtype"] as string | undefined;
          if (subtype === "success") {
            finalText = (resultData["result"] as string) || "";
            contextUsage = extractContextUsage(resultData["usage"], this.contextConfig.autoCompactWindow);
          } else {
            const errors = resultData["errors"] as string[] | undefined;
            finalText = `Agent error: ${errors?.[0] ?? "Unknown error"}`;
          }
        } else if (message.type === "stream_event") {
          forwardStreamEvent({ ...message }, request);
        } else if (isSystemMessage(message)) {
          const compactData = forwardSystemMessageEvents(message, request);
          if (compactData !== undefined) {
            await this.rawLogger?.write({
              type: "llm.compact",
              runtime: this.name,
              userId: request.user.id,
              workspacePath: request.workspacePath,
              sessionId: compactData.sessionId ?? sessionId,
              trigger: compactData.trigger,
              preTokens: compactData.preTokens,
              ...(compactData.postTokens !== undefined ? { postTokens: compactData.postTokens } : {}),
              ...(compactData.durationMs !== undefined ? { durationMs: compactData.durationMs } : {}),
            });
          }
        }
      }
    } catch (error) {
      await this.rawLogger?.write({
        type: "llm.error",
        operation: "agent_runtime",
        runtime: this.name,
        userId: request.user.id,
        workspacePath: request.workspacePath,
        sessionId,
        error: formatUnknownError(error),
        durationMs: Date.now() - startTime,
      });
      throw error;
    }

    await this.rawLogger?.write({
      type: "llm.response",
      operation: "agent_runtime",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      response: serializeSdkResult(sdkResult),
      extractedText: finalText,
      durationMs: Date.now() - startTime,
    });

    return {
      text: finalText || "Agent completed without text output.",
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(contextUsage !== undefined ? { contextUsage } : {}),
    };
  }

  public async disposeSession(): Promise<void> {
    // SDK manages sessions internally; no explicit disposal needed.
  }

  private async logToolEvents(
    message: Extract<SDKMessage, { type: "assistant" }>,
    request: AgentRequest,
    sessionId: string | undefined,
  ): Promise<void> {
    const msgData = message as unknown as { message?: { content?: unknown[] } };
    const content = msgData.message?.content ?? [];

    for (const rawBlock of content) {
      const block = rawBlock as Record<string, unknown>;
      const blockType = block["type"];

      if (blockType === "tool_use") {
        const toolName = block["name"] as string;
        await this.rawLogger?.write({
          type: "agent.tool_call",
          runtime: this.name,
          userId: request.user.id,
          workspacePath: request.workspacePath,
          sessionId,
          userInput: request.text,
          toolName,
          toolUseId: block["id"],
          permittedByRole: canUseConfiguredTool(this.roleConfig, toolName),
          input: block["input"] ?? {},
        });
      } else if (blockType === "tool_result") {
        await this.rawLogger?.write({
          type: "agent.tool_result",
          runtime: this.name,
          userId: request.user.id,
          workspacePath: request.workspacePath,
          sessionId,
          userInput: request.text,
          toolUseId: block["tool_use_id"],
          isError: block["is_error"] === true,
          result: extractToolResultText(block),
        });
      }
    }
  }

  private async canUseTool(toolName: string, input: Record<string, unknown>, workspacePath: string): Promise<PermissionResult> {
    return decideToolPermission(this.roleConfig, toolName, input, this.toolPermissionDecider, workspacePath);
  }
}

function buildSdkOptions(opts: Record<string, unknown>): NonNullable<Parameters<typeof query>[0]["options"]> {
  return opts;
}

function serializeQueryOptions(queryOptions: Record<string, unknown>): Record<string, unknown> {
  const serializableKeys = [
    "cwd",
    "tools",
    "allowedTools",
    "disallowedTools",
    "permissionMode",
    "allowDangerouslySkipPermissions",
    "systemPrompt",
    "includePartialMessages",
    "maxTurns",
    "model",
    "resume",
    "settings",
  ];

  return Object.fromEntries(
    serializableKeys
      .filter((key) => queryOptions[key] !== undefined)
      .map((key) => [key, queryOptions[key]])
  );
}

function serializeSdkResult(result: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (result === undefined) return undefined;

  return {
    subtype: result["subtype"],
    sessionId: result["session_id"],
    result: result["result"],
    errors: result["errors"],
    usage: result["usage"],
  };
}

function extractToolResultText(block: Record<string, unknown>): string {
  const content = block["content"];
  if (!Array.isArray(content)) return "";

  return content
    .map((part: unknown) => {
      if (typeof part === "string") return part;
      if (part !== null && typeof part === "object") {
        const record = part as Record<string, unknown>;
        return typeof record["text"] === "string" ? record["text"] : "";
      }
      return "";
    })
    .join("");
}

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function extractContextUsage(usage: unknown, contextWindow: number): ContextUsageReport | undefined {
  if (usage === null || usage === undefined || typeof usage !== "object") return undefined;
  const u = usage as Record<string, unknown>;
  const inputTokens = u["input_tokens"];
  const outputTokens = u["output_tokens"];
  if (typeof inputTokens !== "number" || typeof outputTokens !== "number") return undefined;

  const cacheReadTokens = u["cache_read_input_tokens"];
  const cacheCreationTokens = u["cache_creation_input_tokens"];
  const usagePercent = contextWindow > 0 ? Math.round((inputTokens / contextWindow) * 100) : 0;

  return {
    inputTokens,
    outputTokens,
    ...(typeof cacheReadTokens === "number" ? { cacheReadTokens } : {}),
    ...(typeof cacheCreationTokens === "number" ? { cacheCreationTokens } : {}),
    contextWindow,
    usagePercent,
  };
}
