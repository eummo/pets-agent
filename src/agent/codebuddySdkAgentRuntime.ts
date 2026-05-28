import type { Message, PermissionResult } from "@tencent-ai/agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport } from "./index.js";
import type { StoredRoleConfig } from "../auth/index.js";
import { arrayField, booleanField, isRecord, numberField, recordField, stringArrayField, stringField } from "../core/unknownRecord.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import type { ResolvedAgentSdkConfig } from "../config/llmConfig.js";
import {
  autoAllowedToolsForRole,
  availableToolsForRole,
  canUseConfiguredTool,
  decideToolPermission,
  disallowedToolsForRole,
  type ToolPermissionDecider,
} from "./toolPolicy.js";
import { buildWorkspacePrompt } from "./workspacePromptBuilder.js";
import {
  extractContextUsage,
  extractToolResultText,
  formatUnknownError,
  serializeQueryOptions,
  serializeSdkResult,
} from "./sdkRuntimeHelpers.js";

export type CodebuddySdkAgentRuntimeOptions = {
  readonly roleConfig: StoredRoleConfig;
  readonly agentSdkConfig: ResolvedAgentSdkConfig;
  readonly contextConfig?: ContextConfig | undefined;
  readonly rawLogger?: JsonlLogger;
  readonly model?: string;
  readonly toolPermissionDecider?: ToolPermissionDecider;
};

export class CodebuddySdkAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly agentSdkConfig: ResolvedAgentSdkConfig;
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

  public constructor(options: CodebuddySdkAgentRuntimeOptions) {
    this.name = `codebuddy-sdk-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.agentSdkConfig = options.agentSdkConfig;
    this.contextConfig = options.contextConfig ?? CodebuddySdkAgentRuntime.DEFAULT_CONTEXT_CONFIG;
    this.rawLogger = options.rawLogger;
    this.model = options.model;
    this.toolPermissionDecider = options.toolPermissionDecider;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const { query } = await import("@tencent-ai/agent-sdk");
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
      env: {
        CODEBUDDY_API_KEY: this.agentSdkConfig.apiKey,
      },
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
    queryOptions["settingSources"] = this.roleConfig.settingSources ?? ["project", "local"];
    if (this.roleConfig.skills !== undefined) {
      queryOptions["skills"] = this.roleConfig.skills;
    }
    if (request.onCompact !== undefined) {
      queryOptions["hooks"] = {
        PostCompact: [{
          hooks: [async (input: Record<string, unknown>) => {
            const summary = stringField(input, "compact_summary");
            if (summary !== undefined) {
              await request.onCompact?.(summary);
            }
          }],
        }],
      };
    }

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
      options: queryOptions,
    });

    let finalText = "";
    let sessionId: string | undefined;
    let contextUsage: ContextUsageReport | undefined;
    let sdkResult: Record<string, unknown> | undefined;

    try {
      for await (const message of stream) {
        if (message.type === "assistant") {
          sessionId = message.session_id;
          await this.logToolEvents(message, request, sessionId);
          forwardAssistantMessageEvents(message, request, this.roleConfig);
        } else if (message.type === "result") {
          const resultData: Record<string, unknown> = isRecord(message) ? message : {};
          sdkResult = resultData;
          sessionId = stringField(resultData, "session_id");
          const subtype = stringField(resultData, "subtype");
          if (subtype === "success") {
            finalText = stringField(resultData, "result") ?? "";
            contextUsage = extractContextUsage(resultData["usage"], this.contextConfig.autoCompactWindow);
          } else {
            const errors = stringArrayField(resultData, "errors");
            finalText = `Agent error: ${errors?.[0] ?? "Unknown error"}`;
          }
        } else if (message.type === "stream_event") {
          if (isRecord(message)) {
            forwardStreamEvent(message, request);
          }
        } else if (message.type === "system") {
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
      text: finalText.length > 0 ? finalText : "Agent completed without text output.",
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(contextUsage !== undefined ? { contextUsage } : {}),
    };
  }

  public async disposeSession(): Promise<void> {
    // SDK manages sessions internally; no explicit disposal needed.
  }

  private async canUseTool(toolName: string, input: Record<string, unknown>, workspacePath: string): Promise<PermissionResult> {
    const result = await decideToolPermission(this.roleConfig, toolName, input, this.toolPermissionDecider, workspacePath);
    if (result.behavior === "allow") {
      return { behavior: "allow", updatedInput: input };
    }
    return {
      behavior: "deny",
      message: result.message ?? `Tool ${toolName} denied.`,
    };
  }

  private async logToolEvents(
    message: Extract<Message, { type: "assistant" }>,
    request: AgentRequest,
    sessionId: string | undefined,
  ): Promise<void> {
    const content = message.message.content;

    for (const rawBlock of content) {
      if (!isRecord(rawBlock)) continue;

      const block = rawBlock;
      const blockType = stringField(block, "type");

      if (blockType === "tool_use") {
        const toolName = stringField(block, "name");
        const toolUseId = stringField(block, "id");
        const input = recordField(block, "input") ?? {};
        if (toolName === undefined) continue;

        await this.rawLogger?.write({
          type: "agent.tool_call",
          runtime: this.name,
          userId: request.user.id,
          workspacePath: request.workspacePath,
          sessionId,
          userInput: request.text,
          toolName,
          toolUseId,
          permittedByRole: canUseConfiguredTool(this.roleConfig, toolName),
          input,
        });
      } else if (blockType === "tool_result") {
        const toolUseId = stringField(block, "tool_use_id");
        const isError = booleanField(block, "is_error") ?? false;
        await this.rawLogger?.write({
          type: "agent.tool_result",
          runtime: this.name,
          userId: request.user.id,
          workspacePath: request.workspacePath,
          sessionId,
          userInput: request.text,
          toolUseId,
          isError,
          result: extractToolResultText(block),
        });
      }
    }
  }
}
// ── Message Mapping (Codebuddy SDK) ──────────────────────────────────────────
// The Codebuddy SDK message types are structurally identical to Claude SDK
// messages, so the mapping logic is the same.

function forwardAssistantMessageEvents(
  msg: Extract<Message, { type: "assistant" }>,
  request: AgentRequest,
  roleConfig: StoredRoleConfig,
): void {
  const content = msg.message.content;
  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) continue;

    const block = rawBlock;
    const blockType = stringField(block, "type");

    if (blockType === "tool_use") {
      const toolName = stringField(block, "name");
      const toolUseId = stringField(block, "id");
      if (toolName === undefined || toolUseId === undefined) continue;

      if (!canUseConfiguredTool(roleConfig, toolName)) {
        const message = `Tool ${toolName} is not permitted for role ${roleConfig.name}.`;
        request.stream?.({ type: "error", message });
        void request.progress?.({
          stage: "agent.error",
          message,
          data: { toolName },
        });
        continue;
      }

      const input = recordField(block, "input") ?? {};
      request.stream?.({
        type: "tool_use_start",
        toolName,
        toolUseId,
        input,
      });
      void request.progress?.({
        stage: "agent.tool_use_start",
        message: `${toolName}: ${JSON.stringify(input).slice(0, 100)}`,
        data: { toolUseId, toolName },
      });
    } else if (blockType === "tool_result") {
      const toolUseId = stringField(block, "tool_use_id");
      if (toolUseId === undefined) continue;

      const isError = booleanField(block, "is_error") ?? false;
      const resultText = (arrayField(block, "content") ?? [])
        .map((p: unknown) => {
          if (typeof p === "string") return p;
          if (!isRecord(p)) return "";
          return stringField(p, "text") ?? "";
        })
        .join("");
      request.stream?.({
        type: "tool_use_result",
        toolUseId,
        result: resultText,
        ...(isError ? { isError: true } : {}),
      });
    }
  }
}

function forwardStreamEvent(event: Record<string, unknown>, request: AgentRequest): void {
  const e = recordField(event, "event");
  if (e === undefined) return;

  const eventType = stringField(e, "type");

  if (eventType === "content_block_delta") {
    const delta = recordField(e, "delta");
    if (delta === undefined) return;

    const deltaType = stringField(delta, "type");
    const text = stringField(delta, "text");
    const thinking = stringField(delta, "thinking");
    if (deltaType === "text_delta" && text !== undefined) {
      request.stream?.({ type: "text_delta", text });
    } else if (deltaType === "thinking_delta" && thinking !== undefined) {
      request.stream?.({ type: "thinking", text: thinking });
    }
  }
}

type CompactBoundaryData = {
  readonly sessionId?: string;
  readonly trigger: "manual" | "auto";
  readonly preTokens: number;
  readonly postTokens?: number;
  readonly durationMs?: number;
};

function forwardSystemMessageEvents(
  msg: Message,
  request: AgentRequest,
): CompactBoundaryData | undefined {
  if (!isRecord(msg)) return undefined;

  const data = msg;
  const subtype = stringField(data, "subtype");

  if (subtype === "status") {
    const status = stringField(data, "status");
    if (status === "compacting") {
      request.stream?.({ type: "compact_start" });
    }
    return undefined;
  }

  if (subtype === "compact_boundary") {
    const metadata = recordField(data, "compact_metadata");
    if (metadata === undefined) return undefined;

    const trigger = stringField(metadata, "trigger");
    const preTokens = numberField(metadata, "pre_tokens");
    if ((trigger !== "manual" && trigger !== "auto") || preTokens === undefined) {
      return undefined;
    }

    const sessionId = stringField(data, "session_id");
    const postTokens = numberField(metadata, "post_tokens");
    const durationMs = numberField(metadata, "duration_ms");
    const compactData: CompactBoundaryData = {
      ...(sessionId !== undefined ? { sessionId } : {}),
      trigger,
      preTokens,
      ...(postTokens !== undefined ? { postTokens } : {}),
      ...(durationMs !== undefined ? { durationMs } : {}),
    };

    request.stream?.({
      type: "compact_complete",
      preTokens: compactData.preTokens,
      ...(compactData.postTokens !== undefined ? { postTokens: compactData.postTokens } : {}),
      ...(compactData.durationMs !== undefined ? { durationMs: compactData.durationMs } : {}),
    });

    return compactData;
  }

  return undefined;
}
