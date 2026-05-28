import { query } from "@anthropic-ai/claude-agent-sdk";
import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import {
  arrayField,
  isRecord,
  recordField,
  stringArrayField,
  stringField
} from "../../core/unknownRecord.js";
import type { ContextConfig } from "../../config/runtimeConfig.js";
import { DEFAULT_CONTEXT_CONFIG } from "../../config/runtimeConfig.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import {
  autoAllowedToolsForRole,
  availableToolsForRole,
  decideToolPermission,
  disallowedToolsForRole,
  toClaudePermissionResult,
  type ToolPermissionDecider
} from "./claudeToolPolicy.js";
import { isAssistantMessage, isResultMessage, isSystemMessage } from "./claudeSdkMessageMapper.js";
import { buildWorkspacePrompt } from "../shared/workspacePromptBuilder.js";
import {
  forwardAssistantContentEvents,
  forwardStreamEvent,
  forwardSystemContentEvents,
  logToolEventsFromContent
} from "../shared/sdkMessageMapper.js";
import {
  extractContextUsage,
  formatUnknownError,
  serializeQueryOptions,
  serializeSdkResult
} from "../shared/sdkRuntimeHelpers.js";

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

  public constructor(options: ClaudeSdkAgentRuntimeOptions) {
    this.name = `claude-sdk-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.contextConfig = options.contextConfig ?? DEFAULT_CONTEXT_CONFIG;
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
      canUseTool: (toolName: string, input: Record<string, unknown>) =>
        this.canUseTool(toolName, input, request.workspacePath)
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
        autoCompactWindow: this.contextConfig.autoCompactWindow
      };
    }
    queryOptions["settingSources"] = this.roleConfig.settingSources ?? ["project", "local"];
    if (this.roleConfig.skills !== undefined) {
      queryOptions["skills"] = this.roleConfig.skills;
    }
    if (request.onCompact !== undefined) {
      queryOptions["hooks"] = {
        PostCompact: [
          {
            hooks: [
              async (input: Record<string, unknown>) => {
                const summary = stringField(input, "compact_summary");
                if (summary !== undefined) {
                  await request.onCompact?.(summary);
                }
              }
            ]
          }
        ]
      };
    }

    const sdkOptions = queryOptions as NonNullable<Parameters<typeof query>[0]["options"]>;
    const startTime = Date.now();
    await this.rawLogger?.write({
      type: "llm.request",
      operation: "agent_runtime",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId: request.sessionId,
      prompt,
      options: serializeQueryOptions(queryOptions)
    });
    const stream = query({
      prompt,
      options: sdkOptions
    });

    let finalText = "";
    let sessionId: string | undefined;
    let contextUsage: ContextUsageReport | undefined;
    let sdkResult: Record<string, unknown> | undefined;

    try {
      for await (const message of stream) {
        if (isAssistantMessage(message)) {
          sessionId = message.session_id;
          const msgData = isRecord(message) ? recordField(message, "message") : undefined;
          const content = msgData === undefined ? [] : (arrayField(msgData, "content") ?? []);
          await logToolEventsFromContent(
            content,
            this.name,
            request,
            sessionId,
            this.roleConfig,
            this.rawLogger
          );
          forwardAssistantContentEvents(content, request, this.roleConfig);
        } else if (isResultMessage(message)) {
          const resultData: Record<string, unknown> = isRecord(message) ? message : {};
          sdkResult = resultData;
          sessionId = stringField(resultData, "session_id");
          const subtype = stringField(resultData, "subtype");
          if (subtype === "success") {
            finalText = stringField(resultData, "result") ?? "";
            contextUsage = extractContextUsage(
              resultData["usage"],
              this.contextConfig.autoCompactWindow
            );
          } else {
            const errors = stringArrayField(resultData, "errors");
            finalText = `Agent error: ${errors?.[0] ?? "Unknown error"}`;
          }
        } else if (message.type === "stream_event") {
          forwardStreamEvent({ ...message }, request);
        } else if (isSystemMessage(message)) {
          const compactData = isRecord(message)
            ? forwardSystemContentEvents(message, request)
            : undefined;
          if (compactData !== undefined) {
            await this.rawLogger?.write({
              type: "llm.compact",
              runtime: this.name,
              userId: request.user.id,
              workspacePath: request.workspacePath,
              sessionId: compactData.sessionId ?? sessionId,
              trigger: compactData.trigger,
              preTokens: compactData.preTokens,
              ...(compactData.postTokens !== undefined
                ? { postTokens: compactData.postTokens }
                : {}),
              ...(compactData.durationMs !== undefined
                ? { durationMs: compactData.durationMs }
                : {})
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
        durationMs: Date.now() - startTime
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
      durationMs: Date.now() - startTime
    });

    return {
      text: finalText.length > 0 ? finalText : "Agent completed without text output.",
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(contextUsage !== undefined ? { contextUsage } : {})
    };
  }

  public disposeSession(sessionId: string): Promise<void> {
    // SDK manages sessions internally; no explicit disposal needed.
    void sessionId;
    return Promise.resolve();
  }

  private async canUseTool(
    toolName: string,
    input: Record<string, unknown>,
    workspacePath: string
  ): Promise<PermissionResult> {
    const result = decideToolPermission(
      this.roleConfig,
      toolName,
      input,
      this.toolPermissionDecider,
      workspacePath
    );
    return toClaudePermissionResult(await result);
  }
}
