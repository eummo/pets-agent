import { query } from "@anthropic-ai/claude-agent-sdk";
import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import { arrayField, isRecord, recordField } from "../../core/unknownRecord.js";
import type { ContextConfig } from "../../config/runtimeConfig.js";
import { DEFAULT_CONTEXT_CONFIG } from "../../config/runtimeConfig.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import {
  decideToolPermission,
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
  buildSdkQueryOptions,
  formatUnknownError,
  handleSdkResultMessage,
  logCompactEvent,
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
    const queryOptions = buildSdkQueryOptions({
      request,
      roleConfig: this.roleConfig,
      contextConfig: this.contextConfig,
      model: this.model,
      canUseTool: (toolName, input) => this.canUseTool(toolName, input, request.workspacePath)
    });

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
    let contextUsage = undefined;
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
          const outcome = handleSdkResultMessage(message, this.contextConfig.autoCompactWindow);
          sdkResult = outcome.sdkResult;
          sessionId = outcome.sessionId;
          finalText = outcome.finalText;
          contextUsage = outcome.contextUsage;
        } else if (message.type === "stream_event") {
          forwardStreamEvent({ ...message }, request);
        } else if (isSystemMessage(message)) {
          const compactData = isRecord(message)
            ? forwardSystemContentEvents(message, request)
            : undefined;
          if (compactData !== undefined) {
            await logCompactEvent(this.rawLogger, compactData, sessionId, this.name, request);
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
