import { query } from "@anthropic-ai/claude-agent-sdk";
import type { PermissionResult, SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime, StoredRoleConfig } from "../core/contracts.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import {
  autoAllowedToolsForRole,
  availableToolsForRole,
  decideToolPermission,
  disallowedToolsForRole,
  type ToolPermissionDecider,
} from "./claudeToolPolicy.js";
import {
  forwardAssistantMessageEvents,
  forwardStreamEvent,
  isAssistantMessage,
  isResultMessage,
} from "./claudeSdkMessageMapper.js";
import { buildWorkspacePrompt } from "./workspacePromptBuilder.js";

export type ClaudeSdkAgentRuntimeOptions = {
  readonly roleConfig: StoredRoleConfig;
  readonly rawLogger?: JsonlLogger;
  readonly model?: string;
  readonly toolPermissionDecider?: ToolPermissionDecider;
};

export class ClaudeSdkAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly model: string | undefined;
  private readonly toolPermissionDecider: ToolPermissionDecider | undefined;

  public constructor(options: ClaudeSdkAgentRuntimeOptions) {
    this.name = `claude-sdk-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.rawLogger = options.rawLogger;
    this.model = options.model;
    this.toolPermissionDecider = options.toolPermissionDecider;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const prompt = await buildWorkspacePrompt(request);
    const queryOptions: Record<string, unknown> = {
      cwd: request.workspacePath,
      tools: availableToolsForRole(this.roleConfig),
      allowedTools: autoAllowedToolsForRole(this.roleConfig),
      disallowedTools: disallowedToolsForRole(this.roleConfig),
      permissionMode: this.roleConfig.permissionMode,
      allowDangerouslySkipPermissions: this.roleConfig.permissionMode === "bypassPermissions",
      systemPrompt: this.roleConfig.systemPrompt,
      includePartialMessages: true,
      canUseTool: (toolName: string, input: Record<string, unknown>) => this.canUseTool(toolName, input),
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

    const sdkOptions = buildSdkOptions(queryOptions);
    const startTime = Date.now();
    const stream = query({
      prompt,
      options: sdkOptions,
    });

    let finalText = "";
    let sessionId: string | undefined;

    for await (const message of stream) {
      if (isAssistantMessage(message)) {
        const assistantMsg = message as Extract<SDKMessage, { type: "assistant" }>;
        sessionId = assistantMsg.session_id;
        forwardAssistantMessageEvents(assistantMsg, request, this.roleConfig);
      } else if (isResultMessage(message)) {
        const resultMsg = message as Extract<SDKMessage, { type: "result" }>;
        const resultData = resultMsg as unknown as Record<string, unknown>;
        sessionId = resultData["session_id"] as string | undefined;
        const subtype = resultData["subtype"] as string | undefined;
        if (subtype === "success") {
          finalText = (resultData["result"] as string) || "";
        } else {
          const errors = resultData["errors"] as string[] | undefined;
          finalText = `Agent error: ${errors?.[0] ?? "Unknown error"}`;
        }
      } else if (message.type === "stream_event") {
        forwardStreamEvent({ ...message }, request);
      }
    }

    await this.rawLogger?.write({
      type: "llm.response",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      extractedText: finalText,
      durationMs: Date.now() - startTime,
    });

    return {
      text: finalText || "Agent completed without text output.",
      ...(sessionId !== undefined ? { sessionId } : {}),
    };
  }

  public async disposeSession(): Promise<void> {
    // SDK manages sessions internally; no explicit disposal needed.
  }

  private async canUseTool(toolName: string, input: Record<string, unknown>): Promise<PermissionResult> {
    return decideToolPermission(this.roleConfig, toolName, input, this.toolPermissionDecider);
  }
}

function buildSdkOptions(opts: Record<string, unknown>): NonNullable<Parameters<typeof query>[0]["options"]> {
  return opts;
}
