import { readFile } from "node:fs/promises";
import path from "node:path";
import { query } from "@anthropic-ai/claude-agent-sdk";
import type { PermissionResult, SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentResponse, AgentRuntime, AgentStreamEvent } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";

// ─── Role Configuration ──────────────────────────────────────────────────────

export type RoleConfig = {
  readonly name: string;
  readonly allowedTools: readonly string[];
  readonly permissionMode: "auto" | "dontAsk" | "acceptEdits" | "bypassPermissions";
  readonly systemPrompt: string;
  readonly maxTurns?: number;
  readonly model?: string;
};

export type ToolPermissionDecider = (
  roleConfig: RoleConfig,
  toolName: string,
  input: Record<string, unknown>,
) => Promise<PermissionResult>;

const FILE_MUTATION_TOOLS = new Set(["Edit", "MultiEdit", "NotebookEdit", "Write"]);

export const REVIEWER_CONFIG: RoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Glob", "Grep", "Bash"],
  permissionMode: "dontAsk",
  systemPrompt: [
    "You are a knowledge-base assistant (文档助手).",
    "Answer questions about the selected workspace or knowledge base.",
    "Answer concisely in the same language as the user.",
    "Treat phrases like current project, this project, system architecture, or business architecture as referring to the selected workspace content, not this assistant service.",
    "Use only the provided workspace context when answering questions.",
    "Do not infer product domain from the project name.",
    "Do not describe the assistant runtime, message channels, model provider, test page, or implementation unless the user explicitly asks how this assistant is built or tested.",
    "Prefer Read, Glob, and Grep for inspection. Use Bash only for non-mutating inspection commands when those tools are insufficient.",
    "If the context is insufficient, say what is missing instead of guessing.",
  ].join("\n"),
  maxTurns: 20,
};

export const DEVELOPER_CONFIG: RoleConfig = {
  name: "developer",
  allowedTools: ["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
  permissionMode: "bypassPermissions",
  systemPrompt: [
    "You are a coding assistant (开发助手) that edits the selected workspace.",
    "Read and understand the codebase, then make the requested changes.",
    "After making changes, run verification commands (npm run check, npm test) to confirm correctness.",
    "Iterate until the task is complete and all checks pass.",
    "Use relative paths inside the selected workspace. Do not include absolute paths.",
    "Keep the change focused on the user's request.",
    "Answer concisely in the same language as the user.",
  ].join("\n"),
  maxTurns: 30,
};

// ─── Type Guards ─────────────────────────────────────────────────────────────

function isAssistantMessage(msg: SDKMessage): boolean {
  return msg.type === "assistant";
}

function isResultMessage(msg: SDKMessage): boolean {
  return msg.type === "result";
}

// ─── Runtime ─────────────────────────────────────────────────────────────────

export type ClaudeSdkAgentRuntimeOptions = {
  readonly roleConfig: RoleConfig;
  readonly rawLogger?: JsonlLogger;
  readonly model?: string;
  readonly toolPermissionDecider?: ToolPermissionDecider;
};

export class ClaudeSdkAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: RoleConfig;
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

    // SDK Options type has complex union; use typed helper for compatibility
    const sdkOptions = buildSdkOptions(queryOptions);
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
        this.processAssistantMessage(assistantMsg, request);
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
        this.processStreamEvent({ ...message }, request);
      }
    }

    await this.rawLogger?.write({
      type: "llm.response",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      extractedText: finalText,
    });

    return {
      text: finalText || "Agent completed without text output.",
      ...(sessionId !== undefined ? { sessionId } : {}),
    };
  }

  public async disposeSession(): Promise<void> {
    // SDK manages sessions internally; no explicit disposal needed
  }

  private async canUseTool(toolName: string, input: Record<string, unknown>): Promise<PermissionResult> {
    if (!this.roleConfig.allowedTools.includes(toolName)) {
      return denyTool(this.roleConfig.name, toolName);
    }

    if (FILE_MUTATION_TOOLS.has(toolName) && !roleCanUseFileMutationTools(this.roleConfig)) {
      return denyTool(this.roleConfig.name, toolName);
    }

    if (toolName === "Bash" && !roleCanUseFileMutationTools(this.roleConfig)) {
      return this.toolPermissionDecider?.(this.roleConfig, toolName, input)
        ?? denyTool(this.roleConfig.name, toolName);
    }

    return { behavior: "allow" };
  }

  private processAssistantMessage(
    msg: Extract<SDKMessage, { type: "assistant" }>,
    request: AgentRequest,
  ): void {
    const msgData = msg as unknown as { message?: { content?: unknown[] } };
    const content = msgData.message?.content ?? [];
    for (const rawBlock of content) {
      const block = rawBlock as Record<string, unknown>;
      const blockType = block["type"] as string;

      if (blockType === "tool_use") {
        const toolName = block["name"] as string;
        if (!canUseTool(this.roleConfig, toolName)) {
          const message = `Tool ${toolName} is not permitted for role ${this.roleConfig.name}.`;
          request.stream?.({ type: "error", message });
          void request.progress?.({
            stage: "agent.error",
            message,
            data: { toolName },
          });
          continue;
        }

        const input = (block["input"] as Record<string, unknown> | null) ?? {};
        const toolEvent: AgentStreamEvent = {
          type: "tool_use_start",
          toolName,
          toolUseId: block["id"] as string,
          input,
        };
        request.stream?.(toolEvent);
        void request.progress?.({
          stage: "agent.tool_use_start",
          message: `${toolName}: ${JSON.stringify(block["input"]).slice(0, 100)}`,
          data: { toolUseId: block["id"], toolName },
        });
      } else if (blockType === "tool_result") {
        const isError = block["is_error"] === true;
        const contentParts = block["content"] as unknown[] | undefined;
        const resultText = (contentParts ?? [])
          .map((p: unknown) => {
            const part = p as Record<string, unknown>;
            return typeof part["text"] === "string" ? part["text"] : "";
          })
          .join("");
        request.stream?.({
          type: "tool_use_result",
          toolUseId: block["tool_use_id"] as string,
          result: resultText,
          ...(isError ? { isError: true } : {}),
        });
      }
      // text_delta and thinking are emitted via stream_event processing;
      // do not re-emit from assistant messages to avoid duplicates.
    }
  }

  private processStreamEvent(event: Record<string, unknown>, request: AgentRequest): void {
    const e = event["event"] as Record<string, unknown> | undefined;
    if (e === undefined) return;

    const eventType = e["type"] as string | undefined;

    if (eventType === "content_block_delta") {
      const delta = e["delta"] as Record<string, unknown> | undefined;
      const deltaType = delta?.["type"] as string | undefined;
      if (deltaType === "text_delta" && typeof delta?.["text"] === "string") {
        request.stream?.({ type: "text_delta", text: delta["text"] });
      } else if (deltaType === "thinking_delta" && typeof delta?.["thinking"] === "string") {
        request.stream?.({ type: "thinking", text: delta["thinking"] });
      }
    }
    // tool_use events are handled exclusively via processAssistantMessage
    // to avoid duplicate tool cards — assistant messages carry full input and tool_result.
  }
}

// Helper to bridge Record<string, unknown> to SDK Options type
function buildSdkOptions(opts: Record<string, unknown>): NonNullable<Parameters<typeof query>[0]["options"]> {
  return opts;
}

async function buildWorkspacePrompt(request: AgentRequest): Promise<string> {
  const workspaceContext = await readWorkspaceContext(request.workspacePath);
  if (workspaceContext === undefined) {
    return request.text;
  }

  return [
    "Selected workspace context:",
    workspaceContext,
    "",
    "Use the selected workspace context above as the primary source of truth.",
    "If the user asks about the current project, architecture, or system, answer about this selected workspace.",
    "Do not answer from the host agent implementation unless the user explicitly asks how this assistant is built.",
    "",
    "User request:",
    request.text,
  ].join("\n");
}

async function readWorkspaceContext(workspacePath: string): Promise<string | undefined> {
  try {
    const content = await readFile(path.join(workspacePath, "CLAUDE.md"), "utf8");
    const normalized = content.trim();
    return normalized.length > 0 ? normalized.slice(0, 4_000) : undefined;
  } catch {
    return undefined;
  }
}

function availableToolsForRole(config: RoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [...config.allowedTools];
  }

  return config.allowedTools.filter((tool) => !FILE_MUTATION_TOOLS.has(tool));
}

function autoAllowedToolsForRole(config: RoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [...config.allowedTools];
  }

  return config.allowedTools.filter((tool) => tool !== "Bash" && !FILE_MUTATION_TOOLS.has(tool));
}

function disallowedToolsForRole(config: RoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [];
  }

  return [...FILE_MUTATION_TOOLS].filter((tool) => config.allowedTools.includes(tool));
}

function canUseTool(config: RoleConfig, toolName: string): boolean {
  if (!config.allowedTools.includes(toolName)) {
    return false;
  }

  return !FILE_MUTATION_TOOLS.has(toolName) || roleCanUseFileMutationTools(config);
}

function roleCanUseFileMutationTools(config: RoleConfig): boolean {
  if (config.permissionMode !== "acceptEdits" && config.permissionMode !== "bypassPermissions") {
    return false;
  }

  return config.allowedTools.some((tool) => FILE_MUTATION_TOOLS.has(tool));
}

function denyTool(roleName: string, toolName: string): PermissionResult {
  return {
    behavior: "deny",
    message: `Tool ${toolName} is not permitted for role ${roleName}.`,
    decisionClassification: "user_reject",
  };
}
