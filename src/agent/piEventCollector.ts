/**
 * Translates pi-coding-agent AgentSessionEvents into the project's AgentStreamEvent format
 * and records structured logs to llm-raw.jsonl.
 */
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import type { AgentRequest, AgentResponse, ContextUsageReport, StoredRoleConfig } from "../core/contracts.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { canUseConfiguredTool } from "./toolPolicy.js";

const CONTEXT_WINDOW = 200_000;

const REVERSE_TOOL_NAME_MAP: Readonly<Record<string, string>> = {
  "read": "Read",
  "bash": "Bash",
  "edit": "Edit",
  "write": "Write",
  "find": "Glob",
  "grep": "Grep",
  "ls": "Glob",
};

export class PiEventCollector {
  private readonly request: AgentRequest;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly runtimeName: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly startTime: number;

  private finalText = "";
  private contextUsage: ContextUsageReport | undefined;
  private turnCount = 0;
  private sessionId = "";
  private preCompactTokens: number | undefined;

  public constructor(
    request: AgentRequest,
    rawLogger: JsonlLogger | undefined,
    runtimeName: string,
    roleConfig: StoredRoleConfig,
  ) {
    this.request = request;
    this.rawLogger = rawLogger;
    this.runtimeName = runtimeName;
    this.roleConfig = roleConfig;
    this.startTime = Date.now();
  }

  public onEvent(event: AgentSessionEvent): void {
    switch (event.type) {
      case "message_start": {
        this.turnCount++;
        break;
      }

      case "message_update": {
        const assistantEvent = event.assistantMessageEvent;
        if (assistantEvent.type === "text_delta") {
          this.finalText += assistantEvent.delta;
          this.request.stream?.({ type: "text_delta", text: assistantEvent.delta });
        } else if (assistantEvent.type === "thinking_delta") {
          this.request.stream?.({ type: "thinking", text: assistantEvent.delta });
        }
        break;
      }

      case "message_end": {
        // Extract usage from completed assistant message
        const msg = event.message;
        if (msg.role === "assistant" && "usage" in msg) {
          const usage = (msg as { usage: { input: number; output: number; cacheRead?: number; cacheWrite?: number } }).usage;
          this.contextUsage = this.extractContextUsage(usage);

          void this.rawLogger?.write({
            type: "llm.response",
            operation: "agent_runtime",
            runtime: this.runtimeName,
            userId: this.request.user.id,
            workspacePath: this.request.workspacePath,
            sessionId: this.sessionId,
            turn: this.turnCount,
            durationMs: Date.now() - this.startTime,
          });
        }
        break;
      }

      case "tool_execution_start": {
        this.request.stream?.({
          type: "tool_use_start",
          toolName: event.toolName,
          toolUseId: event.toolCallId,
          input: isRecord(event.args) ? event.args : {},
        });

        // Log tool call with permission info derived from role config
        const projectToolName = REVERSE_TOOL_NAME_MAP[event.toolName] ?? event.toolName;
        void this.rawLogger?.write({
          type: "agent.tool_call",
          runtime: this.runtimeName,
          userId: this.request.user.id,
          workspacePath: this.request.workspacePath,
          toolName: projectToolName,
          toolUseId: event.toolCallId,
          permittedByRole: canUseConfiguredTool(this.roleConfig, projectToolName),
          input: isRecord(event.args) ? event.args : {},
        });
        break;
      }

      case "tool_execution_end": {
        const resultText = extractToolResultText(event.result);
        this.request.stream?.({
          type: "tool_use_result",
          toolUseId: event.toolCallId,
          result: resultText,
          isError: event.isError,
        });

        void this.rawLogger?.write({
          type: "agent.tool_result",
          runtime: this.runtimeName,
          userId: this.request.user.id,
          workspacePath: this.request.workspacePath,
          toolUseId: event.toolCallId,
          isError: event.isError,
          result: resultText.slice(0, 500),
        });
        break;
      }

      case "compaction_start": {
        this.request.stream?.({ type: "compact_start" });
        break;
      }

      case "compaction_end": {
        if (event.result !== undefined) {
          this.preCompactTokens = "tokensBefore" in event.result ? (event.result as { tokensBefore: number }).tokensBefore : undefined;
        }

        this.request.stream?.({
          type: "compact_complete",
          preTokens: this.preCompactTokens ?? 0,
        });

        void this.rawLogger?.write({
          type: "llm.compact",
          runtime: this.runtimeName,
          userId: this.request.user.id,
          workspacePath: this.request.workspacePath,
          sessionId: this.sessionId,
          trigger: event.reason,
          preTokens: this.preCompactTokens,
        });

        // Notify compaction callback
        if (event.result !== undefined) {
          const summary = "summary" in event.result ? (event.result as { summary: string }).summary : undefined;
          if (summary !== undefined) {
            void this.request.onCompact?.(summary);
          }
        }
        break;
      }

      case "agent_end": {
        // Final event — log final response
        void this.rawLogger?.write({
          type: "llm.response",
          operation: "agent_runtime",
          runtime: this.runtimeName,
          userId: this.request.user.id,
          workspacePath: this.request.workspacePath,
          sessionId: this.sessionId,
          extractedText: this.finalText,
          turnCount: this.turnCount,
          durationMs: Date.now() - this.startTime,
        });
        break;
      }

      default:
        // Other events (auto_retry_start/end, session_info_changed, etc.) are not translated
        break;
    }
  }

  public setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }

  public toResponse(sessionId: string): AgentResponse {
    return {
      text: this.finalText.length > 0 ? this.finalText : "Agent completed without text output.",
      sessionId,
      ...(this.contextUsage !== undefined ? { contextUsage: this.contextUsage } : {}),
    };
  }

  private extractContextUsage(usage: { input: number; output: number; cacheRead?: number; cacheWrite?: number }): ContextUsageReport {
    const usagePercent = Math.round((usage.input / CONTEXT_WINDOW) * 100);

    return {
      inputTokens: usage.input,
      outputTokens: usage.output,
      ...(typeof usage.cacheRead === "number" ? { cacheReadTokens: usage.cacheRead } : {}),
      ...(typeof usage.cacheWrite === "number" ? { cacheCreationTokens: usage.cacheWrite } : {}),
      contextWindow: CONTEXT_WINDOW,
      usagePercent,
    };
  }
}

function extractToolResultText(result: unknown): string {
  if (typeof result === "string") return result;
  if (isRecord(result)) {
    const content = result["content"];
    if (Array.isArray(content)) {
      return content
        .filter((block: unknown): block is { type: "text"; text: string } => isRecord(block) && block["type"] === "text" && typeof block["text"] === "string")
        .map((block) => block.text)
        .join("");
    }
  }
  return String(result);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
