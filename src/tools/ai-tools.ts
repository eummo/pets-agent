/**
 * AI Tools — `/ai` slash command powered by Claude Agent SDK.
 *
 * Spawns a local autonomous agent via @anthropic-ai/claude-agent-sdk
 * and streams tool execution to the TUI via the ExtensionCommandContext API.
 */

import { query } from "@anthropic-ai/claude-agent-sdk";
import type {
  SDKAssistantMessage,
  SDKResultSuccess,
  SDKResultError,
  SDKMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { BetaContentBlock, BetaToolUseBlock, BetaTextBlock, BetaThinkingBlock } from "@anthropic-ai/sdk/resources/beta/messages/messages.mjs";
import type { ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

// ─── Type Guards ─────────────────────────────────────────────────────────────

function isAssistantMessage(msg: SDKMessage): msg is SDKAssistantMessage {
  return msg.type === "assistant";
}

function isSuccessResult(msg: SDKMessage): msg is SDKResultSuccess {
  return msg.type === "result" && (msg as unknown as { subtype: string }).subtype === "success";
}

function isErrorResult(msg: SDKMessage): msg is SDKResultError {
  return msg.type === "result" && (msg as unknown as { subtype: string }).subtype !== "success";
}

// ─── Tool Registration ────────────────────────────────────────────────────────

export function registerAiTools(): (pi: unknown) => void {
  return function register(pi: unknown) {
    const ext = pi as {
      registerCommand(
        name: string,
        options: {
          description: string;
          sourceInfo?: { source: string };
          handler: (args: string, ctx: ExtensionCommandContext) => Promise<void>;
        },
      ): void;
    };

    ext.registerCommand("ai", {
      description: "Run an autonomous AI agent to complete a task",
      sourceInfo: { source: "pets-agent" },
      async handler(args: string, ctx: ExtensionCommandContext) {
        console.error("[ai-tools] /ai handler called, args:", JSON.stringify(args));

        const prompt = args.trim();
        if (!prompt) {
          ctx.ui.notify("Usage: /ai <task description>", "warning");
          return;
        }

        // Set working state
        ctx.ui.setWorkingIndicator({ frames: ["◐", "○"], intervalMs: 400 });
        ctx.ui.setWorkingMessage("AI agent working…");

        const abortController = new AbortController();

        // Wire Ctrl-C to abort
        ctx.ui.onTerminalInput((key) => {
          if (key === "\x03") {
            abortController.abort();
            return { consume: true };
          }
        });

        try {
          const stream = query({
            prompt,
            options: {
              cwd: ctx.cwd,
              allowedTools: [
                "Read",
                "Edit",
                "Write",
                "Bash",
                "Glob",
                "Grep",
                "WebSearch",
                "WebFetch",
              ],
              permissionMode: "acceptEdits",
            },
          });

          let done = false;

          for await (const msg of stream) {
            if (abortController.signal.aborted) break;

            if (isAssistantMessage(msg)) {
              const content: BetaContentBlock[] = (msg as SDKAssistantMessage).message.content as BetaContentBlock[] ?? [];
              for (const block of content) {
                if (block.type === "tool_use") {
                  const tool = block as BetaToolUseBlock;
                  ctx.ui.setWorkingMessage(
                    `🔧 ${tool.name}: ${JSON.stringify(tool.input).slice(0, 120)}`,
                  );
                } else if (block.type === "thinking") {
                  const thinking = block as BetaThinkingBlock;
                  ctx.ui.setWorkingMessage(
                    `💭 ${thinking.thinking?.slice(0, 80) ?? ""}`,
                  );
                } else if (block.type === "text") {
                  const text = block as BetaTextBlock;
                  ctx.ui.setWorkingMessage(text.text?.slice(0, 80) ?? "");
                }
              }
            } else if (isSuccessResult(msg)) {
              done = true;
              ctx.ui.setWorkingMessage(undefined);
              ctx.ui.setWorkingIndicator(undefined);
              ctx.ui.notify(`✅ ${(msg as SDKResultSuccess).result ?? "Task completed"}`, "info");
            } else if (isErrorResult(msg)) {
              done = true;
              ctx.ui.setWorkingMessage(undefined);
              ctx.ui.setWorkingIndicator(undefined);
              const err = msg as SDKResultError;
              const errorMsg = err.errors?.[0] ?? "Task failed";
              ctx.ui.notify(`❌ ${errorMsg}`, "error");
            }
          }

          if (!done) {
            ctx.ui.setWorkingMessage(undefined);
            ctx.ui.setWorkingIndicator(undefined);
          }
        } catch (err: unknown) {
          ctx.ui.setWorkingMessage(undefined);
          ctx.ui.setWorkingIndicator(undefined);
          const message = err instanceof Error ? err.message : String(err);
          ctx.ui.notify(`AI agent error: ${message}`, "error");
        }
      },
    });
  };
}
