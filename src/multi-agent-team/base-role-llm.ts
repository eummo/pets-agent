/**
 * BaseRoleLLM — LLM-powered base class for team roles.
 *
 * Provides:
 * - MiniMax API integration
 * - Structured prompt building per role/phase
 * - AbortSignal support for cancellation
 * - Streaming progress via onUpdate callback
 */

import { Role, type RoleContext } from "./role.js";
import type { TeamRole, ProjectPhase } from "./types.js";
import { PHASE_LABELS, ROLE_DESCRIPTIONS } from "./types.js";

export interface LLMConfig {
  apiKey: string;
  model?: string;
  baseURL?: string;
}

const DEFAULT_BASE_URL = "https://api.minimax.chat/v1";
const DEFAULT_MODEL = "MiniMax-Text-01";

export abstract class BaseRoleLLM extends Role {
  protected abstract role(): TeamRole;
  protected abstract buildUserPrompt(ctx: RoleContext): string;

  /**
   * Override to customize the system prompt. Default uses role description + phase context.
   */
  protected buildSystemPrompt(_ctx: RoleContext): string {
    const role = this.role();
    return `你是${ROLE_DESCRIPTIONS[role]}。

能力范围：
- 理解项目需求和上下文
- 生成结构化文档（Markdown）
- 提供可操作的建议

输出要求：
- 语言：中文（文档）+ 英文（代码/技术术语）
- 格式：Markdown 结构化输出
- 详细程度：中等，适合团队评审`;
  }

  /**
   * Call the LLM with the given user prompt.
   * Uses MiniMax chat completion API.
   */
  protected async callLLM(userPrompt: string, ctx: RoleContext): Promise<string> {
    const apiKey = this.getAPIKey();
    const model = this.getModel();
    const baseURL = this.getBaseURL();
    const systemPrompt = this.buildSystemPrompt(ctx);

    // Honour external signal + optional per-call timeout
    const controller = new AbortController();
    const timeoutMs = ctx.timeoutMs;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (timeoutMs && timeoutMs > 0) {
      timer = setTimeout(() => controller.abort(), timeoutMs);
    }
    const signal = ctx.signal
      ? (() => {
          // Chain: if either external signal or controller fires, abort
          const ext = ctx.signal;
          const merged = new AbortController();
          ext.addEventListener("abort", () => merged.abort());
          controller.signal.addEventListener("abort", () => merged.abort());
          return merged.signal;
        })()
      : controller.signal;

    try {
      const res = await fetch(`${baseURL}/text/chatcompletion_v2`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          temperature: 0.4,
          max_tokens: 4096,
        }),
        signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`LLM API error ${res.status}: ${errText}`);
      }

      const json = await res.json() as { choices?: Array<{ message?: { content?: string } }> };
      const content = json.choices?.[0]?.message?.content ?? "";
      return content;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return "[cancelled]";
      }
      throw err;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  protected getAPIKey(): string {
    const key = process.env.MINIMAX_API_KEY ?? process.env.MINIMAX_KEY;
    if (!key) {
      throw new Error("MINIMAX_API_KEY / MINIMAX_KEY environment variable not set");
    }
    return key;
  }

  protected getModel(): string {
    return process.env.LLM_MODEL ?? DEFAULT_MODEL;
  }

  protected getBaseURL(): string {
    return process.env.LLM_BASE_URL ?? DEFAULT_BASE_URL;
  }
}
