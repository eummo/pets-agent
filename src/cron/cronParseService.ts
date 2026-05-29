import type { Api, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import { withRetry } from "../config/retry.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { isRecord, stringField, formatUnknownError } from "../core/unknownRecord.js";
import { z } from "zod";
import { cronScheduleSchema, deliveryTargetSchema } from "./cronTypes.js";

// ── Parse Result Schema ──────────────────────────────────────────────────────

export const cronParseResultSchema = z.object({
  name: z.string().min(1),
  schedule: cronScheduleSchema,
  prompt: z.string().min(1),
  workspacePath: z.string().min(1),
  role: z.string().min(1).optional(),
  delivery: deliveryTargetSchema,
  timeoutMs: z.number().int().positive().optional(),
  silentOnEmpty: z.boolean().optional(),
});

export type CronParseResult = z.infer<typeof cronParseResultSchema>;

// ── CronParseService Contract ────────────────────────────────────────────────

export type CronParseService = {
  parse(description: string): Promise<CronParseResult>;
};

// ── LLM-based Implementation ─────────────────────────────────────────────────

const CRON_PARSE_SYSTEM_PROMPT = `You are a cron job configuration parser.
Given a natural language description of a scheduled task, produce a JSON object with these fields:

- "name": a short descriptive name for the job (string, required)
- "schedule": an object with a "type" field and associated fields:
  - Cron expression: { "type": "cron", "expression": "0 9 * * 1-5" }
  - Fixed interval: { "type": "interval", "milliseconds": 3600000 }
  - One-shot: { "type": "once", "runAt": "2026-06-01T09:00:00.000Z" }
- "prompt": the task prompt to execute (string, required)
- "workspacePath": workspace path, default ".harness/knowledge-base" (string, required)
- "role": role to run as - "reviewer", "developer", or "admin" (optional string)
- "delivery": { "channels": ["wecom:chat:ID", "sse:admin", ...], "template": optional }
- "timeoutMs": timeout in milliseconds, default 120000 (optional number)
- "silentOnEmpty": whether to skip delivery when output is empty (optional boolean)

Rules:
- Use standard 5-field cron expressions (minute hour day month weekday).
- Convert relative times (e.g., "每天早上9点") to cron expressions.
- Convert "每小时" / "每30分钟" to interval milliseconds.
- Convert specific future dates/times to "once" with ISO 8601 runAt.
- Default workspacePath to ".harness/knowledge-base" unless the description specifies otherwise.
- Default delivery channels to ["sse:admin"] unless the description specifies otherwise.
- Respond with ONLY the JSON object, no markdown fences, no commentary.`;

const CRON_PARSE_TIMEOUT_MS = 10000;
const CRON_PARSE_MAX_RETRIES = 1;

export class LlmCronParseService implements CronParseService {
  public constructor(
    private readonly model: Model<Api>,
    private readonly apiKey: string,
    private readonly rawLogger?: JsonlLogger
  ) {}

  public async parse(description: string): Promise<CronParseResult> {
    const startTime = Date.now();

    await this.rawLogger?.write({
      type: "llm.request",
      operation: "cron_parse",
      description,
    });

    try {
      const response = await withRetry(
        async () => {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), CRON_PARSE_TIMEOUT_MS);

          return complete(
            this.model,
            {
              systemPrompt: CRON_PARSE_SYSTEM_PROMPT,
              messages: [
                {
                  role: "user",
                  content: description,
                  timestamp: Date.now(),
                },
              ],
            },
            {
              apiKey: this.apiKey,
              signal: controller.signal,
            }
          )
            .then((resp) => {
              if (resp.stopReason === "error" && isRetryableProviderResponse(resp)) {
                throw new Error(errorMessageForResponse(resp));
              }
              return resp;
            })
            .finally(() => clearTimeout(timeout));
        },
        {
          retries: CRON_PARSE_MAX_RETRIES,
          shouldRetry: (error) => isAbortError(error) || isRetryableError(error),
          onRetry: ({ attempt, delayMs, error }) => {
            void this.rawLogger?.write({
              type: "cron_parse.retry",
              description,
              attempt,
              delayMs,
              error: formatUnknownError(error),
            });
          },
        }
      );

      if (response.stopReason === "error") {
        const msg = errorMessageForResponse(response);
        throw new Error(`LLM returned error: ${msg}`);
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");

      const parsed = parseJsonResponse(text);
      const result = cronParseResultSchema.parse(parsed);

      await this.rawLogger?.write({
        type: "llm.response",
        operation: "cron_parse",
        description,
        result,
        durationMs: Date.now() - startTime,
      });

      return result;
    } catch (error) {
      await this.rawLogger?.write({
        type: "llm.error",
        operation: "cron_parse",
        description,
        error: formatUnknownError(error),
        durationMs: Date.now() - startTime,
      });
      throw error;
    }
  }
}

function parseJsonResponse(text: string): unknown {
  const trimmed = text.trim();
  // Strip markdown code fences if present
  const fenceMatch = /^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```$/.exec(trimmed);
  const jsonText = fenceMatch?.[1] ?? trimmed;
  return JSON.parse(jsonText);
}

function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException) return error.name === "AbortError";
  if (error instanceof Error) return error.name === "AbortError";
  return false;
}

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = error.message.toLowerCase();
  return (
    message.includes("abort") ||
    message.includes("rate") ||
    message.includes("overload") ||
    message.includes("429") ||
    message.includes("503")
  );
}

function isRetryableProviderResponse(
  response: Awaited<ReturnType<typeof complete>>
): boolean {
  return isRetryableError(new Error(errorMessageForResponse(response)));
}

function errorMessageForResponse(
  response: Awaited<ReturnType<typeof complete>>
): string {
  const errorMessage = isRecord(response) ? stringField(response, "errorMessage") : undefined;
  if (errorMessage !== undefined) {
    return errorMessage;
  }
  return response.stopReason;
}
