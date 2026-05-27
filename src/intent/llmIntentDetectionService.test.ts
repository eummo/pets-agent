import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
  fauxThinking,
} from "@earendil-works/pi-ai";
import { LlmIntentDetectionService } from "./llmIntentDetectionService.js";
import type { UserIntent } from "./index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { withRetry } from "../config/retry.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";

/**
 * Creates a service that simulates retryable failures in detectIntent.
 * The factory is called on each attempt — it can throw (retryable or not) or return a UserIntent.
 * This bypasses the LLM call to directly test retry logic with the same configuration
 * used in production (retries, shouldRetry, onRetry).
 */
function createRetryableService(
  factory: () => UserIntent,
  logger?: JsonlLogger,
): LlmIntentDetectionService {
  class RetryableIntentService extends LlmIntentDetectionService {
    public override async detectIntent(
      userMessage: string,
      role: string,
    ): Promise<UserIntent> {
      const startTime = Date.now();
      try {
        const result = await withRetry(() => Promise.resolve(factory()), {
          retries: 2,
          shouldRetry: (error: unknown) => {
            if (error instanceof DOMException && error.name === "AbortError") return true;
            if (error instanceof Error) {
              const message = error.message.toLowerCase();
              return message.includes("rate") || message.includes("overload")
                || message.includes("429") || message.includes("503");
            }
            return false;
          },
          onRetry: ({ attempt, error }: { attempt: number; delayMs: number; error: unknown }) => {
            void logger?.write({
              type: "intent.retry",
              role,
              userMessage,
              attempt,
              error: error instanceof Error ? error.message : String(error),
            });
          },
        });
        await logger?.write({
          type: "intent.result",
          role,
          userMessage,
          intentType: result.type,
          source: "model",
          durationMs: Date.now() - startTime,
        });
        return result;
      } catch {
        const intent = fallbackIntentFor(userMessage);
        await logger?.write({
          type: "intent.result",
          role,
          userMessage,
          intentType: intent.type,
          source: "fallback",
          reason: "exception",
          durationMs: Date.now() - startTime,
        });
        return intent;
      }
    }
  }

  const reg = registerFauxProvider({ tokensPerSecond: 50 });
  return new RetryableIntentService(reg.getModel(), "test-key", logger);
}

describe("LlmIntentDetectionService", () => {
  let registration: ReturnType<typeof registerFauxProvider>;

  beforeEach(() => {
    registration = registerFauxProvider({ tokensPerSecond: 50 });
  });

  afterEach(() => {
    registration.unregister();
  });

  function createService(responses: ReturnType<typeof fauxAssistantMessage>[]): LlmIntentDetectionService {
    registration.setResponses(responses);
    return new LlmIntentDetectionService(registration.getModel(), "test-key");
  }

  it("classifies query intent", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("query")]),
    ]);
    const result = await service.detectIntent("What is the architecture?", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("logs intent model request, response, and final result", async () => {
    registration.setResponses([
      fauxAssistantMessage([fauxText("query")]),
    ]);
    const rawEvents: Record<string, unknown>[] = [];
    const service = new LlmIntentDetectionService(registration.getModel(), "test-key", {
      filePath: "memory.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      },
    });

    const result = await service.detectIntent("客户订单是怎么创建的", "reviewer");

    expect(result).toEqual({ type: "query" });
    expect(rawEvents.map((event) => event["type"])).toEqual([
      "llm.request",
      "llm.response",
      "intent.result",
    ]);
    expect(rawEvents[0]).toMatchObject({
      type: "llm.request",
      operation: "intent_detection",
      role: "reviewer",
      userMessage: "客户订单是怎么创建的",
    });
    expect(rawEvents[1]).toMatchObject({
      type: "llm.response",
      operation: "intent_detection",
      role: "reviewer",
      userMessage: "客户订单是怎么创建的",
    });
    expect(asRecord(rawEvents[1]?.["response"])["stopReason"]).toBe("stop");
    expect(rawEvents[2]).toMatchObject({
      type: "intent.result",
      role: "reviewer",
      userMessage: "客户订单是怎么创建的",
      intentType: "query",
      source: "model",
    });
  });

  it("extracts text from thinking+text response", async () => {
    const service = createService([
      fauxAssistantMessage([
        fauxThinking("The user wants to change files."),
        fauxText("mutate"),
      ]),
    ]);
    const result = await service.detectIntent("Add checkout support", "reviewer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("classifies update_kb intent", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("update_kb")]),
    ]);
    const result = await service.detectIntent("Please update the documentation", "reviewer");

    expect(result).toEqual({ type: "update_kb" });
  });

  it("classifies mutation requests", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("mutate")]),
    ]);
    const result = await service.detectIntent("Fix the bug in auth.ts", "developer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("classifies Chinese system modification requests", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("mutate")]),
    ]);
    const result = await service.detectIntent("我想修改订单系统", "reviewer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("classifies Chinese creation explanation questions as query", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("query")]),
    ]);
    const result = await service.detectIntent("客户订单是怎么创建的", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("classifies Chinese feature-add requests", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("mutate")]),
    ]);
    const result = await service.detectIntent("添加新的订单功能 增加下单", "reviewer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("classifies ambiguous short message using conversation history", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("update_kb")]),
    ]);
    const history = [
      { role: "assistant" as const, content: "需要补充参数文档" },
    ];
    const result = await service.detectIntent("补充一下", "reviewer", history);

    expect(result).toEqual({ type: "update_kb" });
  });

  it("classifies confirmation as mutate when assistant suggested code change", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("mutate")]),
    ]);
    const history = [
      { role: "assistant" as const, content: "是否需要修改代码？" },
    ];
    const result = await service.detectIntent("好的", "developer", history);

    expect(result).toEqual({ type: "mutate" });
  });

  it("uses deterministic mutation fallback on unrecognized response", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("unknown_label")]),
    ]);
    const result = await service.detectIntent("Please implement order export", "reviewer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("uses deterministic query fallback for Chinese creation explanation questions", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("unknown_label")]),
    ]);
    const result = await service.detectIntent("客户订单是怎么创建的", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("defaults to query on empty text response", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("")]),
    ]);
    const result = await service.detectIntent("Hello", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("uses deterministic knowledge-base fallback on provider error", async () => {
    // No responses set — faux provider will throw
    const service = new LlmIntentDetectionService(registration.getModel(), "test-key");
    const result = await service.detectIntent("请更新知识库里的订单流程", "reviewer");

    expect(result).toEqual({ type: "update_kb" });
  });

  it("uses deterministic mutation fallback when provider returns an error message", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("mutate")], {
        stopReason: "error",
        errorMessage: "provider rejected request",
      }),
    ]);
    const result = await service.detectIntent("Fix the bug in auth.ts", "developer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("retries on AbortError (timeout) and succeeds on second attempt", async () => {
    let attempt = 0;
    const service = createRetryableService(() => {
      attempt++;
      if (attempt === 1) throw new DOMException("The operation was aborted", "AbortError");
      return { type: "mutate" };
    });
    const result = await service.detectIntent("修改代码", "reviewer");

    expect(result).toEqual({ type: "mutate" });
    expect(attempt).toBe(2);
  });

  it("retries on rate-limit error (429) and succeeds on second attempt", async () => {
    let attempt = 0;
    const service = createRetryableService(() => {
      attempt++;
      if (attempt === 1) throw new Error("429 Rate limit exceeded");
      return { type: "query" };
    });
    const result = await service.detectIntent("What is the architecture?", "reviewer");

    expect(result).toEqual({ type: "query" });
    expect(attempt).toBe(2);
  });

  it("retries on 503 overload error and succeeds on second attempt", async () => {
    let attempt = 0;
    const service = createRetryableService(() => {
      attempt++;
      if (attempt === 1) throw new Error("503 Overloaded");
      return { type: "update_kb" };
    });
    const result = await service.detectIntent("更新知识库", "reviewer");

    expect(result).toEqual({ type: "update_kb" });
    expect(attempt).toBe(2);
  });

  it("falls back to heuristic after exhausting retries", async () => {
    let attempt = 0;
    const service = createRetryableService(() => {
      attempt++;
      throw new DOMException("Aborted", "AbortError");
    });
    const result = await service.detectIntent("请修改订单系统", "reviewer");

    // 1 initial + 2 retries = 3 attempts, then fallback
    expect(attempt).toBe(3);
    expect(result).toEqual({ type: "mutate" });
  });

  it("does not retry on non-retryable errors", async () => {
    let attempt = 0;
    const service = createRetryableService(() => {
      attempt++;
      throw new Error("Authentication failed");
    });
    const result = await service.detectIntent("What is the architecture?", "reviewer");

    // Non-retryable error should not trigger retry — goes straight to fallback
    expect(attempt).toBe(1);
    expect(result).toEqual({ type: "query" });
  });

  it("logs retry events", async () => {
    let attempt = 0;
    const rawEvents: Record<string, unknown>[] = [];
    const service = createRetryableService(
      () => {
        attempt++;
        if (attempt === 1) throw new Error("429 Rate limit exceeded");
        return { type: "mutate" };
      },
      {
        filePath: "memory.jsonl",
        write(event: Record<string, unknown>) {
          rawEvents.push(event);
          return Promise.resolve();
        },
      },
    );

    const result = await service.detectIntent("Fix the bug", "developer");

    expect(result).toEqual({ type: "mutate" });
    const retryEvents = rawEvents.filter((e) => e["type"] === "intent.retry");
    expect(retryEvents).toHaveLength(1);
    expect(retryEvents[0]).toMatchObject({
      type: "intent.retry",
      role: "developer",
      userMessage: "Fix the bug",
      attempt: 1,
      error: "429 Rate limit exceeded",
    });
  });
});

function asRecord(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected object, got ${JSON.stringify(value)}.`);
  }
  return value as Record<string, unknown>;
}
