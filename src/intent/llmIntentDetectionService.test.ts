import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
  fauxThinking,
} from "@earendil-works/pi-ai";
import { LlmIntentDetectionService } from "./llmIntentDetectionService.js";

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
});

function asRecord(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected object, got ${JSON.stringify(value)}.`);
  }
  return value as Record<string, unknown>;
}
