import { describe, it, expect, vi, beforeEach } from "vitest";
import { LlmIntentDetectionService } from "./llmIntentDetectionService.js";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";

const mockConfig: ResolvedLlmConfig = {
  baseUrl: "https://api.example.com",
  apiKeyEnv: "TEST_KEY",
  modelId: "test-model",
  apiKey: "test-api-key",
};

describe("LlmIntentDetectionService", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("classifies query intent", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ text: "query" }] }), { status: 200 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("What is the architecture?", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("classifies update_kb intent", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ text: "update_kb" }] }), { status: 200 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Please update the documentation", "reviewer");

    expect(result).toEqual({ type: "update_kb" });
  });

  it("classifies obvious knowledge-base update requests without calling the API", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("请帮我更新知识库里的订单流程", "reviewer");

    expect(result).toEqual({ type: "update_kb" });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("classifies obvious mutation requests without calling the API", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Please fix the bug in auth.ts", "reviewer");

    expect(result).toEqual({ type: "mutate" });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("classifies mutate intent", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ text: "mutate" }] }), { status: 200 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Fix the bug in auth.ts", "developer");

    expect(result).toEqual({ type: "mutate" });
  });

  it("defaults to query on unrecognized response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ text: "unknown_label" }] }), { status: 200 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Hello", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("defaults to query on API error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response("Internal Server Error", { status: 500 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Something", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("defaults to query on network error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("Network error"));

    const service = new LlmIntentDetectionService(mockConfig);
    const result = await service.detectIntent("Something", "reviewer");

    expect(result).toEqual({ type: "query" });
  });

  it("sends correct API request format", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ text: "query" }] }), { status: 200 })
    );

    const service = new LlmIntentDetectionService(mockConfig);
    await service.detectIntent("What is this?", "reviewer");

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const callArgs = fetchSpy.mock.calls[0]!;
    const url = callArgs[0] as string;
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const options = callArgs[1]!;
    expect(url).toBe("https://api.example.com/v1/messages");
    expect(options.method).toBe("POST");
    expect(options.headers).toHaveProperty("x-api-key", "test-api-key");

    const body = JSON.parse(options.body as string) as Record<string, unknown>;
    expect(body["model"]).toBe("test-model");
    expect(body["max_tokens"]).toBe(10);
  });
});
