/**
 * QAAgent — Unit Tests
 * Run: npx vitest run src/tests/qa-agent.test.ts
 *
 * Uses vi.mock for MemoryRetriever and vi.spyOn for fetch.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { QAAgent, type ChatMessage } from "../qa-agent/qa-agent.js";

// Mock MemoryRetriever
const mockRetrieve = vi.fn();
const mockListAll = vi.fn();

vi.mock("../qa-agent/memory-retriever.js", () => ({
  MemoryRetriever: class {
    async init() {}
    retrieve = mockRetrieve;
    listAll = mockListAll;
  },
}));

describe("QAAgent", () => {
  let agent: QAAgent;

  beforeEach(() => {
    mockRetrieve.mockReset();
    mockListAll.mockReset();
    agent = new QAAgent({ apiKey: "test-key" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("constructor", () => {
    it("uses default model and base URL when not specified", () => {
      const a = new QAAgent({ apiKey: "k" });
      expect(a).toBeDefined();
    });

    it("accepts custom model and base URL", () => {
      const a = new QAAgent({
        apiKey: "k",
        model: "custom-model",
        baseURL: "https://custom.api/v1",
      });
      expect(a).toBeDefined();
    });
  });

  describe("clearHistory()", () => {
    it("clears conversation history", () => {
      const a = agent as unknown as { history: ChatMessage[] };
      a.history = [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ];
      agent.clearHistory();
      expect(a.history).toHaveLength(0);
    });
  });

  describe("ask()", () => {
    it("calls LLM and returns the response", async () => {
      mockRetrieve.mockReturnValue("context data");
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: "这是测试回答" } }],
        }),
      }) as unknown as () => Promise<Response>);

      await agent.init();
      const answer = await agent.ask("测试问题");
      expect(answer).toBe("这是测试回答");
    });

    it("passes retrieved context in system message to LLM", async () => {
      mockRetrieve.mockReturnValue("以下是从知识库中检索到的相关信息：\n\n### 命令/模式知识\n- npm run dev");

      let capturedBody: unknown = null;
      vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, opts) => {
        capturedBody = JSON.parse((opts as RequestInit).body as string);
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: "你可以使用 npm run dev" } }],
          }),
        } as unknown as Response;
      });

      await agent.init();
      await agent.ask("怎么启动开发服务器");

      const systemMsg = (capturedBody as { messages: ChatMessage[] }).messages[0];
      expect(systemMsg.role).toBe("system");
      expect(systemMsg.content).toContain("npm run dev");
    });

    it("includes no-knowledge notice when context is empty", async () => {
      mockRetrieve.mockReturnValue("");

      let capturedBody: unknown = null;
      vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, opts) => {
        capturedBody = JSON.parse((opts as RequestInit).body as string);
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: "知识库中暂无相关信息" } }],
          }),
        } as unknown as Response;
      });

      await agent.init();
      await agent.ask("一些无关的问题");

      const systemMsg = (capturedBody as { messages: ChatMessage[] }).messages[0];
      expect(systemMsg.content).toContain("知识库中暂无与该问题相关的信息");
    });

    it("maintains conversation history across multiple asks", async () => {
      mockRetrieve.mockReturnValue("");
      let callCount = 0;
      const responses = ["第一次回答", "第二次回答"];
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: responses[callCount++] } }],
        }),
      }) as unknown as () => Promise<Response>);

      await agent.init();
      await agent.ask("问题一");
      await agent.ask("问题二");

      const internal = agent as unknown as { history: ChatMessage[] };
      expect(internal.history).toHaveLength(4);
      expect(internal.history[0].content).toBe("问题一");
      expect(internal.history[1].content).toBe("第一次回答");
      expect(internal.history[2].content).toBe("问题二");
      expect(internal.history[3].content).toBe("第二次回答");
    });

    it("sends conversation history to LLM on subsequent asks", async () => {
      mockRetrieve.mockReturnValue("");
      let callCount = 0;
      const capturedBodies: unknown[] = [];
      vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, opts) => {
        capturedBodies.push(JSON.parse((opts as RequestInit).body as string));
        callCount++;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: `回答${callCount}` } }],
          }),
        } as unknown as Response;
      });

      await agent.init();
      await agent.ask("第一个问题");
      await agent.ask("第二个问题");

      const secondBody = capturedBodies[1] as { messages: ChatMessage[] };
      const nonSystem = secondBody.messages.filter((m) => m.role !== "system");
      expect(nonSystem.some((m) => m.content === "第一个问题")).toBe(true);
      expect(nonSystem.some((m) => m.content === "回答1")).toBe(true);
    });

    it("throws on non-transient API error", async () => {
      mockRetrieve.mockReturnValue("");
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => ({
        ok: false,
        status: 401,
        text: async () => "Unauthorized",
        json: async () => ({}),
      }) as unknown as Response);

      await agent.init();
      await expect(agent.ask("test")).rejects.toThrow("LLM API error 401");
    });

    it("retries on transient 5xx error", async () => {
      mockRetrieve.mockReturnValue("");
      let callCount = 0;
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        callCount++;
        if (callCount === 1) {
          return {
            ok: false,
            status: 500,
            text: async () => "Internal Server Error",
            json: async () => ({}),
          } as unknown as Response;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: "重试后成功" } }],
          }),
        } as unknown as Response;
      });

      await agent.init();
      const answer = await agent.ask("test");
      expect(answer).toBe("重试后成功");
      expect(callCount).toBe(2);
    });

    it("retries on 429 rate limit error", async () => {
      mockRetrieve.mockReturnValue("");
      let callCount = 0;
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        callCount++;
        if (callCount === 1) {
          return {
            ok: false,
            status: 429,
            text: async () => "Rate limited",
            json: async () => ({}),
          } as unknown as Response;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: "限流重试成功" } }],
          }),
        } as unknown as Response;
      });

      await agent.init();
      const answer = await agent.ask("test");
      expect(answer).toBe("限流重试成功");
    });

    it("returns timeout message on AbortError", async () => {
      mockRetrieve.mockReturnValue("");
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        const err = new DOMException("The operation was aborted", "AbortError");
        throw err;
      });

      await agent.init();
      const answer = await agent.ask("test");
      expect(answer).toBe("[请求超时]");
    });
  });

  describe("listKnowledge()", () => {
    it("returns empty message when no data exists", async () => {
      mockListAll.mockReturnValue("知识库当前为空，暂无任何记录。");
      await agent.init();
      const result = await agent.listKnowledge();
      expect(result).toContain("知识库当前为空");
    });

    it("returns formatted knowledge when data exists", async () => {
      mockListAll.mockReturnValue("以下是知识库的完整概览：\n\n### 命令/模式知识\n- git status");
      await agent.init();
      const result = await agent.listKnowledge();
      expect(result).toContain("命令/模式知识");
      expect(result).toContain("git status");
    });
  });
});
