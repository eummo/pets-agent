/**
 * MemoryRetriever — Unit Tests
 * Run: npx vitest run src/tests/memory-retriever.test.ts
 *
 * Uses vi.mock to isolate from disk state — no real file I/O during tests.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryRetriever } from "../qa-agent/memory-retriever.js";
import type { MemoryEntry } from "../memory/store.js";

// Mock the memory modules
vi.mock("../memory/pattern-memory.js", () => {
  let entries: Array<MemoryEntry & { tags: string[] }> = [];

  return {
    PatternMemory: class {
      async loadAsync() {}
      search(query: string) {
        if (!query.trim()) return entries;
        const q = query.toLowerCase();
        return entries.filter(
          (e) =>
            e.content.toLowerCase().includes(q) ||
            e.tags.some((t) => t.toLowerCase().includes(q))
        );
      }
      all() {
        return entries;
      }
      // Test helper
      static _setEntries(e: typeof entries) {
        entries = e;
      }
      static _clear() {
        entries = [];
      }
    },
  };
});

vi.mock("../memory/preference-memory.js", () => {
  let entries: Array<MemoryEntry & { tags: string[] }> = [];

  return {
    PreferenceMemory: class {
      async loadAsync() {}
      query(query: string) {
        if (!query.trim()) return entries;
        const q = query.toLowerCase();
        return entries.filter(
          (e) =>
            e.content.toLowerCase().includes(q) ||
            e.tags.some((t) => t.toLowerCase().includes(q))
        );
      }
      all() {
        return entries;
      }
      // Test helper
      static _setEntries(e: typeof entries) {
        entries = e;
      }
      static _clear() {
        entries = [];
      }
    },
  };
});

// Mock fs.existsSync / readdirSync / readFileSync for project loading
vi.mock("fs", async () => {
  const actual = await vi.importActual<typeof import("fs")>("fs");
  return {
    ...actual,
    existsSync: vi.fn().mockReturnValue(false),
    readdirSync: vi.fn().mockReturnValue([]),
    readFileSync: vi.fn().mockReturnValue(""),
  };
});

// Import after mocks are set up
import { PatternMemory } from "../memory/pattern-memory.js";
import { PreferenceMemory } from "../memory/preference-memory.js";

describe("MemoryRetriever", () => {
  let retriever: MemoryRetriever;

  beforeEach(() => {
    // @ts-expect-error — accessing static mock helper
    PatternMemory._clear();
    // @ts-expect-error — accessing static mock helper
    PreferenceMemory._clear();
    retriever = new MemoryRetriever();
  });

  describe("retrieve()", () => {
    it("returns empty string when no memory entries exist", async () => {
      await retriever.init();
      const result = retriever.retrieve("anything");
      expect(result).toBe("");
    });

    it("returns pattern entries matching the query", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "npm run build", tags: ["npm", "build"], createdAt: new Date().toISOString(), source: "auto" },
        { id: "2", content: "cargo test", tags: ["rust", "test"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.retrieve("npm");
      expect(result).toContain("命令/模式知识");
      expect(result).toContain("npm run build");
      expect(result).not.toContain("cargo test");
    });

    it("returns preference entries matching the query", async () => {
      // @ts-expect-error — accessing static mock helper
      PreferenceMemory._setEntries([
        { id: "1", content: "✓ claude-code for: fix the build error\n  duration: 30s", tags: ["agent:claude-code", "outcome:success"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.retrieve("claude-code");
      expect(result).toContain("Agent 偏好知识");
      expect(result).toContain("claude-code");
    });

    it("includes header line when results are found", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "git push", tags: ["git"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.retrieve("git");
      expect(result).toContain("以下是从知识库中检索到的相关信息");
    });

    it("returns empty when query matches nothing", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "npm run build", tags: ["npm"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.retrieve("nonexistent-query");
      expect(result).toBe("");
    });

    it("combines results from multiple stores", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "npm install", tags: ["npm"], createdAt: new Date().toISOString(), source: "auto" },
      ]);
      // @ts-expect-error — accessing static mock helper
      PreferenceMemory._setEntries([
        { id: "1", content: "✓ claude-code for: npm install deps", tags: ["agent:claude-code", "outcome:success"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.retrieve("npm");
      expect(result).toContain("命令/模式知识");
      expect(result).toContain("Agent 偏好知识");
    });
  });

  describe("listAll()", () => {
    it("returns empty message when no memory exists", async () => {
      await retriever.init();
      const result = retriever.listAll();
      expect(result).toContain("知识库当前为空");
    });

    it("lists all patterns with count", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "npm test", tags: ["npm", "test"], createdAt: new Date().toISOString(), source: "auto" },
        { id: "2", content: "cargo build", tags: ["rust"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.listAll();
      expect(result).toContain("命令/模式知识");
      expect(result).toContain("2 条");
      expect(result).toContain("npm test");
      expect(result).toContain("cargo build");
    });

    it("lists all preferences with count", async () => {
      // @ts-expect-error — accessing static mock helper
      PreferenceMemory._setEntries([
        { id: "1", content: "✓ pi-agent for: refactor code", tags: ["agent:pi-agent", "outcome:success"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.listAll();
      expect(result).toContain("Agent 偏好知识");
      expect(result).toContain("1 条");
    });

    it("includes header line when entries exist", async () => {
      // @ts-expect-error — accessing static mock helper
      PatternMemory._setEntries([
        { id: "1", content: "echo hello", tags: ["shell"], createdAt: new Date().toISOString(), source: "auto" },
      ]);

      await retriever.init();
      const result = retriever.listAll();
      expect(result).toContain("以下是知识库的完整概览");
    });
  });
});
