/**
 * PatternMemory — Unit Tests
 * Run: npx vitest run src/tests/pattern-memory.test.ts --env node
 */

import { describe, it, expect, beforeEach } from "vitest";
import { patternMemory } from "../memory/pattern-memory.js";

describe("PatternMemory", () => {
  beforeEach(() => {
    // Clear all entries before each test
    const entries = patternMemory.all();
    for (const e of entries) {
      patternMemory.remove(e.id);
    }
  });

  describe("add()", () => {
    it("adds a new pattern entry", () => {
      const result = patternMemory.add("npm run build", { tags: ["npm", "build"] });
      expect(result.success).toBe(true);
    });

    it("rejects empty content", () => {
      const result = patternMemory.add("   ");
      expect(result.success).toBe(false);
      expect(result.error).toContain("empty");
    });

    it("prevents duplicate content", () => {
      patternMemory.add("cargo check", { tags: ["rust"] });
      const dup = patternMemory.add("cargo check", { tags: ["rust"] });
      expect(dup.success).toBe(false);
      expect(dup.error).toContain("exists");
    });
  });

  describe("query() / search()", () => {
    beforeEach(() => {
      patternMemory.add("npm run build", { tags: ["npm", "build"] });
      patternMemory.add("yarn dev", { tags: ["yarn", "dev"] });
      patternMemory.add("pnpm install", { tags: ["pnpm"] });
    });

    it("finds entries by content substring", () => {
      const results = patternMemory.query("npm");
      expect(results.some((e) => e.content.includes("npm"))).toBe(true);
    });

    it("finds entries by tag match", () => {
      const results = patternMemory.query("yarn");
      expect(results.some((e) => e.tags.includes("yarn"))).toBe(true);
    });

    it("returns all entries when query is empty", () => {
      const results = patternMemory.query("");
      expect(results.length).toBeGreaterThanOrEqual(3);
    });

    it("returns empty for non-existent query", () => {
      const results = patternMemory.query("zzzzzzzxyz");
      expect(results).toHaveLength(0);
    });
  });

  describe("remove()", () => {
    it("removes entry by id", () => {
      patternMemory.add("rm -rf /tmp/test", { tags: ["shell"] });
      const entry = patternMemory.query("rm -rf")[0];
      const removed = patternMemory.remove(entry.id);
      expect(removed.success).toBe(true);
      expect(patternMemory.query("rm -rf")).toHaveLength(0);
    });

    it("removes entry by content substring", () => {
      patternMemory.add("cargo build --release", { tags: ["rust"] });
      const removed = patternMemory.remove("cargo build");
      expect(removed.success).toBe(true);
      expect(patternMemory.query("cargo build")).toHaveLength(0);
    });

    it("returns error for non-existent id", () => {
      const result = patternMemory.remove("non-existent-abc123");
      expect(result.success).toBe(false);
    });
  });

  describe("learnFromOutput()", () => {
    it("extracts shell commands from $ prefix lines", () => {
      const output = ["$ npm install express", "$ node index.js"];
      patternMemory.learnFromOutput(output);
      const results = patternMemory.query("npm install");
      expect(results.length).toBeGreaterThan(0);
    });

    it("extracts error fix patterns", () => {
      // Regex requires error/failed/exception AND fix/resolved/solved on the SAME line
      const output = ["Error: module not found — resolved with: npm install express"];
      patternMemory.learnFromOutput(output);
      const results = patternMemory.query("npm install express");
      expect(results.some((e) => e.tags.includes("error-fix"))).toBe(true);
    });
  });

  describe("getSnapshot()", () => {
    it("returns a non-empty snapshot string with header", () => {
      patternMemory.add("echo test", { tags: ["shell"] });
      const snap = patternMemory.getSnapshot();
      expect(typeof snap).toBe("string");
      expect(snap.length).toBeGreaterThan(0);
      expect(snap).toContain("PATTERNS");
    });
  });
});
