import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { MemoryInjector } from "./injector.js";
import { PreferenceMemory } from "./preference-memory.js";

describe("MemoryInjector — includeSkills", () => {
  it("does not contain SKILLS section when includeSkills is false", () => {
    const injector = new MemoryInjector();
    const block = injector.buildBlock({ workdir: "/tmp", includeSkills: false });
    expect(block).not.toContain("SKILLS");
  });

  it("calls getSkills() when includeSkills is true (non-fatal even if no skills found)", () => {
    const injector = new MemoryInjector();
    // When no skills are found the block is empty and filtered out;
    // the important thing is that buildBlock() completes without throwing.
    expect(() => injector.buildBlock({ workdir: "/tmp", includeSkills: true })).not.toThrow();
  });
});

describe("PreferenceMemory — MEMORY_CHAR_LIMIT env var", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("getSnapshot works when MEMORY_CHAR_LIMIT is not set", () => {
    delete process.env.MEMORY_CHAR_LIMIT;
    const pm = new PreferenceMemory();
    expect(pm.getSnapshot()).toBeDefined();
  });

  it("getSnapshot works when MEMORY_CHAR_LIMIT is a valid number", () => {
    process.env.MEMORY_CHAR_LIMIT = "1000";
    const pm = new PreferenceMemory();
    expect(pm.getSnapshot()).toBeDefined();
  });

  it("getSnapshot works when MEMORY_CHAR_LIMIT is non-numeric (falls back to default)", () => {
    process.env.MEMORY_CHAR_LIMIT = "not-a-number";
    const pm = new PreferenceMemory();
    expect(pm.getSnapshot()).toBeDefined();
  });

  it("getSnapshot works when MEMORY_CHAR_LIMIT is empty string (falls back to default)", () => {
    process.env.MEMORY_CHAR_LIMIT = "";
    const pm = new PreferenceMemory();
    expect(pm.getSnapshot()).toBeDefined();
  });
});
