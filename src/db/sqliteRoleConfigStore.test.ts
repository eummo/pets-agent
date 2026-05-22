import { describe, it, expect, beforeEach } from "vitest";
import { createSqliteConnection } from "./sqliteConnection.js";
import { SqliteRoleConfigStore } from "./sqliteRoleConfigStore.js";
import type { StoredRoleConfig } from "../core/ports.js";

describe("SqliteRoleConfigStore", () => {
  let store: SqliteRoleConfigStore;

  beforeEach(() => {
    const db = createSqliteConnection(":memory:");
    store = new SqliteRoleConfigStore(db);
  });

  it("starts empty", async () => {
    const all = await store.getAll();
    expect(all).toEqual([]);
  });

  it("returns undefined for unknown role", async () => {
    const found = await store.getByName("unknown");
    expect(found).toBeUndefined();
  });

  it("upserts and retrieves a role config", async () => {
    const config: StoredRoleConfig = {
      name: "reviewer",
      systemPrompt: "You are a reviewer.",
      allowedTools: ["Read", "Glob", "Grep"],
      permissionMode: "dontAsk",
      maxTurns: 10,
    };

    await store.upsert(config);

    const retrieved = await store.getByName("reviewer");
    expect(retrieved).toEqual(config);
  });

  it("upsert updates an existing role", async () => {
    await store.upsert({
      name: "reviewer",
      systemPrompt: "Old prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
    });

    await store.upsert({
      name: "reviewer",
      systemPrompt: "New prompt",
      allowedTools: ["Read", "Glob"],
      permissionMode: "acceptEdits",
    });

    const retrieved = await store.getByName("reviewer");
    expect(retrieved?.systemPrompt).toBe("New prompt");
    expect(retrieved?.allowedTools).toEqual(["Read", "Glob"]);
    expect(retrieved?.permissionMode).toBe("acceptEdits");
  });

  it("lists all roles ordered by name", async () => {
    await store.upsert({
      name: "developer",
      systemPrompt: "Dev prompt",
      allowedTools: ["Read", "Edit"],
      permissionMode: "bypassPermissions",
    });
    await store.upsert({
      name: "reviewer",
      systemPrompt: "Rev prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
    });

    const all = await store.getAll();
    expect(all.map((r) => r.name)).toEqual(["developer", "reviewer"]);
  });

  it("deletes a role by name", async () => {
    await store.upsert({
      name: "reviewer",
      systemPrompt: "Prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
    });

    const deleted = await store.deleteByName("reviewer");
    expect(deleted).toBe(true);

    const found = await store.getByName("reviewer");
    expect(found).toBeUndefined();
  });

  it("returns false when deleting nonexistent role", async () => {
    const deleted = await store.deleteByName("nonexistent");
    expect(deleted).toBe(false);
  });

  it("handles optional maxTurns and model", async () => {
    await store.upsert({
      name: "custom",
      systemPrompt: "Custom prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
      maxTurns: 15,
      model: "claude-3-haiku",
    });

    const retrieved = await store.getByName("custom");
    expect(retrieved?.maxTurns).toBe(15);
    expect(retrieved?.model).toBe("claude-3-haiku");
  });

  it("handles null maxTurns and model", async () => {
    await store.upsert({
      name: "minimal",
      systemPrompt: "Minimal prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
    });

    const retrieved = await store.getByName("minimal");
    expect(retrieved?.maxTurns).toBeUndefined();
    expect(retrieved?.model).toBeUndefined();
  });
});
