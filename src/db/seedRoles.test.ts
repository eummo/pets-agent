import { describe, expect, it } from "vitest";
import { REVIEWER_CONFIG } from "../agent/claudeSdkAgentRuntime.js";
import { createSqliteConnection } from "./sqliteConnection.js";
import { SqliteRoleConfigStore } from "./sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./seedRoles.js";

describe("seedDefaultRoles", () => {
  it("seeds missing default roles", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));

    await seedDefaultRoles(store);

    await expect(store.getByName("reviewer")).resolves.toEqual(expect.objectContaining({
      name: "reviewer",
      maxTurns: REVIEWER_CONFIG.maxTurns,
      capabilities: ["workspace_read"],
    }));
    await expect(store.getByName("developer")).resolves.toEqual(expect.objectContaining({
      name: "developer",
      capabilities: ["workspace_read", "workspace_mutate"],
    }));
    await expect(store.getByName("admin")).resolves.toEqual(expect.objectContaining({
      name: "admin",
      capabilities: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"],
    }));
  });

  it("raises existing reviewer runtime defaults without replacing the prompt", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));
    await store.upsert({
      name: "reviewer",
      systemPrompt: "Custom reviewer prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
      maxTurns: 10,
    });

    await seedDefaultRoles(store);

    await expect(store.getByName("reviewer")).resolves.toEqual({
      name: "reviewer",
      systemPrompt: "Custom reviewer prompt",
      allowedTools: ["Read", "Glob", "Grep", "Bash"],
      permissionMode: "dontAsk",
      maxTurns: REVIEWER_CONFIG.maxTurns,
    });
  });
});
