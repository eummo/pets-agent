import { describe, expect, it } from "vitest";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";
import { createSqliteConnection } from "./sqliteConnection.js";
import { SqliteRoleConfigStore } from "./sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./seedRoles.js";

const REVIEWER_DEFAULT = DEFAULT_ROLE_CONFIGS.find((c) => c.name === "reviewer");
if (REVIEWER_DEFAULT === undefined) throw new Error("reviewer default config missing");

describe("seedDefaultRoles", () => {
  it("seeds missing default roles", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));

    await seedDefaultRoles(store);

    await expect(store.getByName("reviewer")).resolves.toEqual(
      expect.objectContaining({
        name: "reviewer",
        maxTurns: REVIEWER_DEFAULT.maxTurns,
        capabilities: ["workspace_read", "web_access"],
        skills: "all",
        settingSources: ["user", "project", "local"]
      })
    );
    await expect(store.getByName("developer")).resolves.toEqual(
      expect.objectContaining({
        name: "developer",
        capabilities: ["workspace_read", "workspace_mutate", "knowledge_base_update", "web_access"],
        skills: "all",
        settingSources: ["user", "project", "local"],
        enableWorkflows: true
      })
    );
    await expect(store.getByName("admin")).resolves.toEqual(
      expect.objectContaining({
        name: "admin",
        capabilities: [
          "workspace_read",
          "workspace_mutate",
          "knowledge_base_update",
          "feedback_view",
          "feedback_manage",
          "cron_manage",
          "web_access"
        ],
        skills: "all",
        settingSources: ["user", "project", "local"]
      })
    );
  });

  it("raises existing reviewer runtime defaults without replacing the prompt", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));
    await store.upsert({
      name: "reviewer",
      systemPrompt: "Custom reviewer prompt",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
      maxTurns: 10
    });

    await seedDefaultRoles(store);

    await expect(store.getByName("reviewer")).resolves.toEqual(
      expect.objectContaining({
        name: "reviewer",
        systemPrompt: "Custom reviewer prompt",
        allowedTools: ["Read", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"],
        permissionMode: "dontAsk",
        maxTurns: REVIEWER_DEFAULT.maxTurns
      })
    );
  });

  it("adds missing default capabilities to existing roles without replacing prompts", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));
    await store.upsert({
      name: "developer",
      systemPrompt: "Custom developer prompt",
      allowedTools: ["Read", "Edit", "Write"],
      permissionMode: "bypassPermissions",
      capabilities: ["workspace_read", "workspace_mutate"]
    });

    await seedDefaultRoles(store);

    await expect(store.getByName("developer")).resolves.toEqual(
      expect.objectContaining({
        name: "developer",
        systemPrompt: "Custom developer prompt",
        capabilities: ["workspace_read", "workspace_mutate", "knowledge_base_update", "web_access"]
      })
    );
  });

  it("adds missing workflow defaults to existing developer roles without replacing prompts", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));
    await store.upsert({
      name: "developer",
      systemPrompt: "Custom developer prompt",
      allowedTools: ["Read", "Edit", "Write", "Bash"],
      permissionMode: "bypassPermissions",
      capabilities: ["workspace_read", "workspace_mutate", "knowledge_base_update"]
    });

    await seedDefaultRoles(store);

    await expect(store.getByName("developer")).resolves.toEqual(
      expect.objectContaining({
        name: "developer",
        systemPrompt: "Custom developer prompt",
        enableWorkflows: true
      })
    );
  });

  it("adds missing default setting sources to existing default roles", async () => {
    const store = new SqliteRoleConfigStore(createSqliteConnection(":memory:"));
    await store.upsert({
      name: "developer",
      systemPrompt: "Custom developer prompt",
      allowedTools: ["Read", "Edit", "Write", "Bash"],
      permissionMode: "bypassPermissions",
      capabilities: ["workspace_read", "workspace_mutate", "knowledge_base_update"],
      settingSources: ["project", "local"]
    });

    await seedDefaultRoles(store);

    await expect(store.getByName("developer")).resolves.toEqual(
      expect.objectContaining({
        name: "developer",
        systemPrompt: "Custom developer prompt",
        settingSources: ["project", "local", "user"]
      })
    );
  });
});
