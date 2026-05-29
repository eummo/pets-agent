import type Database from "better-sqlite3";
import { z } from "zod";
import type { RoleConfigStore, StoredRoleConfig } from "../auth/index.js";

type RoleRow = {
  readonly name: string;
  readonly system_prompt: string;
  readonly allowed_tools: string;
  readonly permission_mode: string;
  readonly max_turns: number | null;
  readonly model: string | null;
  readonly capabilities: string | null;
  readonly skills: string | null;
  readonly setting_sources: string | null;
  readonly updated_at: string;
};

const roleCapabilitySchema = z.enum([
  "workspace_read",
  "workspace_mutate",
  "knowledge_base_update",
  "feedback_view",
  "feedback_manage",
  "roles_manage",
  "cron_manage",
]);

const permissionModeSchema = z.enum(["auto", "dontAsk", "acceptEdits", "bypassPermissions"]);
const allowedToolsSchema = z.array(z.string().min(1));
const capabilitiesSchema = z.array(roleCapabilitySchema);
const skillsSchema = z.union([z.array(z.string().min(1)), z.literal("all")]);
const settingSourcesSchema = z.array(z.enum(["user", "project", "local"]));

function rowToConfig(row: RoleRow): StoredRoleConfig {
  const capabilities = row.capabilities !== null
    ? capabilitiesSchema.parse(JSON.parse(row.capabilities))
    : undefined;
  const skills = row.skills !== null
    ? skillsSchema.parse(JSON.parse(row.skills))
    : undefined;
  const settingSources = row.setting_sources !== null
    ? settingSourcesSchema.parse(JSON.parse(row.setting_sources))
    : undefined;
  return {
    name: row.name,
    systemPrompt: row.system_prompt,
    allowedTools: allowedToolsSchema.parse(JSON.parse(row.allowed_tools)),
    permissionMode: permissionModeSchema.parse(row.permission_mode),
    ...(row.max_turns !== null && { maxTurns: row.max_turns }),
    ...(row.model !== null && { model: row.model }),
    ...(capabilities !== undefined && { capabilities }),
    ...(skills !== undefined && { skills }),
    ...(settingSources !== undefined && { settingSources }),
    updatedAt: row.updated_at,
  };
}

export class SqliteRoleConfigStore implements RoleConfigStore {
  public constructor(private readonly db: Database.Database) {}

  public getAll(): Promise<readonly StoredRoleConfig[]> {
    const rows = this.db.prepare("SELECT name, system_prompt, allowed_tools, permission_mode, max_turns, model, capabilities, skills, setting_sources, updated_at FROM roles ORDER BY name").all() as RoleRow[];
    return new Promise((resolve) => resolve(rows.map(rowToConfig)));
  }

  public getByName(name: string): Promise<StoredRoleConfig | undefined> {
    const row = this.db.prepare("SELECT name, system_prompt, allowed_tools, permission_mode, max_turns, model, capabilities, skills, setting_sources, updated_at FROM roles WHERE name = ?").get(name) as RoleRow | undefined;
    return new Promise((resolve) => resolve(row === undefined ? undefined : rowToConfig(row)));
  }

  public upsert(config: StoredRoleConfig): Promise<void> {
    this.db.prepare(`
      INSERT INTO roles (name, system_prompt, allowed_tools, permission_mode, max_turns, model, capabilities, skills, setting_sources, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
      ON CONFLICT(name) DO UPDATE SET
        system_prompt = excluded.system_prompt,
        allowed_tools = excluded.allowed_tools,
        permission_mode = excluded.permission_mode,
        max_turns = excluded.max_turns,
        model = excluded.model,
        capabilities = excluded.capabilities,
        skills = excluded.skills,
        setting_sources = excluded.setting_sources,
        updated_at = datetime('now', 'localtime')
    `).run(
      config.name,
      config.systemPrompt,
      JSON.stringify(config.allowedTools),
      config.permissionMode,
      config.maxTurns ?? null,
      config.model ?? null,
      config.capabilities ? JSON.stringify(config.capabilities) : null,
      config.skills ? JSON.stringify(config.skills) : null,
      config.settingSources ? JSON.stringify(config.settingSources) : null,
    );
    return Promise.resolve();
  }

  public deleteByName(name: string): Promise<boolean> {
    const result = this.db.prepare("DELETE FROM roles WHERE name = ?").run(name);
    return Promise.resolve(result.changes > 0);
  }
}

