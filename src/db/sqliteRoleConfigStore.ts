import type Database from "better-sqlite3";
import type { RoleConfigStore, StoredRoleConfig } from "../core/ports.js";

type RoleRow = {
  readonly name: string;
  readonly system_prompt: string;
  readonly allowed_tools: string;
  readonly permission_mode: string;
  readonly max_turns: number | null;
  readonly model: string | null;
};

function rowToConfig(row: RoleRow): StoredRoleConfig {
  return {
    name: row.name,
    systemPrompt: row.system_prompt,
    allowedTools: JSON.parse(row.allowed_tools) as string[],
    permissionMode: row.permission_mode as StoredRoleConfig["permissionMode"],
    ...(row.max_turns !== null ? { maxTurns: row.max_turns } : {}),
    ...(row.model !== null ? { model: row.model } : {}),
  };
}

export class SqliteRoleConfigStore implements RoleConfigStore {
  public constructor(private readonly db: Database.Database) {}

  public getAll(): Promise<readonly StoredRoleConfig[]> {
    const rows = this.db.prepare("SELECT name, system_prompt, allowed_tools, permission_mode, max_turns, model FROM roles ORDER BY name").all() as RoleRow[];
    return Promise.resolve(rows.map(rowToConfig));
  }

  public getByName(name: string): Promise<StoredRoleConfig | undefined> {
    const row = this.db.prepare("SELECT name, system_prompt, allowed_tools, permission_mode, max_turns, model FROM roles WHERE name = ?").get(name) as RoleRow | undefined;
    return Promise.resolve(row === undefined ? undefined : rowToConfig(row));
  }

  public upsert(config: StoredRoleConfig): Promise<void> {
    this.db.prepare(`
      INSERT INTO roles (name, system_prompt, allowed_tools, permission_mode, max_turns, model, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
      ON CONFLICT(name) DO UPDATE SET
        system_prompt = excluded.system_prompt,
        allowed_tools = excluded.allowed_tools,
        permission_mode = excluded.permission_mode,
        max_turns = excluded.max_turns,
        model = excluded.model,
        updated_at = datetime('now')
    `).run(
      config.name,
      config.systemPrompt,
      JSON.stringify(config.allowedTools),
      config.permissionMode,
      config.maxTurns ?? null,
      config.model ?? null,
    );
    return Promise.resolve();
  }

  public deleteByName(name: string): Promise<boolean> {
    const result = this.db.prepare("DELETE FROM roles WHERE name = ?").run(name);
    return Promise.resolve(result.changes > 0);
  }
}
