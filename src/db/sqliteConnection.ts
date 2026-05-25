import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import path from "node:path";

const CREATE_TABLE_MIGRATIONS = [
  `CREATE TABLE IF NOT EXISTS roles (
    name            TEXT PRIMARY KEY NOT NULL,
    system_prompt   TEXT NOT NULL,
    allowed_tools   TEXT NOT NULL,
    permission_mode TEXT NOT NULL,
    max_turns       INTEGER,
    model           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
  `CREATE TABLE IF NOT EXISTS feedback (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL,
    user_message         TEXT NOT NULL,
    conversation_context TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','reviewed','resolved')),
    created_at           TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
];

const ROLES_METADATA_COLUMNS = [
  { name: "capabilities", definition: "TEXT" },
] as const;

const FEEDBACK_METADATA_COLUMNS = [
  { name: "channel", definition: "TEXT" },
  { name: "message_id", definition: "TEXT" },
  { name: "workspace_path", definition: "TEXT" },
  { name: "intent_type", definition: "TEXT" },
  { name: "role_name", definition: "TEXT" },
] as const;

export function createSqliteConnection(dbPath: string): Database.Database {
  mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);

  // Enable WAL mode for better concurrent read performance
  db.pragma("journal_mode = WAL");

  // Run migrations inside a transaction
  const runMigrations = db.transaction(() => {
    for (const sql of CREATE_TABLE_MIGRATIONS) {
      db.exec(sql);
    }
    migrateRolesMetadataColumns(db);
    migrateFeedbackMetadataColumns(db);
  });
  runMigrations();

  return db;
}

function migrateRolesMetadataColumns(db: Database.Database): void {
  const columns = db.prepare("PRAGMA table_info(roles)").all() as { readonly name: string }[];
  const existingColumnNames = new Set(columns.map((column) => column.name));

  for (const column of ROLES_METADATA_COLUMNS) {
    if (!existingColumnNames.has(column.name)) {
      db.exec(`ALTER TABLE roles ADD COLUMN ${column.name} ${column.definition}`);
    }
  }
}

function migrateFeedbackMetadataColumns(db: Database.Database): void {
  const columns = db.prepare("PRAGMA table_info(feedback)").all() as { readonly name: string }[];
  const existingColumnNames = new Set(columns.map((column) => column.name));

  for (const column of FEEDBACK_METADATA_COLUMNS) {
    if (!existingColumnNames.has(column.name)) {
      db.exec(`ALTER TABLE feedback ADD COLUMN ${column.name} ${column.definition}`);
    }
  }
}
