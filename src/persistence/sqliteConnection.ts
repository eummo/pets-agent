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
  `CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_key   TEXT PRIMARY KEY NOT NULL,
    channel       TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    chat_id       TEXT,
    session_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
  `CREATE TABLE IF NOT EXISTS conversation_histories (
    session_key    TEXT PRIMARY KEY NOT NULL,
    channel        TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    chat_id        TEXT,
    messages_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
  `CREATE TABLE IF NOT EXISTS conversation_history_archives (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key    TEXT NOT NULL,
    channel        TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    chat_id        TEXT,
    messages_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    archived_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
  `CREATE TABLE IF NOT EXISTS cron_jobs (
    id              TEXT PRIMARY KEY NOT NULL,
    name            TEXT NOT NULL,
    schedule_json   TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    workspace_path  TEXT NOT NULL,
    role            TEXT,
    enabled         INTEGER NOT NULL CHECK (enabled IN (0,1)),
    delivery_json   TEXT NOT NULL,
    timeout_ms      INTEGER,
    silent_on_empty INTEGER CHECK (silent_on_empty IN (0,1)),
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
  )`,
  `CREATE TABLE IF NOT EXISTS cron_run_state (
    job_id           TEXT PRIMARY KEY NOT NULL,
    next_run_at      TEXT,
    last_result_json TEXT,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY(job_id) REFERENCES cron_jobs(id) ON DELETE CASCADE
  )`
];

const ROLES_METADATA_COLUMNS = [
  { name: "capabilities", definition: "TEXT" },
  { name: "skills", definition: "TEXT" },
  { name: "setting_sources", definition: "TEXT" },
  { name: "enable_workflows", definition: "INTEGER" },
  { name: "plan_mode_instructions", definition: "TEXT" }
] as const;

const FEEDBACK_METADATA_COLUMNS = [
  { name: "channel", definition: "TEXT" },
  { name: "message_id", definition: "TEXT" },
  { name: "workspace_path", definition: "TEXT" },
  { name: "intent_type", definition: "TEXT" },
  { name: "role_name", definition: "TEXT" }
] as const;

const CREATE_INDEX_MIGRATIONS = [
  "CREATE INDEX IF NOT EXISTS idx_feedback_status_id ON feedback(status, id DESC)",
  "CREATE INDEX IF NOT EXISTS idx_feedback_user_id_id ON feedback(user_id, id DESC)",
  "CREATE INDEX IF NOT EXISTS idx_feedback_workspace_path_id ON feedback(workspace_path, id DESC)",
  "CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user ON conversation_sessions(user_id, updated_at DESC)",
  "CREATE INDEX IF NOT EXISTS idx_conversation_histories_user ON conversation_histories(user_id, updated_at DESC)",
  "CREATE INDEX IF NOT EXISTS idx_conversation_history_archives_key ON conversation_history_archives(session_key, id DESC)",
  "CREATE INDEX IF NOT EXISTS idx_conversation_history_archives_archived_at ON conversation_history_archives(archived_at)",
  "CREATE INDEX IF NOT EXISTS idx_cron_jobs_enabled_updated ON cron_jobs(enabled, updated_at DESC)"
] as const;

export function createSqliteConnection(dbPath: string): Database.Database {
  mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);

  // Enable WAL mode for better concurrent read performance
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");

  // Run migrations inside a transaction
  const runMigrations = db.transaction(() => {
    for (const sql of CREATE_TABLE_MIGRATIONS) {
      db.exec(sql);
    }
    migrateRolesMetadataColumns(db);
    migrateFeedbackMetadataColumns(db);
    for (const sql of CREATE_INDEX_MIGRATIONS) {
      db.exec(sql);
    }
  });
  runMigrations();

  return db;
}

function migrateRolesMetadataColumns(db: Database.Database): void {
  addMissingColumns(db, "roles", ROLES_METADATA_COLUMNS);
}

function migrateFeedbackMetadataColumns(db: Database.Database): void {
  addMissingColumns(db, "feedback", FEEDBACK_METADATA_COLUMNS);
}

function addMissingColumns(
  db: Database.Database,
  table: string,
  columns: readonly { readonly name: string; readonly definition: string }[]
): void {
  const existing = new Set(
    (db.prepare(`PRAGMA table_info(${table})`).all() as { readonly name: string }[]).map(
      (c) => c.name
    )
  );

  for (const column of columns) {
    if (!existing.has(column.name)) {
      db.exec(`ALTER TABLE ${table} ADD COLUMN ${column.name} ${column.definition}`);
    }
  }
}
