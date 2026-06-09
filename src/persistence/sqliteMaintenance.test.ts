import Database from "better-sqlite3";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createSqliteConnection } from "./sqliteConnection.js";
import {
  backupSqliteDatabase,
  restoreSqliteDatabase,
  verifySqliteDatabase
} from "./sqliteMaintenance.js";

describe("sqliteMaintenance", () => {
  it("backs up a SQLite database and verifies the backup", async () => {
    const root = await makeTempRoot();
    const dbPath = path.join(root, "agent.db");
    const backupPath = path.join(root, "backups", "agent.backup.db");
    const db = createSqliteConnection(dbPath);
    db.prepare(
      "INSERT INTO conversation_sessions (session_key, channel, user_id, workspace_path, session_id) VALUES (?, ?, ?, ?, ?)"
    ).run("key-1", "dev-browser", "user-1", "D:/workspace", "session-1");
    db.close();

    const result = await backupSqliteDatabase({ sourcePath: dbPath, backupPath });

    expect(result.backupPath).toBe(path.resolve(backupPath));
    expect(result.totalPages).toBeGreaterThan(0);
    await expect(verifySqliteDatabase(backupPath)).resolves.toBe("ok");
    const backupDb = new Database(backupPath, { readonly: true, fileMustExist: true });
    try {
      const row = backupDb
        .prepare("SELECT session_id FROM conversation_sessions WHERE session_key = ?")
        .get("key-1") as { readonly session_id: string } | undefined;
      expect(row?.session_id).toBe("session-1");
    } finally {
      backupDb.close();
    }
  });

  it("restores a backup only when overwrite is explicit", async () => {
    const root = await makeTempRoot();
    const sourcePath = path.join(root, "source.db");
    const backupPath = path.join(root, "source.backup.db");
    const targetPath = path.join(root, "target.db");
    const source = createSqliteConnection(sourcePath);
    source
      .prepare(
        "INSERT INTO conversation_histories (session_key, channel, user_id, workspace_path, messages_json) VALUES (?, ?, ?, ?, ?)"
      )
      .run(
        "key-1",
        "dev-browser",
        "user-1",
        "D:/workspace",
        JSON.stringify([{ role: "user", content: "hello" }])
      );
    source.close();
    await backupSqliteDatabase({ sourcePath, backupPath });

    const target = createSqliteConnection(targetPath);
    target.close();
    await expect(restoreSqliteDatabase({ backupPath, targetPath })).rejects.toThrow("--force");

    await expect(restoreSqliteDatabase({ backupPath, targetPath, force: true })).resolves.toEqual({
      backupPath: path.resolve(backupPath),
      targetPath: path.resolve(targetPath),
      integrityCheck: "ok"
    });
    const restored = new Database(targetPath, { readonly: true, fileMustExist: true });
    try {
      const row = restored
        .prepare("SELECT messages_json FROM conversation_histories WHERE session_key = ?")
        .get("key-1") as { readonly messages_json: string } | undefined;
      expect(JSON.parse(row?.messages_json ?? "[]")).toEqual([{ role: "user", content: "hello" }]);
    } finally {
      restored.close();
    }
  });

  it("rejects corrupt backup files before restore", async () => {
    const root = await makeTempRoot();
    const backupPath = path.join(root, "corrupt.db");
    await mkdir(path.dirname(backupPath), { recursive: true });
    await writeFile(backupPath, "not sqlite");

    await expect(
      restoreSqliteDatabase({
        backupPath,
        targetPath: path.join(root, "target.db"),
        force: true
      })
    ).rejects.toThrow("SQLite integrity check failed");
    await expect(readFile(path.join(root, "target.db"))).rejects.toThrow();
  });
});

async function makeTempRoot(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "pets-agent-sqlite-maintenance-"));
}
