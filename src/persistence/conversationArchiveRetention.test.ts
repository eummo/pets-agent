import { describe, expect, it, vi, afterEach } from "vitest";
import { createSqliteConnection } from "./sqliteConnection.js";
import {
  cleanupExpiredConversationArchives,
  startConversationArchiveRetention
} from "./conversationArchiveRetention.js";

describe("conversation archive retention", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("deletes archives older than the configured retention window", () => {
    const db = createSqliteConnection(":memory:");
    insertArchive(db, "old", "2026-01-01T00:00:00.000Z");
    insertArchive(db, "fresh", "2026-06-01T00:00:00.000Z");

    const deletedCount = cleanupExpiredConversationArchives({
      db,
      retentionDays: 30,
      now: () => new Date("2026-06-08T00:00:00.000Z")
    });

    expect(deletedCount).toBe(1);
    const rows = db
      .prepare("SELECT session_key FROM conversation_history_archives ORDER BY session_key")
      .all() as { readonly session_key: string }[];
    expect(rows.map((row) => row.session_key)).toEqual(["fresh"]);
  });

  it("logs cleanup results when the retention worker starts", async () => {
    vi.useFakeTimers();
    const db = createSqliteConnection(":memory:");
    insertArchive(db, "old", "2026-01-01T00:00:00.000Z");
    const write = vi.fn(() => Promise.resolve());

    const handle = startConversationArchiveRetention({
      db,
      retentionDays: 30,
      cleanupIntervalMs: 60_000,
      logger: { write },
      now: () => new Date("2026-06-08T00:00:00.000Z")
    });
    await vi.runOnlyPendingTimersAsync();

    expect(write).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "conversation.archive_retention.cleaned",
        retentionDays: 30,
        deletedCount: 1
      })
    );

    handle.stop();
  });
});

function insertArchive(
  db: ReturnType<typeof createSqliteConnection>,
  sessionKey: string,
  archivedAt: string
): void {
  db.prepare(
    `
    INSERT INTO conversation_history_archives (
      session_key, channel, user_id, workspace_path, chat_id, messages_json, created_at, updated_at, archived_at
    )
    VALUES (?, 'dev-browser', 'user-1', 'D:/workspace', NULL, '[]', ?, ?, ?)
  `
  ).run(sessionKey, archivedAt, archivedAt, archivedAt);
}
