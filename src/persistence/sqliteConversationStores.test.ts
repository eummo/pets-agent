import { describe, expect, it } from "vitest";
import { createSqliteConnection } from "./sqliteConnection.js";
import {
  SqliteConversationHistoryStore,
  SqliteConversationSessionStore
} from "./sqliteConversationStores.js";
import type { ConversationSessionKey } from "./index.js";

const key: ConversationSessionKey = {
  channel: "dev-browser",
  userId: "user-1",
  workspacePath: "D:/workspace",
  chatId: "chat-1"
};

describe("SqliteConversationSessionStore", () => {
  it("stores, updates, and deletes session ids", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteConversationSessionStore(db);

    await expect(store.get(key)).resolves.toBeUndefined();
    await store.set(key, "session-1");
    await expect(store.get(key)).resolves.toBe("session-1");

    await store.set(key, "session-2");
    await expect(store.get(key)).resolves.toBe("session-2");

    await store.delete(key);
    await expect(store.get(key)).resolves.toBeUndefined();
  });
});

describe("SqliteConversationHistoryStore", () => {
  it("appends and trims messages by maxMessages", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteConversationHistoryStore(db, { maxMessages: 2 });

    await store.append(key, [
      { role: "user", content: "first" },
      { role: "assistant", content: "second" }
    ]);
    await store.append(key, [{ role: "user", content: "third" }]);

    await expect(store.get(key)).resolves.toEqual([
      { role: "assistant", content: "second" },
      { role: "user", content: "third" }
    ]);
  });

  it("compacts history to a summary plus recent messages", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteConversationHistoryStore(db, { maxMessages: 4 });

    await store.append(key, [
      { role: "user", content: "one" },
      { role: "assistant", content: "two" },
      { role: "user", content: "three" }
    ]);
    await store.compact(key, "earlier conversation");

    await expect(store.get(key)).resolves.toEqual([
      { role: "assistant", content: "[Previous conversation summary]\nearlier conversation" },
      { role: "assistant", content: "two" },
      { role: "user", content: "three" }
    ]);
  });

  it("deletes active history", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteConversationHistoryStore(db);

    await store.append(key, [{ role: "user", content: "hello" }]);
    await store.delete(key);

    await expect(store.get(key)).resolves.toEqual([]);
  });

  it("archives active history and clears the active conversation", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteConversationHistoryStore(db);

    await store.append(key, [{ role: "user", content: "hello" }]);
    await store.archive(key);

    await expect(store.get(key)).resolves.toEqual([]);
    const archiveCount = db
      .prepare("SELECT COUNT(*) AS count FROM conversation_history_archives")
      .get() as { readonly count: number };
    expect(archiveCount.count).toBe(1);
  });
});
