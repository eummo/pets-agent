import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { FileConversationHistoryStore } from "./fileConversationHistoryStore.js";

describe("FileConversationHistoryStore", () => {
  it("persists message history across instances", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await new FileConversationHistoryStore(filePath).append(key, [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" }
    ]);

    await expect(new FileConversationHistoryStore(filePath).get(key)).resolves.toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" }
    ]);
  });

  it("keeps only the latest configured messages", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath, { maxMessages: 2 });
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.append(key, [
      { role: "user", content: "one" },
      { role: "assistant", content: "two" },
      { role: "user", content: "three" }
    ]);

    await expect(store.get(key)).resolves.toEqual([
      { role: "assistant", content: "two" },
      { role: "user", content: "three" }
    ]);
  });

  it("deletes stored history when explicitly requested", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath);
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.append(key, [{ role: "user", content: "hello" }]);
    await store.delete(key);

    await expect(store.get(key)).resolves.toEqual([]);
    await expect(readFile(filePath, "utf8")).resolves.not.toContain("hello");
  });

  it("archives active history without loading it into future turns", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath);
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.append(key, [
      { role: "user", content: "old question" },
      { role: "assistant", content: "old answer" }
    ]);
    await store.archive(key);

    await expect(store.get(key)).resolves.toEqual([]);
    const content = await readFile(filePath, "utf8");
    expect(content).toContain("archives");
    expect(content).toContain("old question");

    await store.append(key, [{ role: "user", content: "new question" }]);
    await expect(store.get(key)).resolves.toEqual([{ role: "user", content: "new question" }]);
  });

  it("keeps all concurrent appends for the same history", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath, { maxMessages: 50 });
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await Promise.all(
      Array.from({ length: 20 }, (_, index) =>
        store.append(key, [{ role: "user", content: `message-${index}` }])
      )
    );

    const history = await store.get(key);
    expect(history).toHaveLength(20);
    expect(new Set(history.map((message) => message.content))).toEqual(
      new Set(Array.from({ length: 20 }, (_, index) => `message-${index}`))
    );
  });

  it("compact replaces history with a summary and the last 2 messages", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath, { maxMessages: 20 });
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.append(key, [
      { role: "user", content: "question 1" },
      { role: "assistant", content: "answer 1" },
      { role: "user", content: "question 2" },
      { role: "assistant", content: "answer 2" },
      { role: "user", content: "question 3" },
      { role: "assistant", content: "answer 3" },
    ]);

    await store.compact(key, "User asked about orders and catalog.");

    const history = await store.get(key);
    expect(history).toEqual([
      { role: "assistant", content: "[Previous conversation summary]\nUser asked about orders and catalog." },
      { role: "user", content: "question 3" },
      { role: "assistant", content: "answer 3" },
    ]);
  });

  it("compact does nothing when there is no existing history", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath);
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.compact(key, "Nothing to compact.");

    await expect(store.get(key)).resolves.toEqual([]);
  });

  it("compact respects the maxMessages limit", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "history-store-"));
    const filePath = path.join(root, "history.json");
    const store = new FileConversationHistoryStore(filePath, { maxMessages: 2 });
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.append(key, [
      { role: "user", content: "question 1" },
      { role: "assistant", content: "answer 1" },
      { role: "user", content: "question 2" },
      { role: "assistant", content: "answer 2" },
    ]);

    await store.compact(key, "Summary of earlier discussion.");

    const history = await store.get(key);
    // maxMessages=2, so [summary, user:question 2, assistant:answer 2].slice(-2)
    expect(history).toEqual([
      { role: "user", content: "question 2" },
      { role: "assistant", content: "answer 2" },
    ]);
  });
});
