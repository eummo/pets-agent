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
});
