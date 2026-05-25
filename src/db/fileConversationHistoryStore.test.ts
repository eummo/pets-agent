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
});
