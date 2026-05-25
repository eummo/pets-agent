import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { FileConversationSessionStore } from "./fileConversationSessionStore.js";

describe("FileConversationSessionStore", () => {
  it("persists session mappings across instances", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "session-store-"));
    const filePath = path.join(root, "sessions.json");
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await new FileConversationSessionStore(filePath).set(key, "session-1");

    await expect(new FileConversationSessionStore(filePath).get(key)).resolves.toBe("session-1");
  });

  it("deletes session mappings", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "session-store-"));
    const filePath = path.join(root, "sessions.json");
    const store = new FileConversationSessionStore(filePath);
    const key = { channel: "dev-browser", userId: "user-1", workspacePath: "D:/kb" };

    await store.set(key, "session-1");
    await store.delete(key);

    await expect(store.get(key)).resolves.toBeUndefined();
    await expect(readFile(filePath, "utf8")).resolves.not.toContain("session-1");
  });

  it("keeps all concurrent session writes", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "session-store-"));
    const filePath = path.join(root, "sessions.json");
    const store = new FileConversationSessionStore(filePath);
    const keys = Array.from({ length: 20 }, (_, index) => ({
      channel: "dev-browser",
      userId: `user-${index}`,
      workspacePath: "D:/kb",
    }));

    await Promise.all(keys.map((key, index) => store.set(key, `session-${index}`)));

    await Promise.all(
      keys.map(async (key, index) => {
        await expect(store.get(key)).resolves.toBe(`session-${index}`);
      })
    );
  });
});
