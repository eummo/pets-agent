import { describe, it, expect, beforeEach } from "vitest";
import { createSqliteConnection } from "./sqliteConnection.js";
import { SqliteFeedbackStore } from "./sqliteFeedbackStore.js";
import type { FeedbackEntry } from "./index.js";

describe("SqliteFeedbackStore", () => {
  let store: SqliteFeedbackStore;

  beforeEach(() => {
    const db = createSqliteConnection(":memory:");
    store = new SqliteFeedbackStore(db);
  });

  it("starts empty", async () => {
    const all = await store.getAll();
    expect(all).toEqual([]);
  });

  it("saves and retrieves feedback entries", async () => {
    const entry: FeedbackEntry = {
      userId: "user1",
      channel: "dev-browser",
      messageId: "message-1",
      workspacePath: "D:/kb",
      intentType: "update_kb",
      roleName: "reviewer",
      userMessage: "I want to update the documentation",
      conversationContext: "Previous: How does auth work?",
      status: "pending",
    };

    const id = await store.save(entry);
    expect(id).toBeGreaterThan(0);

    const all = await store.getAll();
    expect(all).toHaveLength(1);
    expect(all[0]?.userId).toBe("user1");
    expect(all[0]?.channel).toBe("dev-browser");
    expect(all[0]?.messageId).toBe("message-1");
    expect(all[0]?.workspacePath).toBe("D:/kb");
    expect(all[0]?.intentType).toBe("update_kb");
    expect(all[0]?.roleName).toBe("reviewer");
    expect(all[0]?.userMessage).toBe("I want to update the documentation");
    expect(all[0]?.conversationContext).toBe("Previous: How does auth work?");
    expect(all[0]?.status).toBe("pending");
    expect(all[0]?.createdAt).toBeTruthy();
  });

  it("updates feedback status", async () => {
    const id = await store.save({
      userId: "user1",
      userMessage: "Update request",
      conversationContext: "",
      status: "pending",
    });

    await store.updateStatus(id, "reviewed");

    const all = await store.getAll();
    expect(all[0]?.status).toBe("reviewed");
  });

  it("lists entries in reverse order (newest first)", async () => {
    await store.save({
      userId: "user1",
      userMessage: "First",
      conversationContext: "",
      status: "pending",
    });
    await store.save({
      userId: "user2",
      userMessage: "Second",
      conversationContext: "",
      status: "pending",
    });

    const all = await store.getAll();
    expect(all[0]?.userMessage).toBe("Second");
    expect(all[1]?.userMessage).toBe("First");
  });

  it("resolves feedback to resolved status", async () => {
    const id = await store.save({
      userId: "user1",
      userMessage: "Update request",
      conversationContext: "Context",
      status: "pending",
    });

    await store.updateStatus(id, "resolved");

    const all = await store.getAll();
    expect(all[0]?.status).toBe("resolved");
  });

  it("supports pagination and status filtering", async () => {
    await store.save({
      userId: "user1",
      userMessage: "First",
      conversationContext: "",
      status: "pending",
    });
    await store.save({
      userId: "user2",
      userMessage: "Second",
      conversationContext: "",
      status: "reviewed",
    });
    await store.save({
      userId: "user3",
      userMessage: "Third",
      conversationContext: "",
      status: "pending",
    });

    const firstPendingPage = await store.getAll({ status: "pending", limit: 1 });
    const secondPendingPage = await store.getAll({ status: "pending", limit: 1, offset: 1 });

    expect(firstPendingPage.map((entry) => entry.userMessage)).toEqual(["Third"]);
    expect(secondPendingPage.map((entry) => entry.userMessage)).toEqual(["First"]);
  });
});

