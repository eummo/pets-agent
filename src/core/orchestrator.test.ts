import { describe, expect, it } from "vitest";
import type {
  AuthorizationService,
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackEntry,
  FeedbackStore,
  IntentDetectionService,
  KnowledgeWorkspaceResolver
} from "./ports.js";
import { AgentOrchestrator } from "./orchestrator.js";

describe("AgentOrchestrator", () => {
  it("returns a safe error message when the runtime fails", async () => {
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      agentRuntimes: {
        reviewer: {
          name: "failing",
          run() {
            return Promise.reject(new Error('401 {"error":{"message":"invalid api key"}}'));
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toContain("Invalid API key");
    expect(response.text).not.toContain("sk-");
  });

  it("handles /new without calling the runtime", async () => {
    let runtimeCalled = false;
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      agentRuntimes: {
        reviewer: {
          name: "runtime",
          run() {
            runtimeCalled = true;
            return Promise.resolve({ text: "runtime" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("/new"));

    expect(response.text).toContain("New conversation started");
    expect(runtimeCalled).toBe(false);
  });

  it("uses reviewer runtime for reviewer/viewer users", async () => {
    let reviewerCalled = false;
    let developerCalled = false;
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      agentRuntimes: {
        reviewer: {
          name: "reviewer",
          run() {
            reviewerCalled = true;
            return Promise.resolve({ text: "reviewer response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        },
        developer: {
          name: "developer",
          run() {
            developerCalled = true;
            return Promise.resolve({ text: "developer response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toBe("reviewer response");
    expect(reviewerCalled).toBe(true);
    expect(developerCalled).toBe(false);
  });

  it("uses developer runtime for developer users", async () => {
    let reviewerCalled = false;
    let developerCalled = false;
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: developerAuthorization,
      agentRuntimes: {
        reviewer: {
          name: "reviewer",
          run() {
            reviewerCalled = true;
            return Promise.resolve({ text: "reviewer response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        },
        developer: {
          name: "developer",
          run() {
            developerCalled = true;
            return Promise.resolve({ text: "developer response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("refactor the code"));

    expect(response.text).toBe("developer response");
    expect(developerCalled).toBe(true);
    expect(reviewerCalled).toBe(false);
  });

  it("records feedback instead of running runtime when a reviewer asks to mutate", async () => {
    let runtimeCalled = false;
    const feedbackStore = new MemoryFeedbackStore();
    const intentDetection: IntentDetectionService = {
      detectIntent() {
        return Promise.resolve({ type: "update_kb" });
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      feedbackStore,
      intentDetection,
      agentRuntimes: {
        reviewer: {
          name: "reviewer",
          run() {
            runtimeCalled = true;
            return Promise.resolve({ text: "runtime response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("请帮我更新知识库", "user-1", "message-1"));

    expect(response.text).toContain("感谢您的反馈");
    expect(runtimeCalled).toBe(false);
    expect(feedbackStore.entries).toEqual([
      expect.objectContaining({
        userId: "user-1",
        channel: "test",
        messageId: "message-1",
        workspacePath: "D:/kb",
        intentType: "update_kb",
        roleName: "reviewer",
        userMessage: "请帮我更新知识库",
        status: "pending",
      })
    ]);
  });

  it("routes custom roles to their matching runtime", async () => {
    let customCalled = false;
    const customAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("custom-reader");
      },
      can() {
        return Promise.resolve({ allowed: true });
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: customAuthorization,
      agentRuntimes: {
        "custom-reader": {
          name: "custom-reader",
          run() {
            customCalled = true;
            return Promise.resolve({ text: "custom response" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toBe("custom response");
    expect(customCalled).toBe(true);
  });

  it("reuses stored runtime sessions for follow-up messages", async () => {
    const store = new MemorySessionStore();
    const calls: (string | undefined)[] = [];
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      sessionStore: store,
      agentRuntimes: {
        reviewer: {
          name: "runtime",
          run(request) {
            calls.push(request.sessionId);
            return Promise.resolve({ text: "runtime", sessionId: "session-1" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    await orchestrator.handle(testMessage("hello"));
    await orchestrator.handle(testMessage("follow up", "user-1", "2"));

    expect(calls).toEqual([undefined, "session-1"]);
  });

  it("appends new turns to history store", async () => {
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      historyStore,
      agentRuntimes: {
        reviewer: {
          name: "runtime",
          run() {
            return Promise.resolve({ text: "second answer" });
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    await orchestrator.handle(testMessage("second question", "user-1", "2"));

    await expect(historyStore.get(sessionKey)).resolves.toEqual([
      { role: "user", content: "second question" },
      { role: "assistant", content: "second answer" }
    ]);
  });

  it("archives active history and starts a fresh current conversation for /new", async () => {
    const store = new MemorySessionStore();
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    await store.set(sessionKey, "session-1");
    await historyStore.append(sessionKey, [{ role: "user", content: "hello" }]);
    const archivedSessions: string[] = [];
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      sessionStore: store,
      historyStore,
      agentRuntimes: {
        reviewer: {
          name: "runtime",
          run() {
            return Promise.resolve({ text: "runtime" });
          },
          disposeSession(sessionId) {
            archivedSessions.push(sessionId);
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("/new"));

    expect(response.text).toContain("New conversation started");
    expect(archivedSessions).toEqual(["session-1"]);
    await expect(store.get(sessionKey)).resolves.toBeUndefined();
    await expect(historyStore.get(sessionKey)).resolves.toEqual([]);
    expect(historyStore.archivedMessages).toEqual([[{ role: "user", content: "hello" }]]);
  });

  it("returns a generic error for non-API-key runtime failures", async () => {
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      agentRuntimes: {
        reviewer: {
          name: "failing",
          run() {
            return Promise.reject(new Error("Network timeout"));
          },
          disposeSession() {
            return Promise.resolve();
          }
        }
      }
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toContain("Model call failed");
    expect(response.text).toContain("Network timeout");
    expect(response.text).not.toContain("Invalid API key");
  });
});

const workspaceResolver: KnowledgeWorkspaceResolver = {
  resolve() {
    return Promise.resolve([{ kind: "knowledge-base", id: "kb", path: "D:/kb" }]);
  }
};

const reviewerAuthorization: AuthorizationService = {
  roleFor() {
    return Promise.resolve("reviewer" as const);
  },
  can(_user, action) {
    return Promise.resolve(
      action === "mutate"
        ? { allowed: false, reason: "reviewer cannot mutate" }
        : { allowed: true }
    );
  }
};

const developerAuthorization: AuthorizationService = {
  roleFor() {
    return Promise.resolve("developer" as const);
  },
  can() {
    return Promise.resolve({ allowed: true });
  }
};

function testMessage(text: string, userId = "user-1", id = "1") {
  return {
    id,
    channel: "test",
    user: { id: userId },
    text,
    receivedAt: new Date()
  };
}

class MemorySessionStore implements ConversationSessionStore {
  private readonly sessions = new Map<string, string>();

  public get(key: ConversationSessionKey): Promise<string | undefined> {
    return Promise.resolve(this.sessions.get(JSON.stringify(key)));
  }

  public set(key: ConversationSessionKey, sessionId: string): Promise<void> {
    this.sessions.set(JSON.stringify(key), sessionId);
    return Promise.resolve();
  }

  public delete(key: ConversationSessionKey): Promise<void> {
    this.sessions.delete(JSON.stringify(key));
    return Promise.resolve();
  }
}

class MemoryHistoryStore implements ConversationHistoryStore {
  private readonly histories = new Map<string, readonly { readonly role: "user" | "assistant"; readonly content: string }[]>();
  public readonly archivedMessages: { readonly role: "user" | "assistant"; readonly content: string }[][] = [];

  public get(key: ConversationSessionKey): Promise<readonly { readonly role: "user" | "assistant"; readonly content: string }[]> {
    return Promise.resolve(this.histories.get(JSON.stringify(key)) ?? []);
  }

  public append(
    key: ConversationSessionKey,
    messages: readonly { readonly role: "user" | "assistant"; readonly content: string }[]
  ): Promise<void> {
    const keyText = JSON.stringify(key);
    this.histories.set(keyText, [...(this.histories.get(keyText) ?? []), ...messages]);
    return Promise.resolve();
  }

  public delete(key: ConversationSessionKey): Promise<void> {
    this.histories.delete(JSON.stringify(key));
    return Promise.resolve();
  }

  public archive(key: ConversationSessionKey): Promise<void> {
    const keyText = JSON.stringify(key);
    const messages = this.histories.get(keyText) ?? [];
    if (messages.length > 0) {
      this.archivedMessages.push([...messages]);
      this.histories.delete(keyText);
    }
    return Promise.resolve();
  }
}

class MemoryFeedbackStore implements FeedbackStore {
  public readonly entries: FeedbackEntry[] = [];

  public save(entry: FeedbackEntry): Promise<number> {
    this.entries.push({ ...entry, id: this.entries.length + 1 });
    return Promise.resolve(this.entries.length);
  }

  public updateStatus(id: number, status: FeedbackEntry["status"]): Promise<void> {
    const entry = this.entries.find((item) => item.id === id);
    if (entry !== undefined) {
      this.entries.splice(this.entries.indexOf(entry), 1, { ...entry, status });
    }
    return Promise.resolve();
  }

  public getAll(): Promise<readonly FeedbackEntry[]> {
    return Promise.resolve(this.entries);
  }
}
