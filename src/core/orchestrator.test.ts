import { describe, expect, it } from "vitest";
import type {
  AuthorizationService,
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  KnowledgeWorkspaceResolver
} from "./ports.js";
import { AgentOrchestrator } from "./orchestrator.js";

describe("AgentOrchestrator", () => {
  it("returns a safe error message when the runtime fails", async () => {
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: viewerAuthorization,
      agentRuntime: {
        name: "failing",
        run() {
          return Promise.reject(new Error('401 {"error":{"message":"invalid api key"}}'));
        },
        disposeSession() {
          return Promise.resolve();
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
      authorization: viewerAuthorization,
      agentRuntime: {
        name: "runtime",
        run() {
          runtimeCalled = true;
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      }
    });

    const response = await orchestrator.handle(testMessage("/new"));

    expect(response.text).toContain("New conversation started");
    expect(runtimeCalled).toBe(false);
  });

  it("does not route mutate requests to any runtime for viewers", async () => {
    let readRuntimeCalled = false;
    let codeRuntimeCalled = false;
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: viewerAuthorization,
      agentRuntime: {
        name: "runtime",
        run() {
          readRuntimeCalled = true;
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      codeChangeRuntime: {
        name: "code-runtime",
        run() {
          codeRuntimeCalled = true;
          return Promise.resolve({ text: "code runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      }
    });

    const response = await orchestrator.handle(testMessage("重构订单系统"));

    expect(response.text).toContain("修改请求");
    expect(response.text).toContain("不能直接修改文件");
    expect(readRuntimeCalled).toBe(false);
    expect(codeRuntimeCalled).toBe(false);
  });

  it("routes mutate requests through the code change runtime for developers", async () => {
    let readRuntimeCalled = false;
    const requests: string[] = [];
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: developerAuthorization,
      agentRuntime: {
        name: "runtime",
        run() {
          readRuntimeCalled = true;
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      codeChangeRuntime: {
        name: "code-runtime",
        run(request) {
          requests.push(request.text);
          return Promise.resolve({ text: "code change complete" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      }
    });

    const response = await orchestrator.handle(testMessage("重构订单系统", "developer-1"));

    expect(response.text).toBe("code change complete");
    expect(requests).toEqual(["重构订单系统"]);
    expect(readRuntimeCalled).toBe(false);
  });

  it("reuses stored runtime sessions for follow-up messages", async () => {
    const store = new MemorySessionStore();
    const calls: (string | undefined)[] = [];
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: viewerAuthorization,
      sessionStore: store,
      agentRuntime: {
        name: "runtime",
        run(request) {
          calls.push(request.sessionId);
          return Promise.resolve({ text: "runtime", sessionId: "session-1" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      }
    });

    await orchestrator.handle(testMessage("hello"));
    await orchestrator.handle(testMessage("follow up", "user-1", "2"));

    expect(calls).toEqual([undefined, "session-1"]);
  });

  it("passes stored message history to the runtime and appends new turns", async () => {
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    await historyStore.append(sessionKey, [
      { role: "user", content: "first question" },
      { role: "assistant", content: "first answer" }
    ]);
    const histories: unknown[] = [];
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: viewerAuthorization,
      historyStore,
      agentRuntime: {
        name: "runtime",
        run(request) {
          histories.push(request.history);
          return Promise.resolve({ text: "second answer" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      }
    });

    await orchestrator.handle(testMessage("second question", "user-1", "2"));

    expect(histories).toEqual([
      [
        { role: "user", content: "first question" },
        { role: "assistant", content: "first answer" }
      ]
    ]);
    await expect(historyStore.get(sessionKey)).resolves.toEqual([
      { role: "user", content: "first question" },
      { role: "assistant", content: "first answer" },
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
      authorization: viewerAuthorization,
      sessionStore: store,
      historyStore,
      agentRuntime: {
        name: "runtime",
        run() {
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession(sessionId) {
          archivedSessions.push(sessionId);
          return Promise.resolve();
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
      authorization: viewerAuthorization,
      agentRuntime: {
        name: "failing",
        run() {
          return Promise.reject(new Error("Network timeout"));
        },
        disposeSession() {
          return Promise.resolve();
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

const viewerAuthorization: AuthorizationService = {
  roleFor() {
    return Promise.resolve("viewer");
  },
  can(_user, action) {
    return Promise.resolve(
      action === "mutate"
        ? { allowed: false, reason: "viewer cannot mutate" }
        : { allowed: true }
    );
  }
};

const developerAuthorization: AuthorizationService = {
  roleFor() {
    return Promise.resolve("developer");
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
