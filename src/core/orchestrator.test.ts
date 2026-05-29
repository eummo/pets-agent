import { describe, expect, it } from "vitest";
import type {
  AgentRequest,
  AgentRuntime,
  AgentRuntimeFactory
} from "../agent/index.js";
import type { AuthorizationService } from "../auth/index.js";
import type {
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackEntry,
  FeedbackStore
} from "../persistence/index.js";
import type { KnowledgeWorkspaceResolver } from "../workspace/index.js";
import type { UserIntent } from "../intent/index.js";
import { AgentOrchestrator } from "./orchestrator.js";
import { fallbackIntentFor } from "./intentHeuristics.js";

function stubIntentRuntime(intentType: UserIntent["type"] = "query"): AgentRuntime {
  return {
    name: "intent",
    run() {
      return Promise.resolve({ text: intentType });
    },
    disposeSession() {
      return Promise.resolve();
    },
  };
}

function stubIntentRuntimeFromMessage(): AgentRuntime {
  return {
    name: "intent",
    run(request: AgentRequest) {
      return Promise.resolve({ text: fallbackIntentFor(request.text).type });
    },
    disposeSession() {
      return Promise.resolve();
    },
  };
}

function stubRuntimeFactory(
  initialRuntimes: Record<string, AgentRuntime>,
  factoryOverrides?: Partial<AgentRuntimeFactory>,
): AgentRuntimeFactory {
  return {
    warmup() {
      return Promise.resolve(initialRuntimes);
    },
    createRuntime(role: string) {
      const existing = initialRuntimes[role];
      if (existing !== undefined) return Promise.resolve(existing);
      return Promise.resolve(undefined);
    },
    ...factoryOverrides,
  };
}

describe("AgentOrchestrator", () => {
  it("returns a safe error message when the runtime fails with API key error", async () => {
    const runtimes = {
      reviewer: {
        name: "failing",
        run() {
          return Promise.reject(new Error('401 {"error":{"message":"invalid api key"}}'));
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toContain("API key is invalid or not configured");
    expect(response.text).not.toContain("sk-");
    expect(response.text).not.toContain("ANTHROPIC");
  });

  it("handles /new without calling the runtime", async () => {
    let runtimeCalled = false;
    const runtimes = {
      reviewer: {
        name: "runtime",
        run() {
          runtimeCalled = true;
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("/new"));

    expect(response.text).toContain("New conversation started");
    expect(runtimeCalled).toBe(false);
  });

  it("uses reviewer runtime for reviewer/viewer users", async () => {
    let reviewerCalled = false;
    let developerCalled = false;
    const runtimes = {
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
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toBe("reviewer response");
    expect(reviewerCalled).toBe(true);
    expect(developerCalled).toBe(false);
  });

  it("writes internal event logs for workspace, role, intent, and runtime selection", async () => {
    const events: Record<string, unknown>[] = [];
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          return Promise.resolve({ text: "reviewer response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      eventLogger: {
        write(event) {
          events.push(event);
          return Promise.resolve();
        }
      },
    });

    await orchestrator.handle(testMessage("hello"));

    expect(events.map((event) => event["type"])).toEqual([
      "workspace.resolved",
      "role.resolved",
      "intent.classified",
      "runtime.selected",
    ]);
  });

  it("writes internal event logs when permission is denied", async () => {
    const events: Record<string, unknown>[] = [];
    const feedbackStore = new MemoryFeedbackStore();
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime("mutate"),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      feedbackStore,
      eventLogger: {
        write(event) {
          events.push(event);
          return Promise.resolve();
        }
      },
    });

    await orchestrator.handle(testMessage("modify files"));

    expect(events.map((event) => event["type"])).toContain("permission.denied");
  });

  it("uses developer runtime for developer users", async () => {
    let reviewerCalled = false;
    let developerCalled = false;
    const runtimes = {
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
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: developerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("refactor the code"));

    expect(response.text).toBe("developer response");
    expect(developerCalled).toBe(true);
    expect(reviewerCalled).toBe(false);
  });

  it("records feedback instead of running runtime when a reviewer asks to mutate", async () => {
    let runtimeCalled = false;
    const feedbackStore = new MemoryFeedbackStore();
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          runtimeCalled = true;
          return Promise.resolve({ text: "runtime response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime("update_kb"),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      feedbackStore,
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

  it("uses deterministic intent fallback via fallbackIntentFor for mutate detection", async () => {
    let runtimeCalled = false;
    const feedbackStore = new MemoryFeedbackStore();
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          runtimeCalled = true;
          return Promise.resolve({ text: "runtime response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntimeFromMessage(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      feedbackStore,
    });

    const response = await orchestrator.handle(testMessage("请修改订单系统", "user-1", "message-1"));

    expect(response.text).toContain("修改请求");
    expect(runtimeCalled).toBe(false);
    expect(feedbackStore.entries).toEqual([
      expect.objectContaining({
        intentType: "mutate",
        userMessage: "请修改订单系统",
      })
    ]);
  });

  it("uses deterministic knowledge-base fallback via fallbackIntentFor for update_kb detection", async () => {
    const feedbackStore = new MemoryFeedbackStore();
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          return Promise.resolve({ text: "runtime response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntimeFromMessage(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      feedbackStore,
    });

    const response = await orchestrator.handle(testMessage("请更新知识库里的订单流程", "user-1", "message-1"));

    expect(response.text).toContain("感谢您的反馈");
    expect(feedbackStore.entries).toEqual([
      expect.objectContaining({
        intentType: "update_kb",
        userMessage: "请更新知识库里的订单流程",
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
      },
      hasCapability() {
        return Promise.resolve(false);
      }
    };
    const runtimes = {
      "custom-reader": {
        name: "custom-reader",
        run() {
          customCalled = true;
          return Promise.resolve({ text: "custom response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: customAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toBe("custom response");
    expect(customCalled).toBe(true);
  });

  it("reuses stored runtime sessions for follow-up messages", async () => {
    const store = new MemorySessionStore();
    const calls: (string | undefined)[] = [];
    const runtimes = {
      reviewer: {
        name: "runtime",
        run(request: AgentRequest) {
          calls.push(request.sessionId);
          return Promise.resolve({ text: "runtime", sessionId: "session-1" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      sessionStore: store,
    });

    await orchestrator.handle(testMessage("hello"));
    await orchestrator.handle(testMessage("follow up", "user-1", "2"));

    expect(calls).toEqual([undefined, "session-1"]);
  });

  it("appends new turns to history store", async () => {
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    const runtimes = {
      reviewer: {
        name: "runtime",
        run() {
          return Promise.resolve({ text: "second answer" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      historyStore,
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
    const runtimes = {
      reviewer: {
        name: "runtime",
        run() {
          return Promise.resolve({ text: "runtime" });
        },
        disposeSession(sessionId: string) {
          archivedSessions.push(sessionId);
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      sessionStore: store,
      historyStore,
    });

    const response = await orchestrator.handle(testMessage("/new"));

    expect(response.text).toContain("New conversation started");
    expect(archivedSessions).toEqual(["session-1"]);
    await expect(store.get(sessionKey)).resolves.toBeUndefined();
    await expect(historyStore.get(sessionKey)).resolves.toEqual([]);
    expect(historyStore.archivedMessages).toEqual([[{ role: "user", content: "hello" }]]);
  });

  it("saves full conversation context in feedback, not just the last 4 messages", async () => {
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    // Simulate 6 prior turns (12 messages: 6 user + 6 assistant)
    for (let i = 1; i <= 6; i++) {
      await historyStore.append(sessionKey, [
        { role: "user", content: `question ${i}` },
        { role: "assistant", content: `answer ${i}` },
      ]);
    }

    const feedbackStore = new MemoryFeedbackStore();
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          return Promise.resolve({ text: "runtime response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime("mutate"),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      historyStore,
      feedbackStore,
    });

    await orchestrator.handle(testMessage("请修改代码", "user-1", "msg-7"));

    expect(feedbackStore.entries).toHaveLength(1);
    const context = feedbackStore.entries[0]?.conversationContext ?? "";
    // All 6 prior turns should be present, not just the last 2
    expect(context).toContain("question 1");
    expect(context).toContain("answer 1");
    expect(context).toContain("question 6");
    expect(context).toContain("answer 6");
    // The current message that triggered the feedback should also be included
    expect(context).toContain("user: 请修改代码");
    // The denial response should be included in the conversation context
    expect(context).toContain("assistant: 我已识别到这是修改请求");
  });

  it("returns a generic error for non-API-key runtime failures", async () => {
    const runtimes = {
      reviewer: {
        name: "failing",
        run() {
          return Promise.reject(new Error("Network timeout"));
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toContain("Model call failed");
    expect(response.text).not.toContain("Network timeout");
    expect(response.text).not.toContain("Invalid API key");
  });

  it("creates missing runtime via factory when role has no pre-configured runtime", async () => {
    let factoryCalled = false;
    const adminAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("admin");
      },
      can() {
        return Promise.resolve({ allowed: true });
      },
      hasCapability() {
        return Promise.resolve(true);
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: adminAuthorization,
      runtimeFactory: {
        warmup() { return Promise.resolve({ intent: stubIntentRuntime() }); },
        createRuntime(role: string) {
          if (role === "intent") return Promise.resolve(stubIntentRuntime());
          factoryCalled = true;
          expect(role).toBe("admin");
          return Promise.resolve({
            name: "admin",
            run() {
              return Promise.resolve({ text: "admin response" });
            },
            disposeSession() {
              return Promise.resolve();
            }
          });
        }
      },
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(factoryCalled).toBe(true);
    expect(response.text).toBe("admin response");
  });

  it("recreates factory runtime when the runtime cache key changes", async () => {
    let version = "v1";
    let createCount = 0;
    const adminAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("admin");
      },
      can() {
        return Promise.resolve({ allowed: true });
      },
      hasCapability() {
        return Promise.resolve(true);
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: adminAuthorization,
      runtimeFactory: {
        warmup() { return Promise.resolve({ intent: stubIntentRuntime() }); },
        cacheKeyForRole(role: string) {
          if (role === "intent") return Promise.resolve("intent");
          return Promise.resolve(`${role}:${version}`);
        },
        createRuntime(role: string) {
          if (role === "intent") return Promise.resolve(stubIntentRuntime());
          createCount += 1;
          const runtimeVersion = version;
          return Promise.resolve({
            name: `admin-${runtimeVersion}`,
            run() {
              return Promise.resolve({ text: runtimeVersion });
            },
            disposeSession() {
              return Promise.resolve();
            }
          });
        }
      },
    });

    await expect(orchestrator.handle(testMessage("hello", "admin-1", "1"))).resolves.toEqual({ text: "v1" });
    await expect(orchestrator.handle(testMessage("hello again", "admin-1", "2"))).resolves.toEqual({ text: "v1" });
    version = "v2";
    await expect(orchestrator.handle(testMessage("after update", "admin-1", "3"))).resolves.toEqual({ text: "v2" });

    expect(createCount).toBe(2);
  });

  it("disposes the current versioned runtime when starting a new conversation", async () => {
    let version = "v1";
    const disposedSessions: string[] = [];
    const store = new MemorySessionStore();
    const sessionKey = { channel: "test", userId: "admin-1", workspacePath: "D:/kb" };
    await store.set(sessionKey, "session-1");
    const adminAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("admin");
      },
      can() {
        return Promise.resolve({ allowed: true });
      },
      hasCapability() {
        return Promise.resolve(true);
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: adminAuthorization,
      runtimeFactory: {
        warmup() { return Promise.resolve({ intent: stubIntentRuntime() }); },
        cacheKeyForRole(role: string) {
          if (role === "intent") return Promise.resolve("intent");
          return Promise.resolve(`${role}:${version}`);
        },
        createRuntime(role: string) {
          if (role === "intent") return Promise.resolve(stubIntentRuntime());
          const runtimeVersion = version;
          return Promise.resolve({
            name: `admin-${runtimeVersion}`,
            run() {
              return Promise.resolve({ text: runtimeVersion });
            },
            disposeSession(sessionId: string) {
              disposedSessions.push(`${runtimeVersion}:${sessionId}`);
              return Promise.resolve();
            }
          });
        }
      },
      sessionStore: store,
    });

    version = "v2";
    const response = await orchestrator.handle(testMessage("/new", "admin-1", "new"));

    expect(response.text).toContain("New conversation started");
    expect(disposedSessions).toEqual(["v2:session-1"]);
  });

  it("returns no-runtime error when factory returns undefined", async () => {
    const adminAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("admin");
      },
      can() {
        return Promise.resolve({ allowed: true });
      },
      hasCapability() {
        return Promise.resolve(true);
      }
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: adminAuthorization,
      runtimeFactory: {
        warmup() { return Promise.resolve({ intent: stubIntentRuntime() }); },
        createRuntime(role: string) {
          if (role === "intent") return Promise.resolve(stubIntentRuntime());
          return Promise.resolve(undefined);
        }
      },
    });

    const response = await orchestrator.handle(testMessage("hello"));

    expect(response.text).toContain("No runtime configured for role: admin");
  });

  it("isolates sessions by chatId for group chats", async () => {
    const store = new MemorySessionStore();
    const historyStore = new MemoryHistoryStore();
    const runtimes = {
      reviewer: {
        name: "runtime",
        run(request: AgentRequest) {
          return Promise.resolve({ text: "response", sessionId: request.sessionId ?? "s-new" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      sessionStore: store,
      historyStore,
    });

    // Same user, different group chats = different sessions
    await orchestrator.handle(testMessage("hello", "user-1", "1", "group-A"));
    await orchestrator.handle(testMessage("hello", "user-1", "2", "group-B"));

    const sessionKeyA: ConversationSessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb", chatId: "group-A" };
    const sessionKeyB: ConversationSessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb", chatId: "group-B" };

    await expect(store.get(sessionKeyA)).resolves.toBe("s-new");
    await expect(store.get(sessionKeyB)).resolves.toBe("s-new");

    // Verify they are stored as separate entries
    const historyA = await historyStore.get(sessionKeyA);
    const historyB = await historyStore.get(sessionKeyB);
    expect(historyA).toHaveLength(2);
    expect(historyB).toHaveLength(2);
  });

  it("carries conversation history to the new runtime when the role changes", async () => {
    const store = new MemorySessionStore();
    const historyStore = new MemoryHistoryStore();
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };
    const disposedSessions: string[] = [];
    const adminCalls: AgentRequest[] = [];
    let currentRole = "reviewer";
    const roleAuthorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve(currentRole);
      },
      can() {
        return Promise.resolve({ allowed: true });
      },
      hasCapability() {
        return Promise.resolve(true);
      }
    };
    const runtimes = {
      reviewer: {
        name: "reviewer",
        run() {
          return Promise.resolve({ text: "reviewer response", sessionId: "reviewer-session-1" });
        },
        disposeSession(sessionId: string) {
          disposedSessions.push(sessionId);
          return Promise.resolve();
        }
      },
      admin: {
        name: "admin",
        run(request: AgentRequest) {
          adminCalls.push(request);
          return Promise.resolve({ text: "admin response", sessionId: request.sessionId ?? "admin-session-1" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: roleAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      sessionStore: store,
      historyStore,
    });

    // First message as reviewer
    await orchestrator.handle(testMessage("hello", "user-1", "1"));

    // Verify reviewer session and history exist
    await expect(store.get(sessionKey)).resolves.toBe("reviewer-session-1");
    await expect(historyStore.get(sessionKey)).resolves.toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "reviewer response" }
    ]);

    // Switch role to admin
    currentRole = "admin";
    await orchestrator.handle(testMessage("admin task", "user-1", "2"));

    // Old reviewer session should be disposed
    expect(disposedSessions).toEqual(["reviewer-session-1"]);
    // Session store should have the new admin session
    await expect(store.get(sessionKey)).resolves.toBe("admin-session-1");
    // The admin runtime should receive the prior history
    expect(adminCalls).toHaveLength(1);
    expect(adminCalls[0]?.sessionId).toBeUndefined();
    expect(adminCalls[0]?.history).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "reviewer response" }
    ]);
    // History should continue to accumulate (not archived)
    await expect(historyStore.get(sessionKey)).resolves.toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "reviewer response" },
      { role: "user", content: "admin task" },
      { role: "assistant", content: "admin response" }
    ]);
    // No history should be archived
    expect(historyStore.archivedMessages).toEqual([]);
  });

  it("isolates group chat session from single chat session for the same user", async () => {
    const store = new MemorySessionStore();
    const historyStore = new MemoryHistoryStore();
    const runtimes = {
      reviewer: {
        name: "runtime",
        run() {
          return Promise.resolve({ text: "response", sessionId: "s-1" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      sessionStore: store,
      historyStore,
    });

    // User in group chat
    await orchestrator.handle(testMessage("group msg", "user-1", "1", "group-X"));
    // Same user in single chat (no chatId)
    await orchestrator.handle(testMessage("single msg", "user-1", "2"));

    const groupKey: ConversationSessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb", chatId: "group-X" };
    const singleKey: ConversationSessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };

    await expect(store.get(groupKey)).resolves.toBe("s-1");
    await expect(store.get(singleKey)).resolves.toBe("s-1");

    // Both sessions exist independently
    const groupHistory = await historyStore.get(groupKey);
    const singleHistory = await historyStore.get(singleKey);
    expect(groupHistory).toHaveLength(2);
    expect(singleHistory).toHaveLength(2);
  });

  it("calls onCompact to compact history store when SDK compresses context", async () => {
    const historyStore = new MemoryHistoryStore();
    const events: Record<string, unknown>[] = [];
    const sessionKey = { channel: "test", userId: "user-1", workspacePath: "D:/kb" };

    // Pre-fill history
    await historyStore.append(sessionKey, [
      { role: "user", content: "question 1" },
      { role: "assistant", content: "answer 1" },
      { role: "user", content: "question 2" },
      { role: "assistant", content: "answer 2" },
    ]);

    let onCompactCallback: ((summary: string) => Promise<void>) | undefined;
    const runtimes = {
      reviewer: {
        name: "runtime",
        run(request: AgentRequest) {
          onCompactCallback = request.onCompact;
          return Promise.resolve({ text: "response after compact" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      historyStore,
      eventLogger: {
        write(event) {
          events.push(event);
          return Promise.resolve();
        }
      },
    });

    await orchestrator.handle(testMessage("continue"));
    expect(onCompactCallback).toBeDefined();

    // Simulate SDK compacting context
    if (onCompactCallback === undefined) throw new Error("Expected onCompact callback");
    await onCompactCallback("User discussed orders and catalog.");

    // Verify history was compacted
    // At this point, the current turn (continue + response) has been appended after compact
    const history = await historyStore.get(sessionKey);
    expect(history).toEqual([
      { role: "assistant", content: "[Previous conversation summary]\nUser discussed orders and catalog." },
      { role: "user", content: "continue" },
      { role: "assistant", content: "response after compact" },
    ]);

    // Verify event was logged
    expect(events.some((e) => e["type"] === "context.compacted")).toBe(true);
    const compactEvent = events.find((e) => e["type"] === "context.compacted");
    expect(compactEvent).toMatchObject({
      type: "context.compacted",
      workspacePath: "D:/kb",
      summaryLength: "User discussed orders and catalog.".length,
    });
  });

  it("logs context usage event when the runtime reports token usage", async () => {
    const events: Record<string, unknown>[] = [];
    const runtimes = {
      reviewer: {
        name: "runtime",
        run() {
          return Promise.resolve({
            text: "response",
            sessionId: "s-1",
            contextUsage: {
              inputTokens: 100_000,
              outputTokens: 500,
              cacheReadTokens: 60_000,
              cacheCreationTokens: 20_000,
              contextWindow: 150_000,
              usagePercent: 67,
            },
          });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
      eventLogger: {
        write(event) {
          events.push(event);
          return Promise.resolve();
        }
      },
    });

    await orchestrator.handle(testMessage("hello"));

    const usageEvent = events.find((e) => e["type"] === "context.usage");
    expect(usageEvent).toMatchObject({
      type: "context.usage",
      workspacePath: "D:/kb",
      inputTokens: 100_000,
      outputTokens: 500,
      cacheReadTokens: 60_000,
      cacheCreationTokens: 20_000,
      contextWindow: 150_000,
      usagePercent: 67,
    });
  });

  it("falls back to heuristic intent when intent runtime is unavailable", async () => {
    const feedbackStore = new MemoryFeedbackStore();
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: {
        warmup() { return Promise.resolve({}); },
        createRuntime() { return Promise.resolve(undefined); },
      },
      feedbackStore,
    });

    // fallbackIntentFor("请修改订单系统") returns { type: "mutate" }
    const response = await orchestrator.handle(testMessage("请修改订单系统", "user-1", "message-1"));

    expect(response.text).toContain("修改请求");
    expect(feedbackStore.entries).toEqual([
      expect.objectContaining({
        intentType: "mutate",
        userMessage: "请修改订单系统",
      })
    ]);
  });

  it("forwards chatType and chatId to AgentRequest", async () => {
    let capturedRequest: AgentRequest | undefined;
    const runtimes = {
      reviewer: {
        name: "runtime",
        run(request: AgentRequest) {
          capturedRequest = request;
          return Promise.resolve({ text: "response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    await orchestrator.handle(testMessage("hello group", "user-1", "1", "group-A", "group"));

    expect(capturedRequest).toBeDefined();
    if (capturedRequest !== undefined) {
      expect(capturedRequest.chatType).toBe("group");
      expect(capturedRequest.chatId).toBe("group-A");
    }
  });

  it("does not set chatType on AgentRequest when InboundMessage has no chatType", async () => {
    let capturedRequest: AgentRequest | undefined;
    const runtimes = {
      reviewer: {
        name: "runtime",
        run(request: AgentRequest) {
          capturedRequest = request;
          return Promise.resolve({ text: "response" });
        },
        disposeSession() {
          return Promise.resolve();
        }
      },
      intent: stubIntentRuntime(),
    };
    const orchestrator = new AgentOrchestrator({
      workspaceResolver,
      authorization: reviewerAuthorization,
      runtimeFactory: stubRuntimeFactory(runtimes),
      initialRuntimes: runtimes,
    });

    await orchestrator.handle(testMessage("hello"));

    expect(capturedRequest).toBeDefined();
    if (capturedRequest !== undefined) {
      expect(capturedRequest.chatType).toBeUndefined();
      expect(capturedRequest.chatId).toBeUndefined();
    }
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
      action === "mutate" || action === "update_kb"
        ? { allowed: false, reason: "reviewer cannot mutate" }
        : { allowed: true }
    );
  },
  hasCapability() {
    return Promise.resolve(false);
  }
};

const developerAuthorization: AuthorizationService = {
  roleFor() {
    return Promise.resolve("developer" as const);
  },
  can() {
    return Promise.resolve({ allowed: true });
  },
  hasCapability() {
    return Promise.resolve(true);
  }
};

function testMessage(text: string, userId = "user-1", id = "1", chatId?: string, chatType?: "single" | "group") {
  return {
    id,
    channel: "test",
    user: { id: userId },
    text,
    receivedAt: new Date(),
    ...(chatId !== undefined ? { chatId } : {}),
    ...(chatType !== undefined ? { chatType } : {}),
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

  public compact(
    key: ConversationSessionKey,
    summary: string,
  ): Promise<void> {
    const keyText = JSON.stringify(key);
    const existing = this.histories.get(keyText);
    if (existing === undefined || existing.length === 0) {
      return Promise.resolve();
    }
    const compactSummary: { readonly role: "assistant"; readonly content: string } = {
      role: "assistant",
      content: `[Previous conversation summary]\n${summary}`,
    };
    const recentMessages = existing.slice(-2);
    this.histories.set(keyText, [compactSummary, ...recentMessages]);
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

  public updateStatus(id: number, status: FeedbackEntry["status"]): Promise<boolean> {
    const entry = this.entries.find((item) => item.id === id);
    if (entry !== undefined) {
      this.entries.splice(this.entries.indexOf(entry), 1, { ...entry, status });
      return Promise.resolve(true);
    }
    return Promise.resolve(false);
  }

  public getAll(): Promise<readonly FeedbackEntry[]> {
    return Promise.resolve(this.entries);
  }
}
