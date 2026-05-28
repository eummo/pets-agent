import { mkdtemp } from "node:fs/promises";
import { existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ChannelUser } from "../core/index.js";
import type {
  AgentRequest,
  AgentResponse,
  AgentRuntime,
  AgentRuntimeFactory
} from "../agent/index.js";
import type { AuthorizationAction, AuthorizationDecision, AuthorizationService } from "../auth/index.js";
import type { ConversationHistoryStore, ConversationSessionKey, ConversationSessionStore, FeedbackEntry, FeedbackStore } from "../persistence/index.js";
import { AgentOrchestrator } from "../core/orchestrator.js";
import { ConfiguredWorkspaceResolver } from "../workspace/configuredWorkspaceResolver.js";
import { createServer } from "../server/createServer.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";

const stubRuntime: AgentRuntime = {
  name: "stub",
  run(request: AgentRequest): Promise<AgentResponse> {
    return Promise.resolve({ text: `stub: ${request.text}` });
  },
  disposeSession(): Promise<void> {
    return Promise.resolve();
  },
};

/**
 * Regression: after the contracts.ts → index.ts migration, no contracts.ts files
 * or imports referencing contracts.js should remain in the package directories.
 */
function assertNoContractsTsRemnants(): void {
  const srcRoot = path.resolve(import.meta.dirname, "..");
  const packageDirs = ["agent", "auth", "core", "intent", "persistence", "workspace"];

  // 1. No contracts.ts files in package directories
  const forbiddenFiles: string[] = [];
  for (const dir of packageDirs) {
    const candidate = path.join(srcRoot, dir, "contracts.ts");
    if (existsSync(candidate)) {
      forbiddenFiles.push(candidate);
    }
  }

  if (forbiddenFiles.length > 0) {
    throw new Error(
      `Found contracts.ts files that should have been migrated to index.ts:\n` +
      forbiddenFiles.map((f) => `  ${f}`).join("\n")
    );
  }

  // 2. No imports referencing contracts.js in src/
  try {
    const result = execSync(
      `grep -r "from.*contracts\\.js" --include="*.ts" -l "${srcRoot}"`,
      { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] }
    ).trim();
    if (result.length > 0) {
      throw new Error(
        `Found imports referencing contracts.js (should use index.js):\n${result}`
      );
    }
  } catch (error: unknown) {
    // grep returns exit code 1 when no matches — that's the passing case
    if (error instanceof Error && "status" in error && (error as { status: number }).status === 1) {
      // No matches found — pass
    } else {
      throw error;
    }
  }

  console.info("[pass] deterministic-no-contracts-ts-remnants");
}

async function main(): Promise<void> {
  const root = await mkdtemp(path.join(tmpdir(), "pets-agent-deterministic-smoke-"));
  const feedbackStore = new MemoryFeedbackStore();
  const authorization = new ReviewerAuthorization();
  const runtimes: Record<string, AgentRuntime> = {
    reviewer: stubRuntime,
    intent: {
      name: "intent",
      run(request: AgentRequest): Promise<AgentResponse> {
        return Promise.resolve({ text: fallbackIntentFor(request.text).type });
      },
      disposeSession(): Promise<void> {
        return Promise.resolve();
      },
    },
  };
  const runtimeFactory: AgentRuntimeFactory = {
    warmup() { return Promise.resolve(runtimes); },
    createRuntime(role: string) { return Promise.resolve(runtimes[role]); },
  };
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({ knowledgeBasePath: path.join(root, "knowledge-base") }),
    authorization,
    runtimeFactory,
    initialRuntimes: runtimes,
    feedbackStore,
  });
  const server = createServer({
    messageHandler: orchestrator,
    feedbackStore,
    authorization,
    enableDevRoutes: true,
  });

  await assertHealth(server);
  await assertReviewerMutationDeniedWithoutIntentLlm(server, feedbackStore);
  await assertPathTraversalRejected(server);
  await assertRoleSwitchCarriesHistory();
  assertNoContractsTsRemnants();

  await server.close();
}

async function assertHealth(server: ReturnType<typeof createServer>): Promise<void> {
  const response = await server.inject({ method: "GET", url: "/health" });
  if (response.statusCode !== 200) {
    throw new Error(`health failed: ${response.statusCode}`);
  }
  console.info("[pass] deterministic-health");
}

async function assertReviewerMutationDeniedWithoutIntentLlm(
  server: ReturnType<typeof createServer>,
  feedbackStore: MemoryFeedbackStore,
): Promise<void> {
  const response = await server.inject({
    method: "POST",
    url: "/dev/chat",
    payload: {
      userId: "reviewer-1",
      text: "请修改订单系统",
    },
  });
  if (response.statusCode !== 200) {
    throw new Error(`chat failed: ${response.statusCode} ${response.body}`);
  }
  if (!response.body.includes("修改请求")) {
    throw new Error(`expected mutation denial, got: ${response.body}`);
  }
  if (feedbackStore.entries[0]?.intentType !== "mutate") {
    throw new Error("expected denied mutation to be recorded as feedback");
  }
  console.info("[pass] deterministic-reviewer-mutation-denied");
}

async function assertPathTraversalRejected(server: ReturnType<typeof createServer>): Promise<void> {
  const response = await server.inject({ method: "GET", url: "/dev/chat/..%2F..%2Fpackage.json" });
  if (response.statusCode !== 403) {
    throw new Error(`expected path traversal rejection, got: ${response.statusCode}`);
  }
  console.info("[pass] deterministic-path-traversal-rejected");
}

class ReviewerAuthorization implements AuthorizationService {
  public roleFor(): Promise<string> {
    return Promise.resolve("reviewer");
  }

  public can(_user: ChannelUser, action: AuthorizationAction): Promise<AuthorizationDecision> {
    return Promise.resolve(action === "mutate" ? { allowed: false } : { allowed: true });
  }

  public hasCapability(): Promise<boolean> {
    return Promise.resolve(false);
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
    if (entry === undefined) {
      return Promise.resolve(false);
    }
    this.entries.splice(this.entries.indexOf(entry), 1, { ...entry, status });
    return Promise.resolve(true);
  }

  public getAll(): Promise<readonly FeedbackEntry[]> {
    return Promise.resolve(this.entries);
  }
}

async function assertRoleSwitchCarriesHistory(): Promise<void> {
  // Use two runtimes that record whether they received history on creation.
  const disposedSessions: string[] = [];
  const receivedHistory: (readonly import("../core/index.js").AgentConversationMessage[])[] = [];
  const reviewerRuntime: AgentRuntime = {
    name: "stub-reviewer",
    run(request: AgentRequest): Promise<AgentResponse> {
      return Promise.resolve({ text: `reviewer: ${request.text}`, sessionId: "reviewer-s-1" });
    },
    disposeSession(sessionId: string): Promise<void> {
      disposedSessions.push(sessionId);
      return Promise.resolve();
    },
  };
  const adminRuntime: AgentRuntime = {
    name: "stub-admin",
    run(request: AgentRequest): Promise<AgentResponse> {
      if (request.history !== undefined) {
        receivedHistory.push(request.history);
      }
      return Promise.resolve({ text: `admin: ${request.text}`, sessionId: "admin-s-1" });
    },
    disposeSession(): Promise<void> {
      return Promise.resolve();
    },
  };
  const switchingAuth = new SwitchableAuthorization("reviewer");
  const switchRuntimes: Record<string, AgentRuntime> = {
    reviewer: reviewerRuntime,
    admin: adminRuntime,
    intent: {
      name: "intent",
      run(): Promise<AgentResponse> {
        return Promise.resolve({ text: "query" });
      },
      disposeSession(): Promise<void> {
        return Promise.resolve();
      },
    },
  };
  const switchOrchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({ knowledgeBasePath: path.join(await mkdtemp(path.join(tmpdir(), "pets-agent-role-switch-")), "kb") }),
    authorization: switchingAuth,
    runtimeFactory: {
      warmup() { return Promise.resolve(switchRuntimes); },
      createRuntime(role: string) { return Promise.resolve(switchRuntimes[role]); },
    },
    initialRuntimes: switchRuntimes,
    sessionStore: new InMemorySessionStore(),
    historyStore: new InMemoryHistoryStore(),
  });
  const switchServer = createServer({
    messageHandler: switchOrchestrator,
    feedbackStore: new MemoryFeedbackStore(),
    authorization: switchingAuth,
    enableDevRoutes: true,
  });

  // First message as reviewer
  const first = await switchServer.inject({
    method: "POST",
    url: "/dev/chat",
    payload: { userId: "switch-user", text: "hello" },
  });
  if (first.statusCode !== 200) {
    throw new Error(`Role switch first chat failed: ${first.statusCode}`);
  }

  // Switch to admin
  switchingAuth.setRole("admin");
  const second = await switchServer.inject({
    method: "POST",
    url: "/dev/chat",
    payload: { userId: "switch-user", text: "admin task" },
  });
  if (second.statusCode !== 200) {
    throw new Error(`Role switch second chat failed: ${second.statusCode}`);
  }

  // Verify old session was disposed
  if (!disposedSessions.includes("reviewer-s-1")) {
    throw new Error(`Role switch: expected reviewer-s-1 to be disposed. Got: ${disposedSessions.join(", ")}`);
  }

  // Verify the admin runtime received the prior conversation history
  if (receivedHistory.length === 0) {
    throw new Error("Role switch: expected admin runtime to receive prior history.");
  }
  const history = receivedHistory[0];
  if (history === undefined) {
    throw new Error("Role switch: first history entry is undefined.");
  }
  if (history.length < 2) {
    throw new Error(`Role switch: expected at least 2 history messages, got ${history.length}.`);
  }
  const firstMsg = history[0];
  const secondMsg = history[1];
  if (firstMsg === undefined) {
    throw new Error("Role switch: first history message is undefined.");
  }
  if (secondMsg === undefined) {
    throw new Error("Role switch: second history message is undefined.");
  }
  if (firstMsg.role !== "user" || firstMsg.content !== "hello") {
    throw new Error(`Role switch: expected first history message to be user:hello, got ${JSON.stringify(firstMsg)}`);
  }
  if (secondMsg.role !== "assistant" || !secondMsg.content.includes("reviewer")) {
    throw new Error(`Role switch: expected second history message to be assistant with reviewer response, got ${JSON.stringify(secondMsg)}`);
  }

  await switchServer.close();
  console.info("[pass] deterministic-role-switch-carries-history");
}

class SwitchableAuthorization implements AuthorizationService {
  private currentRole: string;

  public constructor(initialRole: string) {
    this.currentRole = initialRole;
  }

  public setRole(role: string): void {
    this.currentRole = role;
  }

  public roleFor(): Promise<string> {
    return Promise.resolve(this.currentRole);
  }

  public can(_user: ChannelUser, action: AuthorizationAction): Promise<AuthorizationDecision> {
    return Promise.resolve(action === "mutate" ? { allowed: this.currentRole !== "reviewer" } : { allowed: true });
  }

  public hasCapability(): Promise<boolean> {
    return Promise.resolve(this.currentRole !== "reviewer");
  }
}

class InMemorySessionStore implements ConversationSessionStore {
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

class InMemoryHistoryStore implements ConversationHistoryStore {
  private readonly histories = new Map<string, { readonly role: "user" | "assistant"; readonly content: string }[]>();

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

  public compact(): Promise<void> {
    return Promise.resolve();
  }

  public delete(key: ConversationSessionKey): Promise<void> {
    this.histories.delete(JSON.stringify(key));
    return Promise.resolve();
  }

  public archive(key: ConversationSessionKey): Promise<void> {
    this.histories.delete(JSON.stringify(key));
    return Promise.resolve();
  }
}

await main();
