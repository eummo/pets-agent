import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type {
  AgentRequest,
  AgentResponse,
  AgentRuntime,
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService,
  ChannelUser,
  FeedbackEntry,
  FeedbackStore,
  UserIntent,
} from "../core/contracts.js";
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

async function main(): Promise<void> {
  const root = await mkdtemp(path.join(tmpdir(), "pets-agent-deterministic-smoke-"));
  const feedbackStore = new MemoryFeedbackStore();
  const authorization = new ReviewerAuthorization();
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({ knowledgeBasePath: path.join(root, "knowledge-base") }),
    authorization,
    feedbackStore,
    intentDetection: {
      detectIntent(userMessage: string): Promise<UserIntent> {
        return Promise.resolve(fallbackIntentFor(userMessage));
      }
    },
    agentRuntimes: {
      reviewer: stubRuntime,
    },
  });
  const server = createServer({
    messageHandler: orchestrator,
    feedbackStore,
    authorization,
  });

  await assertHealth(server);
  await assertReviewerMutationDeniedWithoutIntentLlm(server, feedbackStore);
  await assertPathTraversalRejected(server);

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

await main();
