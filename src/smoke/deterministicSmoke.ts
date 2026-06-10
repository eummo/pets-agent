import { mkdtemp, readFile } from "node:fs/promises";
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
import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService
} from "../auth/index.js";
import type {
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore,
  FeedbackEntry,
  FeedbackStore
} from "../persistence/index.js";
import { AgentOrchestrator } from "../core/orchestrator.js";
import { ConfiguredWorkspaceResolver } from "../workspace/configuredWorkspaceResolver.js";
import { createServer } from "../server/createServer.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";
import { InMemoryLoopStore } from "../loop/loopStore.js";
import { LoopService } from "../loop/loopService.js";
import type {
  ActionExecutor,
  ActionResult,
  LoopEventLogger,
  LoopExecutionContext
} from "../loop/loopTypes.js";

const stubRuntime: AgentRuntime = {
  name: "stub",
  run(request: AgentRequest): Promise<AgentResponse> {
    return Promise.resolve({ text: `stub: ${request.text}` });
  },
  disposeSession(): Promise<void> {
    return Promise.resolve();
  }
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
    const result = execSync(`grep -r "from.*contracts\\.js" --include="*.ts" -l "${srcRoot}"`, {
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"]
    }).trim();
    if (result.length > 0) {
      throw new Error(`Found imports referencing contracts.js (should use index.js):\n${result}`);
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
      }
    }
  };
  const runtimeFactory: AgentRuntimeFactory = {
    warmup() {
      return Promise.resolve(runtimes);
    },
    createRuntime(role: string) {
      return Promise.resolve(runtimes[role]);
    }
  };
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: path.join(root, "knowledge-base")
    }),
    authorization,
    runtimeFactory,
    initialRuntimes: runtimes,
    feedbackStore
  });
  const server = createServer({
    messageHandler: orchestrator,
    feedbackStore,
    authorization,
    enableDevRoutes: true
  });

  await assertHealth(server);
  await assertReviewerMutationDeniedWithoutIntentLlm(server, feedbackStore);
  await assertPathTraversalRejected(server);
  await assertUploadedDocumentReachesRuntime();
  await assertRoleSwitchCarriesHistory();
  assertNoContractsTsRemnants();

  await assertLoopRunCompletesSuccessfully();
  await assertLoopRunStopsOnMaxIterations();
  await assertLoopCancelStopsExecution();
  await assertLoopPauseAndResume();
  await assertLoopRecoveryTransitionsInterruptedSteps();
  await assertLoopEventsCarryExecutionContext();

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
  feedbackStore: MemoryFeedbackStore
): Promise<void> {
  const response = await server.inject({
    method: "POST",
    url: "/dev/chat",
    payload: {
      userId: "reviewer-1",
      text: "请修改订单系统"
    }
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

async function assertUploadedDocumentReachesRuntime(): Promise<void> {
  const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-deterministic-upload-"));
  const uploadRuntime: AgentRuntime = {
    name: "upload-runtime",
    async run(request: AgentRequest): Promise<AgentResponse> {
      const attachment = request.attachments?.[0];
      if (attachment === undefined) {
        return { text: "missing attachment" };
      }
      const content = await readFile(attachment.storagePath, "utf8");
      return {
        text: `upload:${attachment.type}:${attachment.name}:${attachment.mimeType}:${content}`
      };
    },
    disposeSession(): Promise<void> {
      return Promise.resolve();
    }
  };
  const uploadRuntimes: Record<string, AgentRuntime> = {
    reviewer: uploadRuntime,
    intent: {
      name: "intent",
      run(): Promise<AgentResponse> {
        return Promise.resolve({ text: "query" });
      },
      disposeSession(): Promise<void> {
        return Promise.resolve();
      }
    }
  };
  const authorization = new ReviewerAuthorization();
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: path.join(
        await mkdtemp(path.join(tmpdir(), "pets-agent-upload-kb-")),
        "kb"
      )
    }),
    authorization,
    runtimeFactory: {
      warmup() {
        return Promise.resolve(uploadRuntimes);
      },
      createRuntime(role: string) {
        return Promise.resolve(uploadRuntimes[role]);
      }
    },
    initialRuntimes: uploadRuntimes
  });
  const server = createServer({
    messageHandler: orchestrator,
    feedbackStore: new MemoryFeedbackStore(),
    authorization,
    enableDevRoutes: true,
    uploadRootPath
  });
  const content = "deterministic upload fact";

  const response = await server.inject({
    method: "POST",
    url: "/dev/chat",
    payload: {
      userId: "upload-user",
      text: "answer from the upload",
      attachments: [
        {
          name: "facts.md",
          mimeType: "text/markdown",
          contentBase64: Buffer.from(content, "utf8").toString("base64"),
          sizeBytes: Buffer.byteLength(content)
        }
      ]
    }
  });

  if (response.statusCode !== 200) {
    throw new Error(`upload chat failed: ${response.statusCode} ${response.body}`);
  }
  if (!response.body.includes(`upload:document:facts.md:text/markdown:${content}`)) {
    throw new Error(`expected upload document to reach runtime, got: ${response.body}`);
  }

  await server.close();
  console.info("[pass] deterministic-uploaded-document-reaches-runtime");
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
    }
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
    }
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
      }
    }
  };
  const switchOrchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: path.join(
        await mkdtemp(path.join(tmpdir(), "pets-agent-role-switch-")),
        "kb"
      )
    }),
    authorization: switchingAuth,
    runtimeFactory: {
      warmup() {
        return Promise.resolve(switchRuntimes);
      },
      createRuntime(role: string) {
        return Promise.resolve(switchRuntimes[role]);
      }
    },
    initialRuntimes: switchRuntimes,
    sessionStore: new InMemorySessionStore(),
    historyStore: new InMemoryHistoryStore()
  });
  const switchServer = createServer({
    messageHandler: switchOrchestrator,
    feedbackStore: new MemoryFeedbackStore(),
    authorization: switchingAuth,
    enableDevRoutes: true
  });

  // First message as reviewer
  const first = await switchServer.inject({
    method: "POST",
    url: "/dev/chat",
    payload: { userId: "switch-user", text: "hello" }
  });
  if (first.statusCode !== 200) {
    throw new Error(`Role switch first chat failed: ${first.statusCode}`);
  }

  // Switch to admin
  switchingAuth.setRole("admin");
  const second = await switchServer.inject({
    method: "POST",
    url: "/dev/chat",
    payload: { userId: "switch-user", text: "admin task" }
  });
  if (second.statusCode !== 200) {
    throw new Error(`Role switch second chat failed: ${second.statusCode}`);
  }

  // Verify old session was disposed
  if (!disposedSessions.includes("reviewer-s-1")) {
    throw new Error(
      `Role switch: expected reviewer-s-1 to be disposed. Got: ${disposedSessions.join(", ")}`
    );
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
    throw new Error(
      `Role switch: expected first history message to be user:hello, got ${JSON.stringify(firstMsg)}`
    );
  }
  if (secondMsg.role !== "assistant" || !secondMsg.content.includes("reviewer")) {
    throw new Error(
      `Role switch: expected second history message to be assistant with reviewer response, got ${JSON.stringify(secondMsg)}`
    );
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
    return Promise.resolve(
      action === "mutate" ? { allowed: this.currentRole !== "reviewer" } : { allowed: true }
    );
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
  private readonly histories = new Map<
    string,
    { readonly role: "user" | "assistant"; readonly content: string }[]
  >();

  public get(
    key: ConversationSessionKey
  ): Promise<readonly { readonly role: "user" | "assistant"; readonly content: string }[]> {
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

// ── Loop Smoke Helpers ────────────────────────────────────────────────────────

class StubActionExecutor implements ActionExecutor {
  private readonly responses: ActionResult[];
  private nextResponse = 0;

  public constructor(responses: ActionResult[]) {
    this.responses = responses;
  }

  public execute(
    context: LoopExecutionContext,
    action: string,
    signal: AbortSignal
  ): Promise<ActionResult> {
    // Parameters required by ActionExecutor contract but not used by stub
    void context;
    void action;
    void signal;
    const response = this.responses[this.nextResponse];
    if (response === undefined) {
      return Promise.reject(new Error("No more stub responses"));
    }
    this.nextResponse++;
    return Promise.resolve(response);
  }
}

class RecordingLoopEventLogger implements LoopEventLogger {
  public readonly events: Record<string, unknown>[] = [];

  public write(event: Record<string, unknown>): Promise<void> {
    this.events.push(event);
    return Promise.resolve();
  }
}

class SlowStubActionExecutor implements ActionExecutor {
  public constructor(
    private readonly delayMs: number,
    private readonly result: ActionResult
  ) {}

  public execute(
    _context: LoopExecutionContext,
    _action: string,
    signal: AbortSignal
  ): Promise<ActionResult> {
    return new Promise<ActionResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        resolve(this.result);
      }, this.delayMs);

      signal.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true }
      );
    });
  }
}

async function createLoopDefinition(
  store: InMemoryLoopStore,
  overrides?: { maxIterations?: number }
): Promise<string> {
  const created = await store.createDefinition({
    name: "smoke-test",
    goal: "Smoke test goal",
    workspacePath: "/workspace/smoke",
    role: "reviewer",
    maxIterations: overrides?.maxIterations ?? 3,
    timeoutMs: 60_000,
    maxTokenBudget: 100_000,
    triggerType: "manual",
    verificationStrategy: "smoke-check"
  });
  return created.id;
}

// ── Loop Smoke Cases ──────────────────────────────────────────────────────────

async function assertLoopRunCompletesSuccessfully(): Promise<void> {
  const store = new InMemoryLoopStore();
  const executor = new StubActionExecutor([
    { output: "Plan: check health", tokenUsage: 50, evidence: "plan-ok" },
    { output: "DONE: healthy", tokenUsage: 100, evidence: "all-pass" }
  ]);
  const service = new LoopService({
    store,
    actionExecutor: executor,
    ownerId: "smoke-service"
  });
  const definitionId = await createLoopDefinition(store, { maxIterations: 1 });
  const run = await service.startRun(definitionId, "smoke-user");
  await new Promise((resolve) => setTimeout(resolve, 50));

  const finalRun = await store.getRun(run.id);
  if (finalRun?.status !== "completed") {
    throw new Error(`Expected completed, got ${finalRun?.status ?? "undefined"}`);
  }
  if (finalRun.completedAt === null) {
    throw new Error("Expected completedAt to be set");
  }
  const steps = await store.getStepsByRun(run.id);
  if (steps.length !== 1) {
    throw new Error(`Expected 1 step, got ${steps.length}`);
  }
  const step = steps[0];
  if (step === undefined) {
    throw new Error("Step is undefined");
  }
  if (step.phase !== "decide") {
    throw new Error(`Expected phase decide, got ${step.phase}`);
  }
  if (step.status !== "succeeded") {
    throw new Error(`Expected step succeeded, got ${step.status}`);
  }
  console.info("[pass] deterministic-loop-run-completes-successfully");
}

async function assertLoopRunStopsOnMaxIterations(): Promise<void> {
  const store = new InMemoryLoopStore();
  const executor = new StubActionExecutor([
    { output: "Plan: check", tokenUsage: 50, evidence: "plan" },
    { output: "continue checking", tokenUsage: 50, evidence: "more" },
    { output: "Plan: check", tokenUsage: 50, evidence: "plan" },
    { output: "continue checking", tokenUsage: 50, evidence: "more" }
  ]);
  const service = new LoopService({
    store,
    actionExecutor: executor,
    ownerId: "smoke-service"
  });
  const definitionId = await createLoopDefinition(store, { maxIterations: 2 });
  const run = await service.startRun(definitionId, "smoke-user");
  await new Promise((resolve) => setTimeout(resolve, 100));

  const finalRun = await store.getRun(run.id);
  if (finalRun?.status !== "completed") {
    throw new Error(`Expected completed (max iterations), got ${finalRun?.status ?? "undefined"}`);
  }
  const steps = await store.getStepsByRun(run.id);
  if (steps.length > 2) {
    throw new Error(`Expected at most 2 steps (maxIterations=2), got ${steps.length}`);
  }
  console.info("[pass] deterministic-loop-run-stops-on-max-iterations");
}

async function assertLoopCancelStopsExecution(): Promise<void> {
  const store = new InMemoryLoopStore();
  const executor = new SlowStubActionExecutor(5_000, {
    output: "slow",
    tokenUsage: 100,
    evidence: "slow"
  });
  const service = new LoopService({
    store,
    actionExecutor: executor,
    ownerId: "smoke-service"
  });
  const definitionId = await createLoopDefinition(store, { maxIterations: 10 });
  const run = await service.startRun(definitionId, "smoke-user");

  // Cancel immediately while executor is running
  await new Promise((resolve) => setTimeout(resolve, 10));
  await service.cancelRun(run.id);

  const finalRun = await store.getRun(run.id);
  if (finalRun?.status !== "cancelled") {
    throw new Error(`Expected cancelled, got ${finalRun?.status ?? "undefined"}`);
  }

  const steps = await store.getStepsByRun(run.id);
  for (const step of steps) {
    if (step.status === "running") {
      throw new Error(`Step ${step.id} still in running state after cancel`);
    }
  }
  console.info("[pass] deterministic-loop-cancel-stops-execution");
}

async function assertLoopPauseAndResume(): Promise<void> {
  const store = new InMemoryLoopStore();
  const pauseExecutor = new StubActionExecutor([
    { output: "Plan: check", tokenUsage: 50, evidence: "plan" },
    { output: "PAUSE: need approval", tokenUsage: 100, evidence: "pause" }
  ]);
  const service = new LoopService({
    store,
    actionExecutor: pauseExecutor,
    ownerId: "smoke-service"
  });
  const definitionId = await createLoopDefinition(store, { maxIterations: 5 });
  const run = await service.startRun(definitionId, "smoke-user");
  await new Promise((resolve) => setTimeout(resolve, 50));

  const pausedRun = await store.getRun(run.id);
  if (pausedRun?.status !== "paused") {
    throw new Error(`Expected paused, got ${pausedRun?.status ?? "undefined"}`);
  }

  // Resume with a new executor that completes
  const resumeExecutor = new StubActionExecutor([
    { output: "Plan: resumed", tokenUsage: 50, evidence: "resumed" },
    { output: "DONE: approved", tokenUsage: 100, evidence: "approved" }
  ]);
  const resumeService = new LoopService({
    store,
    actionExecutor: resumeExecutor,
    ownerId: "smoke-service"
  });
  await resumeService.resumeRun(run.id);
  await new Promise((resolve) => setTimeout(resolve, 50));

  const finalRun = await store.getRun(run.id);
  if (finalRun?.status !== "completed") {
    throw new Error(`Expected completed after resume, got ${finalRun?.status ?? "undefined"}`);
  }
  console.info("[pass] deterministic-loop-pause-and-resume");
}

async function assertLoopRecoveryTransitionsInterruptedSteps(): Promise<void> {
  const store = new InMemoryLoopStore();
  const executor = new StubActionExecutor([]);
  const service = new LoopService({
    store,
    actionExecutor: executor,
    ownerId: "smoke-service"
  });

  // Manually insert an expired running step
  await store.createStep({
    runId: "run-1",
    iteration: 1,
    attempt: 1,
    status: "running",
    phase: "act",
    idempotencyKey: "run-1:iter:1:attempt:1",
    claimOwner: "old-service",
    leaseExpiry: "2020-01-01T00:00:00",
    actionDescription: "some action",
    observation: null,
    decision: null,
    completedAt: null
  });

  const count = await service.recoverInterruptedSteps();
  if (count !== 1) {
    throw new Error(`Expected 1 recovered step, got ${count}`);
  }

  const steps = await store.getStepsByRun("run-1");
  const step = steps[0];
  if (step === undefined) {
    throw new Error("Step not found");
  }
  if (step.status !== "interrupted") {
    throw new Error(`Expected interrupted, got ${step.status}`);
  }
  if (step.claimOwner !== null) {
    throw new Error(`Expected claimOwner cleared, got ${step.claimOwner}`);
  }
  console.info("[pass] deterministic-loop-recovery-transitions-interrupted-steps");
}

async function assertLoopEventsCarryExecutionContext(): Promise<void> {
  const store = new InMemoryLoopStore();
  const logger = new RecordingLoopEventLogger();
  const executor = new StubActionExecutor([
    { output: "Plan: check", tokenUsage: 50, evidence: "plan" },
    { output: "DONE: complete", tokenUsage: 100, evidence: "done" }
  ]);
  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: logger,
    ownerId: "smoke-service"
  });
  const definitionId = await createLoopDefinition(store, { maxIterations: 1 });
  const run = await service.startRun(definitionId, "smoke-user");
  await new Promise((resolve) => setTimeout(resolve, 50));

  const started = logger.events.find((e) => e["type"] === "loop.started");
  if (started === undefined) {
    throw new Error("Missing loop.started event");
  }

  const completed = logger.events.find((e) => e["type"] === "loop.completed");
  if (completed === undefined) {
    throw new Error("Missing loop.completed event");
  }

  const stepEvents = logger.events.filter((e) => {
    const t = e["type"];
    return typeof t === "string" && t.startsWith("loop.step.");
  });
  if (stepEvents.length === 0) {
    throw new Error("Missing loop.step.* events");
  }

  for (const event of stepEvents) {
    if (event["loopRunId"] !== run.id) {
      throw new Error(`Step event loopRunId mismatch: ${String(event["loopRunId"])} !== ${run.id}`);
    }
    if (typeof event["stepId"] !== "string" || event["stepId"].length === 0) {
      throw new Error("Step event missing stepId");
    }
    if (typeof event["attempt"] !== "number") {
      throw new Error("Step event missing attempt");
    }
  }
  console.info("[pass] deterministic-loop-events-carry-execution-context");
}

await main();
