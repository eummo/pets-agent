import "dotenv/config";
import { complete } from "@earendil-works/pi-ai";
import type { Api, Model } from "@earendil-works/pi-ai";
import { loadRuntimeConfig } from "../config/runtimeConfig.js";
import { buildPiModel } from "../intent/piModel.js";
import { InMemoryLoopStore } from "../loop/loopStore.js";
import { LoopService } from "../loop/loopService.js";
import type {
  ActionExecutor,
  ActionResult,
  LoopEventLogger,
  LoopExecutionContext
} from "../loop/loopTypes.js";

// ── LLM Action Executor ───────────────────────────────────────────────────────

type LlmBlock = { readonly type: "text"; readonly text: string };

const LOOP_SYSTEM_PROMPT = `You are an action executor for a loop-based task system.
You receive an action description and must produce a clear, factual result.

Rules:
- Respond with factual observations only.
- If the task is complete, start your response with "DONE:".
- If you need to pause for approval, start your response with "PAUSE:".
- Keep responses concise and focused on the requested action.`;

class LlmActionExecutor implements ActionExecutor {
  public readonly callCount = { plan: 0, act: 0 };

  public constructor(
    private readonly model: Model<Api>,
    private readonly apiKey: string
  ) {}

  public async execute(
    context: LoopExecutionContext,
    action: string,
    signal: AbortSignal
  ): Promise<ActionResult> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    // Forward external signal
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    try {
      const response = await complete(
        this.model,
        {
          systemPrompt: LOOP_SYSTEM_PROMPT,
          messages: [
            {
              role: "user",
              content: `Loop run ${context.loopRunId}, step ${context.stepId}, attempt ${context.attempt}.\n\nAction: ${action}`,
              timestamp: Date.now()
            }
          ]
        },
        {
          apiKey: this.apiKey,
          signal: controller.signal
        }
      );

      const text = response.content
        .filter((block): block is LlmBlock => block.type === "text")
        .map((block) => block.text)
        .join("");

      // Rough token estimate: ~4 chars per token
      const estimatedTokens = Math.ceil(text.length / 4);

      return {
        output: text.trim(),
        tokenUsage: estimatedTokens,
        evidence: `LLM response (${response.stopReason}): ${text.slice(0, 200)}`
      };
    } finally {
      clearTimeout(timeout);
    }
  }
}

// ── Event Logger ──────────────────────────────────────────────────────────────

class RecordingLoopEventLogger implements LoopEventLogger {
  public readonly events: Record<string, unknown>[] = [];

  public write(event: Record<string, unknown>): Promise<void> {
    this.events.push(event);
    return Promise.resolve();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function createDefinition(
  store: InMemoryLoopStore,
  goal: string,
  maxIterations: number
): Promise<string> {
  const created = await store.createDefinition({
    name: `loop-smoke-${Date.now()}`,
    goal,
    workspacePath: "/workspace/smoke",
    role: "reviewer",
    maxIterations,
    timeoutMs: 120_000,
    maxTokenBudget: 50_000,
    triggerType: "manual",
    verificationStrategy: "llm-check"
  });
  return created.id;
}

function waitFor(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Smoke Cases ───────────────────────────────────────────────────────────────

async function assertLoopCompletesWithRealLlm(
  store: InMemoryLoopStore,
  executor: LlmActionExecutor
): Promise<void> {
  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: new RecordingLoopEventLogger(),
    ownerId: "loop-smoke"
  });

  const definitionId = await createDefinition(
    store,
    "用一句话回答：1+1等于几？回答后以 DONE: 开头结束。",
    1
  );

  const run = await service.startRun(definitionId, "smoke-user");
  // Wait for async execution to complete
  await waitFor(15_000);

  const finalRun = await store.getRun(run.id);
  if (finalRun === undefined) {
    throw new Error("Loop run not found");
  }
  if (finalRun.status !== "completed") {
    throw new Error(`Expected completed, got ${finalRun.status}`);
  }
  if (finalRun.completedAt === null) {
    throw new Error("Expected completedAt to be set");
  }

  const steps = await store.getStepsByRun(run.id);
  if (steps.length === 0) {
    throw new Error("Expected at least 1 step");
  }
  const lastStep = steps[steps.length - 1];
  if (lastStep === undefined) {
    throw new Error("Last step is undefined");
  }
  if (lastStep.phase !== "decide") {
    throw new Error(`Expected last step phase 'decide', got '${lastStep.phase}'`);
  }
  if (lastStep.status !== "succeeded") {
    throw new Error(`Expected last step status 'succeeded', got '${lastStep.status}'`);
  }

  // Verify observation contains meaningful content
  if (lastStep.observation === null || lastStep.observation.length === 0) {
    throw new Error("Expected non-empty observation from real LLM");
  }

  console.info("[pass] loop-smoke-completes-with-real-llm");
}

async function assertLoopMultiIterationWithRealLlm(
  store: InMemoryLoopStore,
  executor: LlmActionExecutor
): Promise<void> {
  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: new RecordingLoopEventLogger(),
    ownerId: "loop-smoke"
  });

  // A task that should complete in 1-2 iterations
  const definitionId = await createDefinition(
    store,
    "回答以下问题并用 DONE: 开头结束：今天星期几？（根据常识给出一个合理的回答）",
    3
  );

  const run = await service.startRun(definitionId, "smoke-user");
  await waitFor(30_000);

  const finalRun = await store.getRun(run.id);
  if (finalRun === undefined) {
    throw new Error("Loop run not found");
  }
  if (finalRun.status !== "completed") {
    throw new Error(`Expected completed, got ${finalRun.status}`);
  }

  const steps = await store.getStepsByRun(run.id);
  if (steps.length === 0) {
    throw new Error("Expected at least 1 step");
  }

  // All steps should have progressed through phases
  for (const step of steps) {
    if (step.phase !== "decide") {
      throw new Error(`Step ${step.id} expected phase 'decide', got '${step.phase}'`);
    }
  }

  console.info("[pass] loop-smoke-multi-iteration-with-real-llm");
}

async function assertLoopEventsLoggedCorrectly(
  store: InMemoryLoopStore,
  executor: LlmActionExecutor
): Promise<void> {
  const logger = new RecordingLoopEventLogger();
  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: logger,
    ownerId: "loop-smoke"
  });

  const definitionId = await createDefinition(
    store,
    "用一句话回答：水的化学式是什么？回答后以 DONE: 开头结束。",
    1
  );

  const run = await service.startRun(definitionId, "smoke-user");
  await waitFor(15_000);

  // Verify started event
  const startedEvent = logger.events.find((e) => e["type"] === "loop.started");
  if (startedEvent === undefined) {
    throw new Error("Missing loop.started event");
  }
  if (startedEvent["requestedBy"] !== "smoke-user") {
    throw new Error(`Expected requestedBy 'smoke-user', got ${String(startedEvent["requestedBy"])}`);
  }

  // Verify completed event
  const completedEvent = logger.events.find((e) => e["type"] === "loop.completed");
  if (completedEvent === undefined) {
    throw new Error("Missing loop.completed event");
  }

  // Verify step events with execution context
  const stepEvents = logger.events.filter((e) => {
    const t = e["type"];
    return typeof t === "string" && t.startsWith("loop.step.");
  });
  if (stepEvents.length === 0) {
    throw new Error("Missing loop.step.* events");
  }

  // Verify step events carry execution context
  for (const event of stepEvents) {
    if (event["loopRunId"] !== run.id) {
      throw new Error(
        `Step event loopRunId mismatch: ${String(event["loopRunId"])} !== ${run.id}`
      );
    }
    if (typeof event["stepId"] !== "string" || event["stepId"].length === 0) {
      throw new Error("Step event missing stepId");
    }
    if (typeof event["attempt"] !== "number") {
      throw new Error("Step event missing attempt");
    }
  }

  // Verify phase progression events
  const phases = ["planned", "acted", "observed", "verified", "decided"];
  for (const phase of phases) {
    const found = stepEvents.find((e) => e["type"] === `loop.step.${phase}`);
    if (found === undefined) {
      throw new Error(`Missing loop.step.${phase} event`);
    }
  }

  console.info("[pass] loop-smoke-events-logged-correctly");
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.info("Loading runtime config...");
  const config = await loadRuntimeConfig();

  console.info(`Using model: ${config.llm.modelId}`);
  const model = buildPiModel(config.llm);
  const executor = new LlmActionExecutor(model, config.llm.apiKey);

  let passed = 0;
  let failed = 0;

  const cases: { name: string; fn: (store: InMemoryLoopStore, executor: LlmActionExecutor) => Promise<void> }[] = [
    { name: "completes-with-real-llm", fn: assertLoopCompletesWithRealLlm },
    { name: "multi-iteration-with-real-llm", fn: assertLoopMultiIterationWithRealLlm },
    { name: "events-logged-correctly", fn: assertLoopEventsLoggedCorrectly }
  ];

  for (const testCase of cases) {
    const store = new InMemoryLoopStore();
    try {
      await testCase.fn(store, executor);
      passed++;
    } catch (error: unknown) {
      failed++;
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[fail] loop-smoke-${testCase.name}: ${message}`);
    }
  }

  console.info(`\nLoop smoke: ${passed} passed, ${failed} failed`);

  if (failed > 0) {
    process.exit(1);
  }
}

await main();
