/**
 * DeepAgents Runtime — wraps deepagents (LangGraph) as a subagent provider.
 *
 * Integrates into pets-agent's AgentManager as an alternative to:
 *   - claude-code  (Windows exe spawned via child_process)
 *   - codex        (npx openai/codex --acp --stdio)
 *   - kiro         (kiro --acp --stdio)
 *
 * Usage:
 *   // Inside AgentManager.spawn():
 *   case "deepagents":
 *     return this.spawnDeepAgents(prompt, opts);
 */

import type { Task } from "./task.js";
import { taskHistory } from "./task-history.js";

// Re-export so agent-manager.ts can reference Task type
export type { Task };

// Lazy-load deepagents to avoid import errors during tsx startup
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type DeepAgentAny = any;

let _deepagents: DeepAgentAny | null = null;

async function getDeepAgents(): Promise<DeepAgentAny> {
  if (_deepagents) return _deepagents;

  // Dynamic import — deepagents brings in a massive LangChain dependency tree
  // that shouldn't block the main agent from starting
  const mod = await import("deepagents");
  _deepagents = {
    createDeepAgent: mod.createDeepAgent,
    LocalShellBackend: mod.LocalShellBackend,
    ChatOpenAI: (await import("@langchain/openai")).ChatOpenAI,
  };
  return _deepagents;
}

interface DeepAgentsSession {
  task: Task;
  abortController: AbortController;
  cleanup: () => void;
}

const _sessions = new Map<string, DeepAgentsSession>();

/**
 * Spawn a deepagents session (runs in-process, not a child process).
 * This gives the orchestrator a LangGraph-powered agent with filesystem tools
 * and the ability to call subagents via the `task` tool.
 *
 * Output is streamed into task.progress so the REPL can display it in real time.
 */
export async function spawnDeepAgents(
  task: Task,
  prompt: string,
  workdir?: string
): Promise<void> {
  const config = await import("../config.js").then((m) => m.loadConfig());
  const getApiKey = (await import("../config.js")).getApiKey;
  const apiKey = getApiKey();

  const providerName = config.llm.provider as string;
  const modelId = config.llm.providers[config.llm.provider].model_id as string;

  const baseUrls: Record<string, string> = {
    "minimax-cn": "https://api.minimax.chat/v1",
    "openai": "https://api.openai.com/v1",
  };
  const baseUrl = baseUrls[providerName] ?? `https://api.${providerName}.com/v1`;

  const { createDeepAgent, LocalShellBackend, ChatOpenAI } = await getDeepAgents();

  const model = new ChatOpenAI({
    model: modelId,
    apiKey,
    configuration: { baseURL: baseUrl },
    temperature: 0,
    streaming: true,
  });

  const backend = await LocalShellBackend.create({
    rootDir: workdir ?? process.cwd(),
  });

  // Subagents mirror the external agents pets-agent can spawn.
  // They run inside the LangGraph agent as callable `task` subagents.
  const subagents = [
    {
      name: "claude-code",
      description: "Claude Code — Anthropic's CLI coding agent",
      systemPrompt:
        "You are Claude Code. Execute the user's request using the Claude Code CLI. " +
        "Install: npm install -g @anthropic/claude-code. " +
        "Command: claude -p --dangerously-skip-permissions --no-session-persistence --bare '<prompt>'",
    },
    {
      name: "codex",
      description: "OpenAI Codex CLI coding agent",
      systemPrompt:
        "You are OpenAI Codex. Execute the user's request via `npx -y openai/codex --acp --stdio`. " +
        "Send JSON message via stdin: {type:'user_message', content:'<prompt>'}",
    },
    {
      name: "kiro",
      description: "Kiro AI CLI coding agent",
      systemPrompt:
        "You are Kiro. Execute the user's request via `kiro --acp --stdio`.",
    },
  ];

  const agent = createDeepAgent({
    model,
    backend,
    subagents,
    generalPurposeAgent: false,
  });

  const abortController = new AbortController();

  _sessions.set(task.id, {
    task,
    abortController,
    cleanup: () => abortController.abort(),
  });

  // Stream the agent's output into task.progress
  task.status = "running";
  task.startedAt = new Date();

  try {
    const stream = await agent.stream({ messages: [] });

    let finished = false;

    for await (const chunk of stream) {
      if (finished || abortController.signal.aborted) break;

      const lines = extractLines(chunk);
      for (const line of lines) {
        if (line.trim()) {
          task.progress.push(line);
          taskHistory.appendLog(task.id, [line]);
        }
      }

      if (isFinished(chunk)) {
        finished = true;
        break;
      }
    }

    if (!finished && !abortController.signal.aborted) {
      task.status = "done";
    } else if (abortController.signal.aborted) {
      task.status = "cancelled";
    }
  } catch (err) {
    task.error = err instanceof Error ? err.message : String(err);
    task.status = "failed";
  }

  task.endedAt = new Date();
  taskHistory.add(task);
  _sessions.delete(task.id);
}

/** Abort a running deepagents session */
export function abortDeepAgents(taskId: string): void {
  const session = _sessions.get(taskId);
  if (session) {
    session.cleanup();
    _sessions.delete(taskId);
  }
}

// ---------------------------------------------------------------------------
// Stream utilities
// ---------------------------------------------------------------------------

function extractLines(chunk: unknown): string[] {
  if (!chunk || typeof chunk !== "object") return [];

  const c = chunk as Record<string, unknown>;

  // LangGraph ReactAgent message chunk
  if (c.type === "message" && c.data && typeof c.data === "object") {
    const d = c.data as Record<string, unknown>;
    if (Array.isArray(d.messages)) {
      const msgs = d.messages as Array<{ content?: string | unknown[] }>;
      const last = msgs[msgs.length - 1];
      if (typeof last?.content === "string") return last.content.split("\n");
      if (Array.isArray(last?.content)) {
        return (last.content as Array<{ text?: string }>)
          .map((b) => b.text ?? "")
          .join("\n")
          .split("\n");
      }
    }
  }

  // Tool call chunk
  if (c.type === "tool" && typeof c.data === "object") {
    const d = c.data as Record<string, unknown>;
    if (typeof d.content === "string") return d.content.split("\n");
  }

  // Generic text fields
  for (const key of ["content", "text", "message"]) {
    if (typeof c[key] === "string" && (c[key] as string).length < 1000) {
      return (c[key] as string).split("\n");
    }
  }

  // Serialise unknown shapes for debugging (short ones only)
  try {
    const s = JSON.stringify(chunk);
    if (s.length < 200) return [s];
  } catch {
    // ignore
  }

  return [];
}

function isFinished(chunk: unknown): boolean {
  if (!chunk || typeof chunk !== "object") return false;
  const c = chunk as Record<string, unknown>;
  return (
    c.type === "end" ||
    c.type === "done" ||
    c.event === "end" ||
    c.event === "done" ||
    Boolean(c.done) ||
    Boolean(c.__end__)
  );
}
