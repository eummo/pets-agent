/**
 * Task Tools — spawn_agent, list_tasks, get_task, kill_task,
 * list_task_history, decompose_task, get_task_tree, wait_for_tasks
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { agentManager } from "../tasks/agent-manager.js";
import { taskHistory } from "../tasks/task-history.js";
import type { AgentType } from "../tasks/task.js";
import type { Task } from "../tasks/task.js";

// ─── Runtime validation helpers ─────────────────────────────────────────────────

/** A validation error returned as a tool result */
function validationError(message: string, details: Record<string, unknown> = {}): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } {
  return { content: [{ type: "text", text: `Validation error: ${message}` }], details: { ...details, validationError: true } };
}

/**
 * Check that a required string param is truthy.
 * Returns an error result if invalid; otherwise returns null.
 */
function requireString(value: unknown, name: string): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null || (typeof value === "string" && !value.trim())) {
    return validationError(`${name} is required and must be a non-empty string`, { param: name });
  }
  return null;
}

/**
 * Validate optional numeric constraints.
 * Returns an error result if value is defined and fails the predicate; otherwise null.
 */
function validateOptionalNumber(value: unknown, name: string, predicate: (n: number) => boolean, message: string): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null) return null;
  if (typeof value !== "number" || !predicate(value)) {
    return validationError(`${name} must be ${message}`, { param: name, value });
  }
  return null;
}

/**
 * Validate that a value is one of the allowed enum string values.
 */
function validateEnum(value: unknown, name: string, allowed: string[]): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null) return null;
  if (typeof value !== "string" || !allowed.includes(value)) {
    return validationError(`${name} must be one of: ${allowed.join(", ")}`, { param: name, value });
  }
  return null;
}

// ─── Tool definition helpers ─────────────────────────────────────────────────

export function registerTaskTools(pi: ExtensionAPI): void {
  const SpawnAgentParams = Type.Object({
    agentType: Type.String({
      description: "Agent type: claude-code (preferred), pi-agent, codex, kiro, custom",
      enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
    }),
    prompt: Type.String({ description: "Task description for the sub-agent" }),
    name: Type.Optional(Type.String({ description: "Task name (optional)" })),
    workdir: Type.Optional(Type.String({ description: "Working directory (optional)" })),
    timeoutSec: Type.Optional(Type.Number({ description: "Timeout in seconds (default: no timeout)" })),
    maxRetries: Type.Optional(Type.Number({ description: "Max retry attempts on failure (default: 0)" })),
    priority: Type.Optional(Type.Number({ description: "Task priority 1-10, higher runs first (default: 5)" })),
  });

  const ListTasksParams = Type.Object({
    includeSuperseded: Type.Optional(Type.Boolean({ description: "Include superseded (retried) tasks. Default: false" })),
  });

  const GetTaskParams = Type.Object({
    taskId: Type.String({ description: "Task ID" }),
  });

  const KillTaskParams = Type.Object({
    taskId: Type.String({ description: "Task ID" }),
  });

  const ListTaskHistoryParams = Type.Object({
    taskId: Type.Optional(Type.String({ description: "Task ID (optional, get full log)" })),
    agentType: Type.Optional(Type.String({ description: "Filter by agent type" })),
    status: Type.Optional(Type.String({ description: "Filter by status: done, failed, cancelled" })),
    since: Type.Optional(Type.String({ description: "Start time ISO format, e.g. 2026-05-01" })),
    limit: Type.Optional(Type.Number({ description: "Max results, default 20" })),
    offset: Type.Optional(Type.Number({ description: "Number of results to skip for pagination, default 0" })),
  });

  const DecomposeTaskParams = Type.Object({
    taskDescription: Type.String({ description: "Complex task to decompose" }),
    subtasks: Type.Array(
      Type.Object({
        title: Type.String({ description: "Subtask title" }),
        agentType: Type.String({
          description: "Agent type",
          enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
        }),
        prompt: Type.String({ description: "Task description for the sub-agent" }),
      }),
      { description: "Subtasks array" },
    ),
    parentId: Type.Optional(Type.String({ description: "Parent task ID (optional)" })),
  });

  const RECENT_LINES = 20;

  // ─── spawn_agent ───────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "spawn_agent",
    label: "Spawn Sub-Agent",
    description: [
      "Launch a sub-agent (Claude Code, Codex, Kiro, etc.) in an isolated process.",
      "Preferred agent: claude-code for general coding tasks.",
      "Use pi-agent when you need pi-mono framework capabilities.",
    ].join(" "),
    parameters: SpawnAgentParams,

    async execute(_toolCallId, params, signal, onUpdate, _ctx) {
      // ─── Runtime validation ───────────────────────────────────────────────
      const err1 = requireString(params.agentType, "agentType");
      if (err1) return err1;
      const err2 = requireString(params.prompt, "prompt");
      if (err2) return err2;
      const err3 = validateEnum(params.agentType, "agentType", ["claude-code", "pi-agent", "codex", "kiro", "custom"]);
      if (err3) return err3;
      const err4 = validateOptionalNumber(params.timeoutSec, "timeoutSec", (n) => n > 0, "a positive integer (seconds)");
      if (err4) return err4;
      const err5 = validateOptionalNumber(params.maxRetries, "maxRetries", (n) => n >= 0, "a non-negative integer");
      if (err5) return err5;
      const err6 = validateOptionalNumber(params.priority, "priority", (n) => n >= 1 && n <= 10, "an integer between 1 and 10");
      if (err6) return err6;
      // ─── End validation ─────────────────────────────────────────────────

      const timeoutMs = params.timeoutSec != null ? params.timeoutSec * 1000 : undefined;
      const maxRetries = params.maxRetries ?? 0;

      // Wire AbortSignal to kill the task if the outer execution is cancelled
      let abortHandler: (() => void) | undefined;
      if (signal) {
        abortHandler = () => agentManager.killByToken(params.name ?? params.agentType);
        signal.addEventListener("abort", abortHandler);
      }

      const task = agentManager.spawnWithRetry(
        params.agentType as AgentType,
        params.prompt,
        {
          name: params.name,
          workdir: params.workdir,
          timeoutMs,
          maxRetries,
          priority: params.priority,
          token: params.name ?? params.agentType,
        },
      );

      // Stream progress updates via onUpdate if provided
      let unsub: (() => void) | undefined;
      if (onUpdate) {
        unsub = agentManager.subscribe(task.id, (update) => {
          onUpdate({ content: [{ type: "text", text: formatTaskUpdate(update) }], details: { taskId: task.id, status: update.status } });
        });
      }

      // Clean up abort handler and subscription on return
      const cleanup = () => {
        if (abortHandler && signal) signal.removeEventListener("abort", abortHandler);
        unsub?.();
      };

      const initialLines = task.progress.slice(-5);
      const initialOutput =
        initialLines.length > 0
          ? `Task started: ${task.name} (${task.id})\nRecent output:\n${initialLines.join("\n")}`
          : `Task started: ${task.name} (${task.id})\nWaiting for output...`;

      // Note: caller is responsible for calling cleanup() when the tool call completes.
      // For sync return, we attach cleanup via AbortSignal if available.
      if (signal) {
        const origAbort = abortHandler;
        signal.addEventListener("abort", () => {
          origAbort?.();
          unsub?.();
        });
      }

      return {
        content: [{ type: "text", text: initialOutput }],
        details: { taskId: task.id, name: task.name, status: task.status },
      };
    },
  }));

  // ─── list_tasks ────────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "list_tasks",
    label: "List Tasks",
    description: "List all sub-agent tasks and their current status.",
    parameters: ListTasksParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const tasks = agentManager.list(params.includeSuperseded);
      const active = tasks.filter((t) => t.status === "running" || t.status === "pending");
      const done = tasks.filter((t) => t.status === "done");
      const failed = tasks.filter((t) => t.status === "failed");
      const cancelled = tasks.filter((t) => t.status === "cancelled");

      const lines: string[] = [
        `Total: ${tasks.length} | Running: ${active.length} | Done: ${done.length} | Failed: ${failed.length} | Cancelled: ${cancelled.length}`,
        "",
      ];

      if (active.length > 0) {
        lines.push("--- Running ---");
        for (const t of active) {
          const pri = t.priority != null ? ` pri:${t.priority}` : "";
          lines.push(`[${t.status}] ${t.name} (${t.id}) - ${t.agentType}${pri}`);
        }
      }

      if (done.length > 0) {
        lines.push("--- Done ---");
        for (const t of done) {
          lines.push(`[done] ${t.name} (${t.id}) - exit:${t.exitCode ?? "?"}`);
        }
      }

      if (failed.length > 0) {
        lines.push("--- Failed ---");
        for (const t of failed) {
          lines.push(`[failed] ${t.name} (${t.id}) - ${t.error ?? "unknown"}`);
        }
      }

      if (cancelled.length > 0) {
        lines.push("--- Cancelled ---");
        for (const t of cancelled) {
          lines.push(`[cancelled] ${t.name} (${t.id})`);
        }
      }

      if (tasks.length === 0) {
        lines.push("(no tasks)");
      }

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { tasks },
      };
    },
  }));

  // ─── get_task ───────────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "get_task",
    label: "Get Task Details",
    description: "View detailed status and recent output of a specific task.",
    parameters: GetTaskParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err = requireString(params.taskId, "taskId");
      if (err) return err;

      const task = agentManager.get(params.taskId);

      if (!task) {
        return {
          content: [{ type: "text", text: `Task not found: ${params.taskId}` }],
          details: { error: "Task not found", taskId: params.taskId },
        };
      }

      const recentProgress = task.progress.slice(-RECENT_LINES);
      const output = [
        `Task: ${task.name}`,
        `ID: ${task.id}`,
        `Type: ${task.agentType}`,
        `Status: ${task.status}`,
        task.priority != null ? `Priority: ${task.priority}` : "",
        `Created: ${task.createdAt.toISOString()}`,
        task.startedAt ? `Started: ${task.startedAt.toISOString()}` : "",
        task.endedAt ? `Ended: ${task.endedAt.toISOString()}` : "",
        task.exitCode !== undefined ? `Exit code: ${task.exitCode}` : "",
        task.error ? `Error: ${task.error}` : "",
        task.workdir ? `Working dir: ${task.workdir}` : "",
        task.parentId ? `Parent: ${task.parentId}` : "",
        task.attempt != null ? `Attempt: ${task.attempt}` : "",
        "",
        `--- Recent output (${task.progress.length} lines) ---`,
        ...(recentProgress.length > 0 ? recentProgress : ["(no output)"]),
      ]
        .filter(Boolean)
        .join("\n");

      return {
        content: [{ type: "text", text: output }],
        details: { task },
      };
    },
  }));

  // ─── kill_task ──────────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "kill_task",
    label: "Kill Task",
    description: "Forcefully stop a running sub-agent task.",
    parameters: KillTaskParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err = requireString(params.taskId, "taskId");
      if (err) return err;

      const task = agentManager.get(params.taskId);

      if (!task) {
        return {
          content: [{ type: "text", text: `Task not found: ${params.taskId}` }],
          details: { error: "Task not found", taskId: params.taskId },
        };
      }

      agentManager.kill(params.taskId);

      return {
        content: [{ type: "text", text: `Killed task: ${task.name} (${params.taskId})` }],
        details: { taskId: params.taskId, status: "cancelled" },
      };
    },
  }));

  // ─── list_task_history ─────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "list_task_history",
    label: "Task History",
    description: "Query sub-agent execution history. Pass taskId for full log, or filter by agentType/status/time.",
    parameters: ListTaskHistoryParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      // Full log for specific taskId
      if (params.taskId) {
        const logLines = taskHistory.readLog(params.taskId);
        const entry = taskHistory.getAll().find((e) => e.id === params.taskId);

        if (logLines.length === 0 && !entry) {
          return {
            content: [{ type: "text", text: `Task not found: ${params.taskId}` }],
            details: { error: "Task not found", taskId: params.taskId },
          };
        }

        const lines: string[] = [`=== Task ${params.taskId} ===`];
        if (entry) {
          lines.push(
            `Name: ${entry.name} | Type: ${entry.agentType} | Status: ${entry.status}`,
            `Created: ${new Date(entry.createdAt).toLocaleString("zh-CN")}`,
          );
          const promptFormatted = entry.prompt.replace(/\n+/g, " ").trim();
          const promptDisplay =
            promptFormatted.length > 100 ? promptFormatted.slice(0, 100) + "..." : promptFormatted;
          lines.push(`User request: ${promptDisplay}`, "");
        }
        lines.push(...(logLines.length > 0 ? logLines : ["(log empty)"]));

        return { content: [{ type: "text", text: lines.join("\n") }], details: { entry, logLines } };
      }

      // Query mode
      const query = {
        agentType: params.agentType,
        status: params.status,
        since: params.since,
        limit: params.limit ?? 20,
        offset: params.offset ?? 0,
      };
      const entries = taskHistory.query(query);

      if (entries.length === 0) {
        return { content: [{ type: "text", text: "No history records found" }], details: {} };
      }

      const lines: string[] = [`Found ${entries.length} records:`, ""];
      for (const e of entries) {
        const date = new Date(e.createdAt).toLocaleString("zh-CN");
        const duration =
          e.startedAt && e.endedAt
            ? Math.round(
                (new Date(e.endedAt).getTime() - new Date(e.startedAt).getTime()) / 1000,
              ) + "s"
            : "-";
        const files = e.fileCount !== undefined ? ` ${e.fileCount} files` : "";
        const error =
          e.status === "failed" ? ` ⚠️ ${(e.error ?? "error").slice(0, 50)}` : "";
        lines.push(
          `[${e.status}] ${date} | ${e.agentType} | ${duration} | ${e.name}${files}${error}`,
        );
        const promptSummary =
          e.prompt.length > 60 ? e.prompt.slice(0, 60) + "..." : e.prompt;
        lines.push(`  → ${promptSummary}`);
      }

      return { content: [{ type: "text", text: lines.join("\n") }], details: { entries } };
    },
  }));

  // ─── decompose_task ─────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "decompose_task",
    label: "Decompose Task",
    description: [
      "Decompose a complex task into multiple parallel subtasks.",
      "Use when: multi-step workflows, cross-domain research, large implementations.",
      "All subtasks start in parallel automatically.",
      "Monitor with list_tasks / get_task.",
    ].join(" "),
    parameters: DecomposeTaskParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.taskDescription, "taskDescription");
      if (err1) return err1;
      if (!params.subtasks || params.subtasks.length === 0) {
        return {
          content: [
            { type: "text", text: "Error: subtasks cannot be empty. Provide at least one subtask." },
          ],
          details: { error: "Empty subtasks" },
        };
      }
      for (const [i, sub] of params.subtasks.entries()) {
        const err = requireString(sub.agentType, `subtasks[${i}].agentType`);
        if (err) return err;
        const errP = requireString(sub.prompt, `subtasks[${i}].prompt`);
        if (errP) return errP;
      }
      if (params.parentId && !agentManager.get(params.parentId)) {
        return {
          content: [{ type: "text", text: `Error: parent task not found: ${params.parentId}` }],
          details: { error: "Parent task not found", parentId: params.parentId },
        };
      }

      const spawned = [];
      const lines: string[] = [
        `Task decomposition: "${params.taskDescription.slice(0, 80)}${
          params.taskDescription.length > 80 ? "..." : ""
        }"`,
        "",
      ];

      for (const sub of params.subtasks) {
        const opts: { name?: string; workdir?: string; parentId?: string } = { name: sub.title };
        if (params.parentId) opts.parentId = params.parentId;

        const task =
          params.parentId
            ? agentManager.spawnChild(params.parentId, sub.agentType as AgentType, sub.prompt, opts)
            : agentManager.spawn(sub.agentType as AgentType, sub.prompt, opts);

        spawned.push({ id: task.id, name: task.name, agentType: sub.agentType, status: task.status });
        lines.push(`  ✓ [${sub.agentType}] ${sub.title} → ${task.id}`);
      }

      lines.push("", `Started ${spawned.length} subtasks. Use list_tasks to monitor progress.`);

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { taskDescription: params.taskDescription, parentId: params.parentId, subtasks: spawned },
      };
    },
  }));

  // ─── get_task_tree ─────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "get_task_tree",
    label: "Get Task Tree",
    description: "View the task dependency DAG (tree) for a task and all its descendants.",
    parameters: Type.Object({
      taskId: Type.Optional(Type.String({ description: "Root task ID. Omit to show all top-level tasks." })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const allTasks = agentManager.list(true); // include superseded

      interface TreeNode {
        id: string;
        name: string;
        status: string;
        agentType: string;
        attempt?: number;
        children: TreeNode[];
      }

      const buildTree = (taskId: string): TreeNode | null => {
        const t = allTasks.find((x) => x.id === taskId);
        if (!t) return null;
        return {
          id: t.id,
          name: t.name,
          status: t.status,
          agentType: t.agentType,
          attempt: t.attempt,
          children: (t.children ?? []).map((cid) => buildTree(cid)).filter(Boolean) as TreeNode[],
        };
      };

      const renderTree = (node: TreeNode, indent = 0): string[] => {
        const prefix = "  ".repeat(indent);
        const superseded = allTasks.find((t) => t.id === node.id)?.supersededBy;
        const supers = superseded ? ` [superseded by ${superseded.slice(0, 8)}]` : "";
        const attempt = node.attempt != null ? ` x${node.attempt}` : "";
        const lines = [`${prefix}├── [${node.status}] ${node.name} (${node.id.slice(0, 8)}) ${node.agentType}${attempt}${supers}`];
        for (let i = 0; i < node.children.length; i++) {
          const child = node.children[i];
          if (i === node.children.length - 1) {
            lines.push(...renderTree(child!, indent + 1).map((l) => l.replace("├──", "└──")));
          } else {
            lines.push(...renderTree(child!, indent + 1));
          }
        }
        return lines;
      };

      const lines: string[] = [];
      if (params.taskId) {
        const root = buildTree(params.taskId);
        if (!root) {
          return { content: [{ type: "text", text: `Task not found: ${params.taskId}` }], details: {} };
        }
        lines.push(`Task DAG for: ${root.name} (${root.id})`, "");
        lines.push(...renderTree(root));
      } else {
        // Show all root tasks (no parent)
        const roots = allTasks.filter((t) => !t.parentId && !t.supersededBy);
        lines.push(`All top-level tasks (${roots.length} roots):`, "");
        for (const root of roots) {
          lines.push(...renderTree(buildTree(root.id)!));
          lines.push("");
        }
      }

      return { content: [{ type: "text", text: lines.join("\n") }], details: {} };
    },
  }));

  // ─── wait_for_tasks ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "wait_for_tasks",
    label: "Wait For Tasks",
    description: "Wait for one or more tasks to complete. Returns when all targets are done/failed/cancelled, or on timeout.",
    parameters: Type.Object({
      taskIds: Type.Array(Type.String(), { description: "Task IDs to wait for" }),
      timeoutSec: Type.Optional(Type.Number({ description: "Max seconds to wait (default: 300)" })),
      pollIntervalMs: Type.Optional(Type.Number({ description: "Poll interval in ms (default: 2000)" })),
    }),

    async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
      if (!Array.isArray(params.taskIds) || params.taskIds.length === 0) {
        return validationError("taskIds must be a non-empty array of task ID strings", { param: "taskIds" });
      }
      const err1 = validateOptionalNumber(params.timeoutSec, "timeoutSec", (n) => n > 0, "a positive integer");
      if (err1) return err1;
      const err2 = validateOptionalNumber(params.pollIntervalMs, "pollIntervalMs", (n) => n >= 100, "a positive integer >= 100ms");
      if (err2) return err2;

      const timeoutMs = (params.timeoutSec ?? 300) * 1000;
      const pollMs = params.pollIntervalMs ?? 2000;
      const deadline = Date.now() + timeoutMs;

      const checkDone = (): { done: boolean; results: Array<{ taskId: string; status: string; error?: string }> } => {
        const results: Array<{ taskId: string; status: string; error?: string }> = [];
        let allDone = true;
        for (const id of params.taskIds) {
          const task = agentManager.get(id);
          if (!task) {
            results.push({ taskId: id, status: "not_found" });
            continue;
          }
          if (task.status === "done" || task.status === "failed" || task.status === "cancelled") {
            results.push({ taskId: id, status: task.status, error: task.error });
          } else {
            allDone = false;
            results.push({ taskId: id, status: task.status });
          }
        }
        return { done: allDone, results };
      };

      // Poll until done or timeout
      while (Date.now() < deadline) {
        if (signal?.aborted) {
          return { content: [{ type: "text", text: "Wait aborted by signal." }], details: {} };
        }
        const { done, results } = checkDone();
        if (done) {
          const lines = ["All tasks completed:", ""];
          for (const r of results) {
            const icon = r.status === "done" ? "✓" : r.status === "failed" ? "✗" : "○";
            lines.push(`  ${icon} [${r.status}] ${r.taskId}${r.error ? ` — ${r.error}` : ""}`);
          }
          return { content: [{ type: "text", text: lines.join("\n") }], details: { results } };
        }
        // Wait before next poll
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }

      const { results } = checkDone();
      return {
        content: [{ type: "text", text: `Timeout after ${params.timeoutSec ?? 300}s. Still pending: ${results.filter((r) => !["done", "failed", "cancelled", "not_found"].includes(r.status)).map((r) => r.taskId).join(", ")}` }],
        details: { results, timedOut: true },
      };
    },
  }));
}

function formatTaskUpdate(update: { id?: string; status: string; progress?: string[]; error?: string; exitCode?: number }): string {
  const lines: string[] = [];
  if (update.status) lines.push(`[${update.status}]`);
  if (update.progress && update.progress.length > 0) {
    lines.push(...update.progress.slice(-5));
  }
  if (update.error) lines.push(`Error: ${update.error}`);
  if (update.exitCode !== undefined) lines.push(`Exit code: ${update.exitCode}`);
  return lines.join("\n") || `Task update: ${update.status}`;
}
