/**
 * Pets-Agent Extension for pi-coding-agent
 *
 * Registers orchestrator tools (spawn_agent, decompose_task, list_tasks, etc.)
 * that delegate to sub-agents (claude-code, codex, pi-agent, kiro) via
 * the existing AgentManager process-spawning engine.
 *
 * Usage: pi --extension pets-agent
 */

import { defineTool, type ExtensionAPI, type Theme } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { agentManager } from "./tasks/agent-manager.js";
import { taskHistory } from "./tasks/task-history.js";
import type { AgentType } from "./tasks/task.js";
import "./tasks/task-history.js"; // side-effect: ensures singleton init
import { patternMemory } from "./memory/pattern-memory.js";
import { preferenceMemory } from "./memory/preference-memory.js";
import { projectMemory } from "./memory/project-memory.js";
import { memoryInjector } from "./memory/injector.js";

// ============================================================================
// Tool Parameters (TypeBox schemas)
// ============================================================================

const SpawnAgentParams = Type.Object({
  agentType: Type.String({
    description: "Agent type: claude-code (preferred), pi-agent, codex, kiro, custom",
    enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
  }),
  prompt: Type.String({ description: "Task description for the sub-agent" }),
  name: Type.Optional(Type.String({ description: "Task name (optional)" })),
  workdir: Type.Optional(Type.String({ description: "Working directory (optional)" })),
});

const ListTasksParams = Type.Object({});

const GetTaskParams = Type.Object({
  taskId: Type.String({ description: "Task ID" }),
});

const KillTaskParams = Type.Object({
  taskId: Type.String({ description: "Task ID" }),
});

const ListTaskHistoryParams = Type.Object({
  taskId: Type.Optional(Type.String({ description: "Task ID (optional, get full log)" })),
  agentType: Type.Optional(
    Type.String({ description: "Filter by agent type: claude-code, codex, kiro, pi-agent" }),
  ),
  status: Type.Optional(Type.String({ description: "Filter by status: done, failed, cancelled" })),
  since: Type.Optional(Type.String({ description: "Start time (ISO format, e.g. 2026-05-01)" })),
  limit: Type.Optional(Type.Number({ description: "Max results, default 20" })),
});

const DecomposeTaskParams = Type.Object({
  taskDescription: Type.String({ description: "Complex task to decompose" }),
  subtasks: Type.Array(
    Type.Object({
      title: Type.String({ description: "Subtask title" }),
      agentType: Type.String({
        description: "Agent type: claude-code, pi-agent, codex, kiro, custom",
        enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
      }),
      prompt: Type.String({ description: "Task description for the sub-agent" }),
    }),
    { description: "Subtasks array" },
  ),
  parentId: Type.Optional(Type.String({ description: "Parent task ID (optional)" })),
});

// Memory tools

const RememberPatternParams = Type.Object({
  pattern: Type.String({ description: "Command or workflow pattern to remember" }),
  tags: Type.Optional(Type.String({ description: "Comma-separated tags, e.g. 'npm,build,vite'" })),
});

const RememberPrefsParams = Type.Object({
  agentType: Type.String({ description: "Agent type: claude-code, pi-agent, codex, kiro" }),
  taskPrompt: Type.String({ description: "Brief description of the task type" }),
  success: Type.Boolean({ description: "Whether the task succeeded" }),
  exitCode: Type.Optional(Type.Number()),
  durationSec: Type.Optional(Type.Number()),
});

const RememberProjectParams = Type.Object({
  workdir: Type.String({ description: "Project directory path" }),
  content: Type.String({ description: "Context to remember about this project" }),
  tags: Type.Optional(Type.String({ description: "Comma-separated tags" })),
});

const GetMemoryParams = Type.Object({
  type: Type.Optional(Type.String({ description: "Memory type: patterns, preferences, project. Default: all" })),
  workdir: Type.Optional(Type.String({ description: "Project directory for project memory" })),
  query: Type.Optional(Type.String({ description: "Search query for patterns/preferences" })),
});

const ForgetMemoryParams = Type.Object({
  type: Type.String({ description: "Memory type: patterns, preferences", enum: ["patterns", "preferences"] }),
  idOrText: Type.String({ description: "Entry ID or substring to match and remove" }),
});

// ============================================================================
// Tool result types
// ============================================================================

interface TaskDetails {
  taskId: string;
  name: string;
  status: string;
}

interface TaskListDetails {
  tasks: ReturnType<typeof agentManager.list>;
}

interface DecomposeDetails {
  taskDescription: string;
  parentId?: string;
  subtasks: Array<{ id: string; name: string; agentType: string; status: string }>;
}

const RECENT_LINES = 20;

// ============================================================================
// Extension Factory
// ============================================================================

export default function petsAgentExtension(pi: ExtensionAPI): void {
  // -------------------------------------------------------------------------
  // spawn_agent
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "spawn_agent",
      label: "Spawn Sub-Agent",
      description: [
        "Launch a sub-agent (Claude Code, Codex, Kiro, etc.) in an isolated process.",
        "Preferred agent: claude-code for general coding tasks.",
        "Use pi-agent when you need pi-mono framework capabilities.",
      ].join(" "),
      parameters: SpawnAgentParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        const task = agentManager.spawn(params.agentType as AgentType, params.prompt, {
          name: params.name,
          workdir: params.workdir,
        });

        const initialLines = task.progress.slice(-5);
        const initialOutput =
          initialLines.length > 0
            ? `Task started: ${task.name} (${task.id})\nRecent output:\n${initialLines.join("\n")}`
            : `Task started: ${task.name} (${task.id})\nWaiting for output...`;

        return {
          content: [{ type: "text", text: initialOutput }],
          details: { taskId: task.id, name: task.name, status: task.status } as TaskDetails,
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // list_tasks
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "list_tasks",
      label: "List Tasks",
      description: "List all sub-agent tasks and their current status.",
      parameters: ListTasksParams,

      async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
        const tasks = agentManager.list();
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
            lines.push(`[${t.status}] ${t.name} (${t.id}) - ${t.agentType}`);
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
          details: { tasks } as TaskListDetails,
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // get_task
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "get_task",
      label: "Get Task Details",
      description: "View detailed status and recent output of a specific task.",
      parameters: GetTaskParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
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
          `Created: ${task.createdAt.toISOString()}`,
          task.startedAt ? `Started: ${task.startedAt.toISOString()}` : "",
          task.endedAt ? `Ended: ${task.endedAt.toISOString()}` : "",
          task.exitCode !== undefined ? `Exit code: ${task.exitCode}` : "",
          task.error ? `Error: ${task.error}` : "",
          task.workdir ? `Working dir: ${task.workdir}` : "",
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
    }),
  );

  // -------------------------------------------------------------------------
  // kill_task
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "kill_task",
      label: "Kill Task",
      description: "Forcefully stop a running sub-agent task.",
      parameters: KillTaskParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
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
    }),
  );

  // -------------------------------------------------------------------------
  // list_task_history
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "list_task_history",
      label: "Task History",
      description:
        "Query sub-agent execution history. Pass taskId for full log, or filter by agentType/status/time.",
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
    }),
  );

  // -------------------------------------------------------------------------
  // decompose_task
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
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
        if (!params.subtasks || params.subtasks.length === 0) {
          return {
            content: [
              { type: "text", text: "Error: subtasks cannot be empty. Provide at least one subtask." },
            ],
            details: { error: "Empty subtasks" },
          };
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
          details: { taskDescription: params.taskDescription, parentId: params.parentId, subtasks: spawned } as DecomposeDetails,
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // remember_pattern — store a successful command/workflow pattern
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "remember_pattern",
      label: "Remember Pattern",
      description: "Save a successful command or workflow pattern for future reference.",
      parameters: RememberPatternParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        const tags = params.tags ? params.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];
        const result = patternMemory.add(params.pattern, { tags, source: "user" });

        if (result.success) {
          const usage = patternMemory.usage();
          return {
            content: [{ type: "text", text: `Pattern saved [${usage.pct}% — ${usage.current}/${usage.limit} chars]\n${params.pattern}` }],
            details: { saved: true },
          };
        }
        return {
          content: [{ type: "text", text: `Failed to save pattern: ${result.error}` }],
          details: { saved: false, error: result.error },
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // remember_prefs — record agent preference outcome
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "remember_prefs",
      label: "Remember Preference",
      description: "Record which agent type succeeded for a given task type.",
      parameters: RememberPrefsParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        preferenceMemory.recordOutcome({
          agentType: params.agentType,
          taskPrompt: params.taskPrompt,
          success: params.success,
          exitCode: params.exitCode,
          durationSec: params.durationSec,
        });
        const suggested = preferenceMemory.suggestAgentType(params.taskPrompt);
        const msg = suggested
          ? `Recorded. Suggested agent for '${params.taskPrompt.slice(0, 50)}': ${suggested}`
          : "Recorded. No strong pattern yet.";
        return { content: [{ type: "text", text: msg }], details: { suggested } };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // remember_project — store per-project context
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "remember_project",
      label: "Remember Project",
      description: "Save context about a specific project (tech stack, key files, conventions).",
      parameters: RememberProjectParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        const tags = params.tags ? params.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];
        const store = projectMemory.store(params.workdir);
        const result = store.addToProject(params.content, { tags, source: "user" });

        if (result.success) {
          const usage = store.usage();
          return {
            content: [{ type: "text", text: `Project context saved for ${params.workdir} [${usage.pct}%]\n${params.content}` }],
            details: { saved: true },
          };
        }
        return {
          content: [{ type: "text", text: `Failed: ${result.error}` }],
          details: { saved: false, error: result.error },
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // get_memory — view/search memory stores
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "get_memory",
      label: "Get Memory",
      description: "View or search memory stores. Shows status summary by default.",
      parameters: GetMemoryParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        const type = params.type ?? "all";

        if (type === "patterns" || type === "all") {
          const query = params.query ?? "";
          const results = patternMemory.search(query);
          const usage = patternMemory.usage();
          const lines = [
            `--- Patterns [${usage.pct}% — ${results.length} entries] ---`,
            ...results.map((e) => {
              const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
              return `${tags}\n${e.content}`;
            }),
          ];
          if (type === "patterns") return { content: [{ type: "text", text: lines.join("\n") }], details: { results } };
        }

        if (type === "preferences" || type === "all") {
          const query = params.query ?? "";
          const results = preferenceMemory.query(query);
          const usage = preferenceMemory.usage();
          const lines = [
            `--- Preferences [${usage.pct}% — ${results.length} entries] ---`,
            ...results.map((e) => {
              const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
              return `${tags}\n${e.content}`;
            }),
          ];
          if (type === "preferences") return { content: [{ type: "text", text: lines.join("\n") }], details: { results } };
        }

        if (type === "project" && params.workdir) {
          const store = projectMemory.store(params.workdir);
          const snap = store.getSnapshot();
          const stack = projectMemory.detectTechStack(params.workdir);
          const lines = [
            `--- Project: ${params.workdir} [${stack.join(", ") || "unknown stack"}] ---`,
            snap || "(no entries)",
          ];
          return { content: [{ type: "text", text: lines.join("\n") }], details: {} };
        }

        // Default: status overview
        const status = memoryInjector.status();
        return {
          content: [{
            type: "text",
            text: [
              "Memory Status:",
              `  Patterns: ${status.patterns.count} entries, ${status.patterns.usage.pct}%`,
              `  Preferences: ${status.preferences.count} entries, ${status.preferences.usage.pct}%`,
              "",
              "Use get_memory(type='patterns'|'preferences'|'project', workdir='...') to view details.",
            ].join("\n"),
          }],
          details: { status },
        };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // forget_memory — remove a memory entry
  // -------------------------------------------------------------------------
  pi.registerTool(
    defineTool({
      name: "forget_memory",
      label: "Forget Memory",
      description: "Remove a memory entry by ID or content match.",
      parameters: ForgetMemoryParams,

      async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
        const result =
          params.type === "patterns"
            ? patternMemory.remove(params.idOrText)
            : preferenceMemory.remove(params.idOrText);

        if (result.success) {
          return { content: [{ type: "text", text: `Removed from ${params.type}: '${params.idOrText}'` }], details: { removed: true } };
        }
        return { content: [{ type: "text", text: result.error ?? "Not found" }], details: { removed: false } };
      },
    }),
  );

  // -------------------------------------------------------------------------
  // System prompt — inject orchestrator instructions into the agent
  // -------------------------------------------------------------------------
  pi.on("before_agent_start", (event) => {
    const workdir = event.systemPromptOptions.cwd ?? process.cwd();
    const memoryBlock = memoryInjector.buildBlock({ workdir });

    const orchestratorSection = `
## Orchestration Capabilities

You are a development assistant with agent orchestration capabilities.

**Orchestrator Tools:**
- spawn_agent(agentType, prompt) — launch a sub-agent (claude-code preferred for coding)
- list_tasks — view all running/completed tasks
- get_task(taskId) — view task details and recent output
- kill_task(taskId) — stop a running task
- list_task_history — query past task executions
- decompose_task(taskDescription, subtasks) — split complex tasks into parallel subtasks

**Memory Tools:**
- remember_pattern(pattern, tags?) — save a successful command/workflow
- remember_prefs(agentType, taskPrompt, success, exitCode?, durationSec?) — record agent performance
- remember_project(workdir, content, tags?) — save per-project context
- get_memory(type?, workdir?, query?) — view/search memory
- forget_memory(type, idOrText) — remove a memory entry

**Agent Selection Priority:**
1. claude-code — general coding, file operations, debugging
2. pi-agent — when pi-mono framework capabilities are needed
3. codex / kiro — fallback options

**Task Decomposition:**
When a task spans multiple domains, requires independent steps, or is large in scope,
use decompose_task to split it into parallel subtasks, then monitor with list_tasks.
Simple single-step tasks should use spawn_agent directly.
`.trim();

    return {
      systemPrompt: `${event.systemPrompt}${memoryBlock}\n\n${orchestratorSection}`,
    };
  });

  // -------------------------------------------------------------------------
  // Custom header — replace built-in onboarding with pets-agent branding
  // -------------------------------------------------------------------------
  pi.on("session_start", async (_event, ctx) => {
    if (ctx.hasUI) {
      ctx.ui.setHeader((_tui, theme) => {
        const accent = (text: string) => theme.fg("accent", text);
        const muted = (text: string) => theme.fg("muted", text);
        const dim = (text: string) => theme.fg("dim", text);
        return {
          render(_width: number): string[] {
            const width = 43;
            const content = "   Pets-Agent  ·  开发助手";
            const contentCells = 27;
            const pad = width - contentCells;
            const left = " ".repeat(pad >> 1);
            const right = " ".repeat(pad - (pad >> 1));
            return [
              "",
              left + content + right,
              "",
              `${muted("/commands")} ${dim("· /help for available slash commands")}`,
              "",
            ];
          },
          invalidate() {},
        };
      });
    }
  });
}
