import { registry } from "./registry.js";
import { agentManager } from "../tasks/agent-manager.js";
import type { AgentType } from "../tasks/task.js";
import { taskHistory } from "../tasks/task-history.js";
import type { TaskHistoryQuery } from "../tasks/task-history.js";
import type { ToolDef } from "./registry.js";

const RECENT_LINES = 20;

function makeTool(def: Omit<ToolDef, "execute"> & { execute: ToolDef["execute"] }): ToolDef {
  return def as ToolDef;
}

const spawnAgentTool: ToolDef = makeTool({
  name: "spawn_agent",
  label: "启动子 Agent",
  description: "启动一个子 agent（Claude Code、Codex、Kiro 等），在独立进程中运行",
  parameters: {
    type: "object",
    properties: {
      agentType: {
        type: "string",
        description: "Agent 类型: claude-code (优先), pi-agent, codex, kiro, custom",
        enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
      },
      prompt: {
        type: "string",
        description: "给子 agent 的任务描述",
      },
      name: {
        type: "string",
        description: "任务名称（可选）",
      },
      workdir: {
        type: "string",
        description: "工作目录（可选）",
      },
    },
    required: ["agentType", "prompt"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { agentType: AgentType; prompt: string; name?: string; workdir?: string };
  },
  async execute(_toolCallId, params) {
    const p = params as { agentType: AgentType; prompt: string; name?: string; workdir?: string };
    const task = agentManager.spawn(p.agentType, p.prompt, { name: p.name, workdir: p.workdir });

    const initialLines = task.progress.slice(-5);
    const initialOutput = initialLines.length > 0
      ? `任务已启动: ${task.name} (${task.id})\n最近输出:\n${initialLines.join("\n")}`
      : `任务已启动: ${task.name} (${task.id})\n等待输出...`;

    return {
      content: [{ type: "text", text: initialOutput }],
      details: { taskId: task.id, name: task.name, status: task.status },
    };
  },
});

const listTasksTool: ToolDef = makeTool({
  name: "list_tasks",
  label: "列出所有任务",
  description: "列出所有已创建的子 agent 任务及其当前状态",
  parameters: {
    type: "object",
    properties: {},
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as Record<string, unknown>;
  },
  async execute(_toolCallId) {
    const tasks = agentManager.list();
    const active = tasks.filter((t) => t.status === "running" || t.status === "pending");
    const done = tasks.filter((t) => t.status === "done");
    const failed = tasks.filter((t) => t.status === "failed");
    const cancelled = tasks.filter((t) => t.status === "cancelled");

    const lines = [
      `总任务数: ${tasks.length}`,
      `运行中: ${active.length} | 完成: ${done.length} | 失败: ${failed.length} | 已取消: ${cancelled.length}`,
      "",
    ];

    if (active.length > 0) {
      lines.push("--- 运行中 ---");
      for (const t of active) {
        lines.push(`[${t.status}] ${t.name} (${t.id}) - ${t.agentType}`);
      }
    }

    if (done.length > 0) {
      lines.push("--- 已完成 ---");
      for (const t of done) {
        lines.push(`[done] ${t.name} (${t.id}) - exit:${t.exitCode ?? "?"}`);
      }
    }

    if (failed.length > 0) {
      lines.push("--- 失败 ---");
      for (const t of failed) {
        lines.push(`[failed] ${t.name} (${t.id}) - ${t.error ?? "unknown"}`);
      }
    }

    return {
      content: [{ type: "text", text: lines.join("\n") }],
      details: { tasks },
    };
  },
});

const getTaskTool: ToolDef = makeTool({
  name: "get_task",
  label: "查看任务详情",
  description: "查看指定任务的详细信息和最新进度输出",
  parameters: {
    type: "object",
    properties: {
      taskId: {
        type: "string",
        description: "任务 ID",
      },
    },
    required: ["taskId"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { taskId: string };
  },
  async execute(_toolCallId, params) {
    const p = params as { taskId: string };
    const task = agentManager.get(p.taskId);

    if (!task) {
      return {
        content: [{ type: "text", text: `任务不存在: ${p.taskId}` }],
        details: { error: "Task not found" },
      };
    }

    const recentProgress = task.progress.slice(-RECENT_LINES);
    const output = [
      `任务: ${task.name}`,
      `ID: ${task.id}`,
      `类型: ${task.agentType}`,
      `状态: ${task.status}`,
      `创建: ${task.createdAt.toISOString()}`,
      task.startedAt ? `开始: ${task.startedAt.toISOString()}` : "",
      task.endedAt ? `结束: ${task.endedAt.toISOString()}` : "",
      task.exitCode !== undefined ? `退出码: ${task.exitCode}` : "",
      task.error ? `错误: ${task.error}` : "",
      task.workdir ? `工作目录: ${task.workdir}` : "",
      "",
      `--- 最近输出 (${task.progress.length} 行) ---`,
      ...(recentProgress.length > 0 ? recentProgress : ["(无输出)"]),
    ].filter(Boolean).join("\n");

    return {
      content: [{ type: "text", text: output }],
      details: { task },
    };
  },
});

const killTaskTool: ToolDef = makeTool({
  name: "kill_task",
  label: "停止任务",
  description: "强制停止一个运行中的子 agent 任务",
  parameters: {
    type: "object",
    properties: {
      taskId: {
        type: "string",
        description: "任务 ID",
      },
    },
    required: ["taskId"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { taskId: string };
  },
  async execute(_toolCallId, params) {
    const p = params as { taskId: string };
    const task = agentManager.get(p.taskId);

    if (!task) {
      return {
        content: [{ type: "text", text: `任务不存在: ${p.taskId}` }],
        details: { error: "Task not found" },
      };
    }

    agentManager.kill(p.taskId);

    return {
      content: [{ type: "text", text: `已停止任务: ${task.name} (${p.taskId})` }],
      details: { taskId: p.taskId, status: "cancelled" },
    };
  },
});

const listTaskHistoryTool: ToolDef = makeTool({
  name: "list_task_history",
  label: "查看任务历史",
  description: "查询子 agent 执行历史记录，支持按类型/状态/时间过滤。传入 taskId 可查看完整日志",
  parameters: {
    type: "object",
    properties: {
      taskId: {
        type: "string",
        description: "任务 ID（可选，传入则返回该任务的完整日志内容）",
      },
      agentType: {
        type: "string",
        description: "按 agent 类型过滤: claude-code, codex, kiro, pi-agent",
      },
      status: {
        type: "string",
        description: "按状态过滤: done, failed, cancelled",
      },
      since: {
        type: "string",
        description: "起始时间 (ISO 格式，如 2026-05-01)",
      },
      limit: {
        type: "number",
        description: "返回条数，默认 20",
      },
    },
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { taskId?: string; agentType?: string; status?: string; since?: string; limit?: number };
  },
  async execute(_toolCallId, params) {
    const p = params as { taskId?: string; agentType?: string; status?: string; since?: string; limit?: number };

    // 如果传了 taskId，直接返回该任务的完整日志
    if (p.taskId) {
      const logLines = taskHistory.readLog(p.taskId);
      const entry = taskHistory.getAll().find((e) => e.id === p.taskId);
      if (logLines.length === 0 && !entry) {
        return { content: [{ type: "text", text: `未找到任务: ${p.taskId}` }], details: {} };
      }
      const lines = [`=== 任务 ${p.taskId} ===`];
      if (entry) {
        lines.push(`名称: ${entry.name} | 类型: ${entry.agentType} | 状态: ${entry.status}`);
        lines.push(`创建: ${new Date(entry.createdAt).toLocaleString("zh-CN")}`);
        // 格式化 prompt
        const promptFormatted = entry.prompt.replace(/\n+/g, " ").trim();
        const promptDisplay = promptFormatted.length > 100
          ? promptFormatted.slice(0, 100) + "..."
          : promptFormatted;
        lines.push(`用户需求: ${promptDisplay}`);
        lines.push("");
      }
      if (logLines.length > 0) {
        lines.push(...logLines);
      } else {
        lines.push("(日志为空)");
      }
      return { content: [{ type: "text", text: lines.join("\n") }], details: { entry, logLines } };
    }

    const query: TaskHistoryQuery = {
      agentType: p.agentType,
      status: p.status,
      since: p.since,
      limit: p.limit ?? 20,
    };
    const entries = taskHistory.query(query);

    if (entries.length === 0) {
      return { content: [{ type: "text", text: "没有找到历史记录" }], details: {} };
    }

    const lines = [`共 ${entries.length} 条记录：`, ""];
    for (const e of entries) {
      const date = new Date(e.createdAt).toLocaleString("zh-CN");
      const duration = e.startedAt && e.endedAt
        ? Math.round((new Date(e.endedAt).getTime() - new Date(e.startedAt).getTime()) / 1000) + "s"
        : "-";
      const files = e.fileCount !== undefined ? ` ${e.fileCount}文件` : "";
      const error = e.status === "failed" ? ` ⚠️ ${e.error?.slice(0, 50) ?? "error"}` : "";
      lines.push(`[${e.status}] ${date} | ${e.agentType} | ${duration} | ${e.name}${files}${error}`);
      const promptSummary = e.prompt.length > 60 ? e.prompt.slice(0, 60) + "..." : e.prompt;
      lines.push(`  → ${promptSummary}`);
    }

    return { content: [{ type: "text", text: lines.join("\n") }], details: { entries } };
  },
});

const decomposeTaskTool: ToolDef = makeTool({
  name: "decompose_task",
  label: "分解复杂任务",
  description: "将一个复杂任务分解为多个可并行的子任务图，返回子任务列表。可指定 parentId 将子任务挂载到已有父任务下",
  parameters: {
    type: "object",
    properties: {
      taskDescription: {
        type: "string",
        description: "需要分解的复杂任务描述",
      },
      parentId: {
        type: "string",
        description: "父任务 ID（可选，用于将子任务挂载到已有任务下）",
      },
      subtasks: {
        type: "array",
        description: "子任务列表，每个包含 title、agentType、prompt",
        items: {
          type: "object",
          properties: {
            title: { type: "string", description: "子任务标题" },
            agentType: {
              type: "string",
              description: "Agent 类型: claude-code, pi-agent, codex, kiro, custom",
              enum: ["claude-code", "pi-agent", "codex", "kiro", "custom"],
            },
            prompt: { type: "string", description: "给子 agent 的任务描述" },
          },
          required: ["title", "agentType", "prompt"],
        },
      },
    },
    required: ["taskDescription", "subtasks"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as {
      taskDescription: string;
      parentId?: string;
      subtasks: Array<{ title: string; agentType: AgentType; prompt: string }>;
    };
  },
  async execute(_toolCallId, params) {
    const p = params as {
      taskDescription: string;
      parentId?: string;
      subtasks: Array<{ title: string; agentType: AgentType; prompt: string }>;
    };

    if (!p.subtasks || p.subtasks.length === 0) {
      return {
        content: [{ type: "text", text: "错误: subtasks 不能为空，请提供至少一个子任务" }],
        details: { error: "Empty subtasks" },
      };
    }

    // If parentId specified, validate it exists
    if (p.parentId && !agentManager.get(p.parentId)) {
      return {
        content: [{ type: "text", text: `错误: 父任务不存在: ${p.parentId}` }],
        details: { error: "Parent task not found" },
      };
    }

    const spawned = [];
    const lines = [`任务分解: "${p.taskDescription.slice(0, 80)}${p.taskDescription.length > 80 ? "..." : ""}"`, ""];

    for (const sub of p.subtasks) {
      const opts: { name?: string; workdir?: string; parentId?: string } = { name: sub.title };
      if (p.parentId) opts.parentId = p.parentId;

      const task = p.parentId
        ? agentManager.spawnChild(p.parentId, sub.agentType, sub.prompt, opts)
        : agentManager.spawn(sub.agentType, sub.prompt, opts);

      spawned.push({ id: task.id, name: task.name, agentType: sub.agentType, status: task.status });
      lines.push(`  ✓ [${sub.agentType}] ${sub.title} → ${task.id}`);
    }

    lines.push("", `已启动 ${spawned.length} 个子任务，使用 list_tasks 查看进度`);

    return {
      content: [{ type: "text", text: lines.join("\n") }],
      details: { taskDescription: p.taskDescription, parentId: p.parentId, subtasks: spawned },
    };
  },
});

export function registerAgentManagerTools(): void {
  registry.register(spawnAgentTool);
  registry.register(listTasksTool);
  registry.register(getTaskTool);
  registry.register(killTaskTool);
  registry.register(listTaskHistoryTool);
  registry.register(decomposeTaskTool);
}
