import { registry } from "./registry.js";
import { agentManager } from "../tasks/agent-manager.js";
import type { AgentType } from "../tasks/task.js";
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
        description: "Agent 类型: claude-code, codex, kiro, pi-agent, custom",
        enum: ["claude-code", "codex", "kiro", "pi-agent", "custom"],
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

export function registerAgentManagerTools(): void {
  registry.register(spawnAgentTool);
  registry.register(listTasksTool);
  registry.register(getTaskTool);
  registry.register(killTaskTool);
}
