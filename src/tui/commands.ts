/**
 * Slash command handler - processes /commands from user input
 */

import { truncateToWidth } from "@earendil-works/pi-tui";
import { agentManager } from "../tasks/agent-manager.js";
import { taskHistory } from "../tasks/task-history.js";
import type { ChatLog } from "./components.js";
import type { TUI } from "@earendil-works/pi-tui";
import type { Agent } from "@earendil-works/pi-agent-core";

export type AgentFactory = () => ReturnType<typeof import("../orchestrator.js").createOrchestratorAgent>;

export async function handleSlashCommand(
  cmd: string,
  chatLog: ChatLog,
  tui: TUI,
  createAgent: AgentFactory
): Promise<void> {
  tui.requestRender();

  switch (cmd) {
    case "/quit":
    case "/exit": {
      tui.stop();
      agentManager.destroy();
      process.exit(0);
      break;
    }

    case "/clear": {
      chatLog.clear();
      tui.requestRender();
      break;
    }

    case "/tasks": {
      const tasks = agentManager.list();
      if (tasks.length === 0) {
        chatLog.pushTool("No tasks running.");
      } else {
        for (const t of tasks) {
          const id = t.id.slice(0, 8);
          const status = t.status === "running" ? "[running]" : t.status === "failed" ? "[failed]" : "[done]";
          chatLog.pushTool(`${status} [${id}] ${t.name}`);
        }
      }
      tui.requestRender();
      break;
    }

    case "/history": {
      const entries = taskHistory.getAll();
      if (entries.length === 0) {
        chatLog.pushTool("No history.");
      } else {
        for (const e of entries.slice(-10)) {
          const date = new Date(e.createdAt).toLocaleString("zh-CN");
          const duration = e.startedAt && e.endedAt
            ? Math.round((new Date(e.endedAt).getTime() - new Date(e.startedAt).getTime()) / 1000) + "s"
            : "-";
          const promptSingleLine = e.prompt.replace(/\n+/g, " ").trim();
          const promptSummary = promptSingleLine.length > 60 ? promptSingleLine.slice(0, 60) + "..." : promptSingleLine;
          const statusLine = `[${e.status}] ${date} | ${e.agentType} | ${duration}`;
          chatLog.pushTool(truncateToWidth(statusLine, 80, ""));
          chatLog.pushTool(truncateToWidth(`  → ${promptSummary}`, 80, ""));
        }
      }
      tui.requestRender();
      break;
    }

    case "/logs": {
      const running = agentManager.getActiveTasks();
      if (running.length === 0) {
        chatLog.pushTool("No running tasks.");
      } else {
        for (const t of running) {
          chatLog.pushTool(`--- ${t.name} [${t.id.slice(0, 8)}] ---`);
          if (t.progress.length > 0) {
            t.progress.slice(-10).forEach((l: string) => chatLog.pushTool("  " + l));
          } else {
            chatLog.pushTool("  (no output yet)");
          }
        }
      }
      tui.requestRender();
      break;
    }

    case "/help": {
      const { buildHelpText } = await import("./components.js");
      chatLog.pushAgent(buildHelpText());
      tui.requestRender();
      break;
    }

    default: {
      if (cmd.startsWith("/logs ")) {
        const taskId = cmd.slice(6).trim();
        const t = agentManager.get(taskId);
        if (!t) {
          chatLog.pushTool(`Task not found: ${taskId}`);
        } else {
          chatLog.pushTool(`--- ${t.name} [${t.id.slice(0, 8)}] ---`);
          t.progress.forEach((l: string) => chatLog.pushTool("  " + l));
        }
        tui.requestRender();
        break;
      }
      if (cmd.startsWith("/history ")) {
        const taskId = cmd.slice(9).trim();
        const logLines = taskHistory.readLog(taskId);
        if (logLines.length === 0) {
          chatLog.pushTool(`No logs for task: ${taskId}`);
        } else {
          logLines.forEach((l: string) => chatLog.pushTool("  " + l));
        }
        tui.requestRender();
        break;
      }
      chatLog.pushTool(`Unknown command: ${cmd}`);
      tui.requestRender();
      break;
    }
  }
}
