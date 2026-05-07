import "dotenv/config";
import * as readline from "readline";
import { createOrchestratorAgent, subscribeToOrchestrator } from "./orchestrator.js";
import { agentManager } from "./tasks/agent-manager.js";
import { taskHistory } from "./tasks/task-history.js";
import type { Task } from "./tasks/task.js";

// Ensure history is persisted on unexpected exit
process.on("exit", () => taskHistory.flush());
process.on("SIGTERM", () => {
  taskHistory.flush();
  agentManager.destroy();
});

const SLASH_COMMANDS = ["/quit", "/exit", "/clear", "/logs", "/tasks", "/history", "/help"];

function createRL(): readline.Interface {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    completer: (line: string): [string[], string] => {
      const hits = SLASH_COMMANDS.filter((c) => c.startsWith(line));
      return [hits, line];
    },
  });
}

function printBanner() {
  console.log("=".repeat(50));
  console.log("  Pets Agent - Agent Orchestration Platform");
  console.log("=".repeat(50));
  console.log("命令: /help 查看帮助  /quit 退出");
  console.log();
}

function printHelp() {
  console.log(`
可用命令:
  <任意文字>     - 发送消息给 agent
  /help          - 显示此帮助
  /quit, /exit   - 退出程序
  /clear         - 清屏
  /tasks         - 列出所有任务（当前会话）
  /history       - 查看任务执行历史
  /logs [taskId] - 实时查看任务输出（省略 taskId 查看所有运行中任务）
`);
}

function formatTask(t: Task): string {
  const id = t.id.slice(0, 8);
  const running = t.progress.length > 0 ? ` (${t.progress.length} 行输出)` : "";
  return `  [${id}] ${t.name} - ${t.status}${running}`;
}

async function runREPL() {
  printBanner();
  console.log("初始化 Orchestrator Agent...\n");

  const agent = createOrchestratorAgent();
  subscribeToOrchestrator(agent);

  // Forward agent manager events to console (use write to avoid REPL buffering)
  agentManager.on("update", (update) => {
    if (update.progress.length > 0) {
      const latest = update.progress[update.progress.length - 1];
      process.stdout.write(`\n[任务 ${update.id.slice(0, 8)}] ${latest}\n`);
    }
  });

  agentManager.on("exit", ({ taskId, exitCode }) => {
    process.stdout.write(`\n[任务 ${taskId.slice(0, 8)}] 已结束，退出码: ${exitCode}\n`);
  });

  // --- Task watching state ---
  let watchingTaskId: string | null = null;
  let watchInterval: ReturnType<typeof setInterval> | null = null;

  function stopWatching() {
    if (watchInterval) {
      clearInterval(watchInterval);
      watchInterval = null;
      watchingTaskId = null;
    }
  }

  function startWatching(taskId: string) {
    stopWatching();
    watchingTaskId = taskId;
    console.log(`\n实时监控任务 ${taskId.slice(0, 8)}... (输入任意内容退出监控)`);

    const unsubscribe = agentManager.subscribe(taskId, (update) => {
      if (update.progress.length > 0) {
        const newLines = update.progress.slice(-10);
        newLines.forEach((line) => console.log(`  ${line}`));
      }
      if (update.status === "done" || update.status === "failed" || update.status === "cancelled") {
        console.log(`\n任务已结束: ${update.status}`);
        stopWatching();
        // unsubscribe is called by stopWatching via the interval pattern
      }
    });

    // Replace unsubscribe to also stop interval
    const originalUnsub = unsubscribe;
    const newUnsub = () => {
      stopWatching();
      originalUnsub();
    };

    watchInterval = setInterval(() => {
      const t = agentManager.get(taskId);
      if (!t || t.status === "done" || t.status === "failed" || t.status === "cancelled") {
        stopWatching();
      }
    }, 2000);
  }

  function printTasks() {
    const tasks = agentManager.list();
    if (tasks.length === 0) {
      console.log("\n暂无任务");
      return;
    }
    console.log("\n任务列表:");
    for (const t of tasks) {
      console.log(formatTask(t));
    }
  }

  function printRunningOutputs() {
    const running = agentManager.getActiveTasks();
    if (running.length === 0) {
      console.log("\n没有运行中的任务");
      return;
    }
    console.log("\n运行中任务:");
    for (const t of running) {
      console.log(`\n--- ${t.name} [${t.id.slice(0, 8)}] ---`);
      if (t.progress.length > 0) {
        t.progress.slice(-10).forEach((line) => console.log(`  ${line}`));
      } else {
        console.log("  (暂无输出)");
      }
    }
  }

  async function sendToAgent(input: string) {
    process.stdout.write(`\n用户: ${input}\n`);
    process.stdout.write(`${"=".repeat(50)}\n`);
    await agent.prompt(input);
    await agent.waitForIdle();
    process.stdout.write(`${"=".repeat(50)}\n`);
  }

  // --- REPL loop ---
  const rl = createRL();
  let agentBusy = false;

  const promptUser = () => {
    rl.question("\n请输入您的请求: ", async (input) => {
      const trimmed = input.trim();

      if (!trimmed) {
        promptUser();
        return;
      }

      // If watching, any input stops watching and sends to agent
      if (watchingTaskId) {
        stopWatching();
        console.log("(已退出监控模式)");
      }

      if (trimmed === "/quit" || trimmed === "/exit") {
        rl.close();
        agentManager.destroy();
        console.log("再见!");
        process.exit(0);
        return;
      }

      if (trimmed === "/clear") {
        process.stdout.write("\x1b[2J\x1b[H"); // ANSI clear screen
        printBanner();
        promptUser();
        return;
      }

      if (trimmed === "/help") {
        printHelp();
        promptUser();
        return;
      }

      if (trimmed === "/tasks") {
        printTasks();
        promptUser();
        return;
      }

      if (trimmed === "/history") {
        const parts = trimmed.split(" ");
        const taskId = parts[1]?.trim();

        if (taskId) {
          // 查看指定任务的完整日志
          const logLines = taskHistory.readLog(taskId);
          const entry = taskHistory.getAll().find((e) => e.id === taskId);
          if (logLines.length === 0 && !entry) {
            console.log(`\n未找到任务: ${taskId}`);
          } else {
            console.log(`\n=== 任务 ${taskId} 日志 ===`);
            if (entry) {
              const date = new Date(entry.createdAt).toLocaleString("zh-CN");
              console.log(`名称: ${entry.name} | 类型: ${entry.agentType} | 状态: ${entry.status}`);
              console.log(`创建: ${date}`);
              console.log(`提示: ${entry.prompt}`);
              console.log();
            }
            if (logLines.length > 0) {
              logLines.forEach((line) => console.log(line));
            } else {
              console.log("(日志文件为空或不存在)");
            }
          }
        } else {
          // 列出所有历史
          const entries = taskHistory.getAll();
          if (entries.length === 0) {
            console.log("\n暂无历史记录");
          } else {
            console.log(`\n历史记录 (共 ${entries.length} 条):`);
            for (const e of entries.slice(0, 20)) {
              const date = new Date(e.createdAt).toLocaleString("zh-CN");
              const duration = e.startedAt && e.endedAt
                ? Math.round((new Date(e.endedAt).getTime() - new Date(e.startedAt).getTime()) / 1000) + "s"
                : "-";
              const files = e.fileCount !== undefined ? ` ${e.fileCount}文件` : "";
              const error = e.status === "failed" ? ` ⚠️` : "";
              // 合并换行符，限制总长度
              const promptSingleLine = e.prompt.replace(/\n+/g, " ").trim();
              const promptSummary = promptSingleLine.length > 70 ? promptSingleLine.slice(0, 70) + "..." : promptSingleLine;
              console.log(`[${e.status}] ${date} | ${e.agentType} | ${duration} | ${e.name}${files}${error}`);
              console.log(`  → ${promptSummary}`);
            }
            if (entries.length > 20) console.log(`\n(还有 ${entries.length - 20} 条记录)`);
          }
        }
        promptUser();
        return;
      }

      if (trimmed.startsWith("/logs")) {
        const parts = trimmed.split(" ");
        if (parts[1]?.trim()) {
          await startWatching(parts[1].trim());
        } else {
          printRunningOutputs();
        }
        promptUser();
        return;
      }

      if (agentBusy) {
        console.log("Agent 正忙，请稍候...");
        promptUser();
        return;
      }

      agentBusy = true;
      rl.pause();
      try {
        await sendToAgent(trimmed);
      } finally {
        agentBusy = false;
        rl.resume();
        promptUser();
      }
    });
  };

  process.on("SIGINT", () => {
    taskHistory.flush();
    if (watchingTaskId) {
      stopWatching();
      console.log("\n(监控已退出)");
      promptUser();
    } else {
      console.log("\n(使用 /quit 退出)");
      promptUser();
    }
  });

  promptUser();
}

runREPL().catch((err) => {
  console.error("错误:", err);
  agentManager.destroy();
  process.exit(1);
});
