import "dotenv/config";
import * as readline from "readline";
import { createOrchestratorAgent, subscribeToOrchestrator } from "./orchestrator.js";
import { agentManager } from "./tasks/agent-manager.js";
import type { Task } from "./tasks/task.js";

const SLASH_COMMANDS = ["/quit", "/exit", "/clear", "/logs", "/tasks", "/help"];

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
  /tasks         - 列出所有任务
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

  // Forward agent manager events to console
  agentManager.on("update", (update) => {
    if (update.progress.length > 0) {
      const latest = update.progress[update.progress.length - 1];
      console.log(`\n[任务 ${update.id.slice(0, 8)}] ${latest}`);
    }
  });

  agentManager.on("exit", ({ taskId, exitCode }) => {
    console.log(`\n[任务 ${taskId.slice(0, 8)}] 已结束，退出码: ${exitCode}`);
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
    console.log(`\n用户: ${input}`);
    console.log("=".repeat(50));
    await agent.prompt(input);
    await agent.waitForIdle();
    console.log("=".repeat(50));
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
        console.clear();
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
