import "dotenv/config";
import * as readline from "readline";
import { createOrchestratorAgent, subscribeToOrchestrator } from "./orchestrator.js";
import { agentManager } from "./tasks/agent-manager.js";

async function main() {
  console.log("=".repeat(50));
  console.log("  Pets Agent - Agent Orchestration Platform");
  console.log("=".repeat(50));
  console.log();
  console.log("初始化 Orchestrator Agent...");

  const agent = createOrchestratorAgent();
  subscribeToOrchestrator(agent);

  // Subscribe to task updates for real-time display
  agentManager.on("update", (update) => {
    if (update.status === "running" && update.progress.length > 0) {
      const latest = update.progress[update.progress.length - 1];
      console.log(`\n[任务 ${update.id.slice(0, 8)}] ${latest}`);
    }
  });

  agentManager.on("exit", ({ taskId, exitCode }) => {
    console.log(`\n[任务 ${taskId.slice(0, 8)}] 已结束，退出码: ${exitCode}`);
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const userPrompt = await new Promise<string>((resolve) =>
    rl.question("\n请输入您的请求: ", resolve),
  );
  rl.close();

  console.log(`\n用户: ${userPrompt}`);
  console.log("=".repeat(50));

  await agent.prompt(userPrompt);
  await agent.waitForIdle();

  console.log("\n" + "=".repeat(50));
  console.log("Agent 处理完成");
  console.log("=".repeat(50));

  // Cleanup
  agentManager.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error("错误:", err);
  agentManager.destroy();
  process.exit(1);
});
