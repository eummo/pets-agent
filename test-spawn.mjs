import { agentManager } from "./src/tasks/agent-manager.js";
import { randomBytes } from "crypto";

const taskId = randomBytes(4).toString("hex");

console.log(`Spawning test task: ${taskId}`);

const task = agentManager.spawn("claude-code", "用中文回复，只说四个字：运行成功", {
  name: "test-task",
  workdir: "/tmp",
});

console.log("Task:", JSON.stringify(task, null, 2));

agentManager.subscribe(task.id, (update) => {
  console.log(`[${update.status}] progress lines: ${update.progress?.length ?? 0}`);
  if (update.progress?.length) {
    console.log("  last:", update.progress[update.progress.length - 1].slice(0, 200));
  }
  if (update.status === "done" || update.status === "failed") {
    console.log("Final:", JSON.stringify(update, null, 2));
    process.exit(0);
  }
});

setTimeout(() => {
  console.log("Timeout — killing task");
  agentManager.kill(task.id);
  process.exit(1);
}, 30000);
