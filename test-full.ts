import { agentManager } from "./src/tasks/agent-manager.js";
import * as fs from "fs";

const logFile = "/tmp/todo-test.log";
const log = (msg: string) => {
  const line = `${new Date().toISOString()} ${msg}\n`;
  fs.appendFileSync(logFile, line);
  console.error(line);
};

const task = agentManager.spawn("claude-code",
  `实现一个 TODO 应用。

功能要求：
1. 可以添加新的 TODO 事项
2. 可以标记 TODO 为已完成
3. 可以删除 TODO
4. 可以查看所有 TODO 列表
5. 已完成的和未完成的分开显示

技术要求：
- 使用现代化的前端框架（React/Vue/Svelte 等）
- 实现本地数据持久化（localStorage）
- 响应式设计，支持移动端

请创建一个完整的、可运行的项目，包含代码和必要的配置文件。
完成后总结项目结构和运行方法。`,
  { name: "todo-app", workdir: "/tmp/todo-app" }
);

log(`Task spawned: ${task.id}`);

const seen = new Set<string>();
let lastStatus = "";

const timer = setInterval(() => {
  const t = agentManager.get(task.id);
  if (!t) {
    log("Task no longer in registry");
    clearInterval(timer);
    return;
  }

  const key = `${t.status}|${t.progress.length}|${t.exitCode ?? ""}`;
  if (!seen.has(key)) {
    seen.add(key);
    const last = t.progress.length > 0 ? t.progress[t.progress.length - 1].slice(0, 300) : "(none)";
    log(`Status: ${t.status} | lines: ${t.progress.length} | exitCode: ${t.exitCode ?? "null"} | last: ${last}`);
  }

  if (t.status === "done" || t.status === "failed") {
    clearInterval(timer);
    log(`FINAL STATUS: ${t.status}`);
    if (t.error) log(`ERROR: ${t.error}`);
    log(`Progress lines total: ${t.progress.length}`);
    for (const p of t.progress.slice(-20)) {
      log(`  ${p.slice(0, 300)}`);
    }
    process.exit(0);
  }
}, 2000);
