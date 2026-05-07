import { agentManager } from "./src/tasks/agent-manager.js";
import * as fs from "fs";

const logFile = "/tmp/todo-full.log";
const log = (m: string) => {
  const line = new Date().toISOString() + " " + m;
  fs.appendFileSync(logFile, line + "\n");
  console.error(line);
};

// Clean workdir
const workdir = "/tmp/todo-app-full";
try { fs.rmSync(workdir, { recursive: true }); } catch {}
try { fs.mkdirSync(workdir, { recursive: true }); } catch {}

const prompt = `实现一个 TODO 应用。

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
完成后总结项目结构和运行方法。`;

const t = agentManager.spawn("claude-code", prompt, { name: "todo-app", workdir });
log("spawned: " + t.id);

const seen = new Set<string>();
const timer = setInterval(() => {
  const x = agentManager.get(t.id);
  if (!x) { log("NOT FOUND in registry"); clearInterval(timer); return; }

  const key = x.status + "|" + x.progress.length;
  if (!seen.has(key)) {
    seen.add(key);
    const last = x.progress.length > 0 ? x.progress[x.progress.length - 1].slice(0, 300) : "(none)";
    log("status=" + x.status + " lines=" + x.progress.length + " last: " + last);
  }

  if (x.status === "done" || x.status === "failed") {
    clearInterval(timer);
    log("FINAL: " + x.status + " exitCode=" + x.exitCode);
    if (x.error) log("ERROR: " + x.error);
    // List created files
    try {
      const files: string[] = [];
      const walk = (d: string) => {
        for (const e of fs.readdirSync(d)) {
          const p = d + "/" + e;
          const s = fs.statSync(p);
          if (s.isDirectory()) walk(p);
          else files.push(p);
        }
      };
      walk(workdir);
      log("Created " + files.length + " files:");
      for (const f of files) log("  " + f.replace(workdir, ""));
    } catch (e: any) { log("walk error: " + e.message); }
    process.exit(0);
  }
}, 3000);
