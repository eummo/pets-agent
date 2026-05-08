import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import { randomBytes } from "crypto";
import * as fs from "fs";
import type { Task, TaskStatus, AgentType } from "./task.js";
import { taskHistory } from "./task-history.js";
import { spawnDeepAgents, abortDeepAgents } from "./deepagents-runtime.js";

function generateId(): string {
  return randomBytes(8).toString("hex");
}

export interface RunningTask {
  task: Task;
  child: ChildProcess;
  stdoutBuffer: string;
  stderrBuffer: string;
}

export type TaskUpdate = Pick<Task, "id" | "status" | "progress" | "error" | "exitCode" | "startedAt" | "endedAt">;

class AgentManager extends EventEmitter {
  private tasks = new Map<string, Task>();
  private running = new Map<string, RunningTask>();
  private reapInterval: ReturnType<typeof setInterval> | null = null;
  private subscriptions = new Map<string, Set<(update: TaskUpdate) => void>>();

  constructor() {
    super();
    this.startReaper();
  }

  private startReaper(): void {
    this.reapInterval = setInterval(() => {
      for (const [taskId, rt] of this.running) {
        if (rt.child.exitCode !== null) {
          this.handleProcessExit(taskId);
        }
      }
    }, 5000);
  }

  private handleProcessExit(taskId: string): void {
    const rt = this.running.get(taskId);
    if (!rt) return;

    const { task, child } = rt;
    const hadError = child.exitCode !== 0;

    task.status = hadError ? "failed" : "done";
    task.endedAt = new Date();
    task.exitCode = child.exitCode ?? undefined;

    if (hadError && task.error) {
      task.error = task.error.trim();
    } else if (hadError) {
      task.error = `Process exited with code ${child.exitCode}`;
    }

    this.running.delete(taskId);
    this.emit("update", this.broadcastUpdate(task));
    this.emit("exit", { taskId, exitCode: child.exitCode });
    taskHistory.add(task);
  }

  private broadcastUpdate(task: Task): TaskUpdate {
    return {
      id: task.id,
      status: task.status,
      progress: task.progress.length > 0 ? [...task.progress] : undefined,
      error: task.error,
      exitCode: task.exitCode,
      startedAt: task.startedAt,
      endedAt: task.endedAt,
    };
  }

  private notifySubscribers(taskId: string, update: TaskUpdate): void {
    const subs = this.subscriptions.get(taskId);
    if (subs) {
      for (const cb of subs) {
        try {
          cb(update);
        } catch (err) {
          console.error(`[AgentManager] Subscriber error for task ${taskId.slice(0, 8)}:`, err);
        }
      }
    }
  }

  private emitUpdate(task: Task): void {
    const update = this.broadcastUpdate(task);
    this.emit("update", update);
    this.notifySubscribers(task.id, update);
  }

  private buildCommand(agentType: AgentType): { cmd: string; args: string[]; passPromptViaStdin: boolean } {
    switch (agentType) {
      case "claude-code": {
        // In WSL, the "claude" bash wrapper calls PowerShell which corrupts args with
        // newlines/spaces. Call the Windows exe directly via the full NT path.
        // -p = print mode (non-interactive, exits after completion)
        // positional arg = prompt
        // --dangerously-skip-permissions = skip permission prompts
        // --no-session-persistence = don't save session to disk
        // --bare = minimal mode
        const claudeExe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";
        return {
          cmd: claudeExe,
          args: ["-p", "--dangerously-skip-permissions", "--no-session-persistence", "--bare"],
          passPromptViaStdin: false,
        };
      }
      case "codex":
        return { cmd: "npx", args: ["-y", "openai/codex", "--acp", "--stdio"], passPromptViaStdin: true };
      case "kiro":
        return { cmd: "kiro", args: ["--acp", "--stdio"], passPromptViaStdin: true };
      default:
        return { cmd: agentType, args: ["--acp", "--stdio"], passPromptViaStdin: true };
    }
  }

  private parseAcpMessage(line: string): { type: string; taskId?: string; [key: string]: unknown } | null {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }

  spawn(agentType: AgentType, prompt: string, opts?: { name?: string; workdir?: string }): Task {
    const id = generateId();
    const task: Task = {
      id,
      name: opts?.name ?? `${agentType}-${id}`,
      agentType,
      prompt,
      status: "pending",
      createdAt: new Date(),
      progress: [],
      workdir: opts?.workdir,
    };

    this.tasks.set(id, task);

    // ── deepagents: runs in-process via LangGraph (no child process) ──────────
    if (agentType === "deepagents") {
      task.status = "pending";
      this.emitUpdate(task);

      // Fire-and-forget async — spawnDeepAgents streams into task.progress
      // and calls taskHistory.add() when done.
      spawnDeepAgents(task, prompt, opts?.workdir).catch((err) => {
        task.error = err instanceof Error ? err.message : String(err);
        task.status = "failed";
        task.endedAt = new Date();
        this.emitUpdate(task);
        taskHistory.add(task);
      });

      return task;
    }

    // ── External agents: spawn as child process ────────────────────────────────
    const { cmd, args, passPromptViaStdin } = this.buildCommand(agentType);

    // Filter WSL-only environment variables so child processes don't inherit them
    const filteredEnv: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (k.startsWith("HERMES_") || k === "WSL_DISTRO_NAME" || k === "WSLENV") continue;
      if (v !== undefined) filteredEnv[k] = v;
    }

    // Ensure workdir exists before spawning
    const workdir = opts?.workdir ?? process.cwd();
    if (workdir && !fs.existsSync(workdir)) {
      fs.mkdirSync(workdir, { recursive: true });
    }

    const spawnArgs = passPromptViaStdin ? args : [...args, prompt];
    const spawnOpts: Parameters<typeof spawn>[2] = {
      cwd: workdir,
      env: filteredEnv,
    };

    if (passPromptViaStdin) {
      spawnOpts.stdio = ["pipe", "pipe", "pipe", "ipc"];
    } else {
      // Prompt is passed as positional arg — stdin not needed.
      // Use "ignore" so the child doesn't wait for input.
      spawnOpts.stdio = ["ignore", "pipe", "pipe", "ipc"];
    }

    const child = spawn(cmd, spawnArgs, spawnOpts);

    const rt: RunningTask = {
      task,
      child,
      stdoutBuffer: "",
      stderrBuffer: "",
    };

    this.running.set(id, rt);

    child.stdout?.on("data", (data: Buffer) => {
      const chunk = data.toString();
      const lines = (rt.stdoutBuffer + chunk).split("\n");
      rt.stdoutBuffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.trim()) {
          task.progress.push(line);
          taskHistory.appendLog(task.id, [line]);
          this.emitUpdate(task);
        }
      }
    });

    child.stderr?.on("data", (data: Buffer) => {
      const chunk = data.toString();
      const lines = (rt.stderrBuffer + chunk).split("\n");
      rt.stderrBuffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.trim()) {
          task.progress.push(`[stderr] ${line}`);
          taskHistory.appendLog(task.id, [`[stderr] ${line}`]);
          this.emitUpdate(task);
        }
      }
    });

    child.on("error", (err) => {
      task.status = "failed";
      task.error = err.message;
      task.endedAt = new Date();
      this.running.delete(id);
      this.emitUpdate(task);
      // Record failure in history
      taskHistory.add(task);
    });

    child.on("close", (code) => {
      if (rt.stdoutBuffer.trim()) {
        task.progress.push(rt.stdoutBuffer);
        taskHistory.appendLog(task.id, [rt.stdoutBuffer]);
        this.emitUpdate(task);
      }
      if (this.running.has(id)) {
        task.exitCode = code ?? undefined;
        this.handleProcessExit(id);
      }
    });

    if (passPromptViaStdin) {
      // Simple mode: write prompt to stdin and wait for process to finish
      task.status = "running";
      task.startedAt = new Date();
      this.emitUpdate(task);

      setTimeout(() => {
        if (rt.child.stdin) {
          rt.child.stdin.write(prompt + "\n");
          rt.child.stdin.end();
        }
      }, 500);
    } else {
      // Positional arg mode (e.g. claude -p <prompt>): just start and stream output
      task.status = "running";
      task.startedAt = new Date();
      this.emitUpdate(task);
    }

    // IPC channel for control messages
    child.on("message", (msg: unknown) => {
      try {
        const m = msg as { type?: string; taskId?: string };
        if (m.type === "ping") {
          child.send?.({ type: "pong", taskId: m.taskId });
        }
      } catch (err) {
        console.warn(`[AgentManager] Failed to handle IPC message: ${err}`);
      }
    });

    return task;
  }

  private sendPrompt(taskId: string, prompt: string): void {
    const rt = this.running.get(taskId);
    if (!rt) return;

    const msg = {
      type: "user_message",
      taskId,
      content: prompt,
    };

    try {
      rt.child.stdin?.write(JSON.stringify(msg) + "\n");
    } catch (err) {
      const task = this.tasks.get(taskId);
      if (task) {
        task.error = `Failed to send prompt: ${err}`;
      }
    }
  }

  kill(taskId: string): void {
    const task = this.tasks.get(taskId);

    // deepagents tasks are in-process — use AbortController
    if (task?.agentType === "deepagents") {
      abortDeepAgents(taskId);
      task.status = "cancelled";
      task.endedAt = new Date();
      this.emitUpdate(task);
      this.emit("kill", { taskId });
      return;
    }

    const rt = this.running.get(taskId);
    if (!rt) return;

    if (task) {
      task.status = "cancelled";
      task.endedAt = new Date();
    }

    try {
      rt.child.kill("SIGTERM");
    } catch (err) {
      console.warn(`[AgentManager] Failed to kill task ${taskId.slice(0, 8)}:`, err);
    }

    this.running.delete(taskId);
    if (task) this.emitUpdate(task);
    this.emit("kill", { taskId });
  }

  list(): Task[] {
    return Array.from(this.tasks.values());
  }

  get(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  subscribe(taskId: string, callback: (update: TaskUpdate) => void): () => void {
    if (!this.subscriptions.has(taskId)) {
      this.subscriptions.set(taskId, new Set());
    }
    this.subscriptions.get(taskId)!.add(callback);

    const task = this.tasks.get(taskId);
    if (task) {
      callback(this.broadcastUpdate(task));
    }

    return () => {
      const subs = this.subscriptions.get(taskId);
      if (subs) {
        subs.delete(callback);
        if (subs.size === 0) {
          this.subscriptions.delete(taskId);
        }
      }
    };
  }

  getActiveTasks(): Task[] {
    return this.list().filter((t) => t.status === "running" || t.status === "pending");
  }

  destroy(): void {
    if (this.reapInterval) {
      clearInterval(this.reapInterval);
      this.reapInterval = null;
    }
    for (const [taskId] of this.running) {
      this.kill(taskId);
    }
    taskHistory.destroy();
    this.removeAllListeners();
  }
}

// Singleton instance
export const agentManager = new AgentManager();
export { AgentManager };
