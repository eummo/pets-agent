import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import { randomBytes } from "crypto";
import * as fs from "fs";
import * as path from "path";
import type { Task, TaskStatus, AgentType } from "./task.js";
import { taskHistory } from "./task-history.js";

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
  private children = new Map<string, string[]>();
  private reapInterval: ReturnType<typeof setInterval> | null = null;
  private subscriptions = new Map<string, Set<(update: TaskUpdate) => void>>();

  constructor() {
    super();
    this.children = new Map();
    this.startReaper();
  }

  private startReaper(): void {
    this.reapInterval = setInterval(() => {
      for (const [taskId, rt] of this.running) {
        if (rt.child.exitCode !== null) {
          this.handleProcessExit(taskId);
        }
      }
    }, 2000);
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

    // Flush remaining buffer
    if (rt.stdoutBuffer.trim()) {
      task.progress.push(rt.stdoutBuffer);
      taskHistory.appendLog(task.id, [rt.stdoutBuffer]);
    }
    if (rt.stderrBuffer.trim()) {
      task.progress.push(`[stderr] ${rt.stderrBuffer}`);
      taskHistory.appendLog(task.id, [`[stderr] ${rt.stderrBuffer}`]);
    }

    this.running.delete(taskId);
    this.emit("update", this.broadcastUpdate(task));
    this.emit("exit", { taskId, exitCode: child.exitCode });
    taskHistory.add(task);
    this.checkChildrenDone(taskId);
  }

  private broadcastUpdate(task: Task): TaskUpdate {
    return {
      id: task.id,
      status: task.status,
      progress: task.progress.length > 0 ? task.progress.slice() : [],
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

  private buildCommand(agentType: AgentType, workdir?: string): { cmd: string; args: string[]; passPromptViaStdin: boolean; name: string } {
    switch (agentType) {
      case "claude-code": {
        // Try npm global bin first, then PATH
        const npmGlobalBin = process.env.npm_config_global_prefix
          ? path.join(process.env.npm_config_global_prefix, "bin", "claude")
          : null;
        const candidates = [
          npmGlobalBin,
          "claude",
          "/usr/local/bin/claude",
          "/usr/bin/claude",
        ].filter(Boolean) as string[];
        return {
          cmd: candidates[0],
          args: ["-p", "--dangerously-skip-permissions", "--no-session-persistence", "--bare"],
          passPromptViaStdin: false,
          name: "claude-code",
        };
      }
      case "pi-agent": {
        // pi coding-agent via npx - runs in print mode with JSON output
        // workdir is passed via --input option or we use cwd
        return {
          cmd: "npx",
          args: ["-y", "@earendil-works/pi-coding-agent", "-p", "--no-input"],
          passPromptViaStdin: false,
          name: "pi-agent",
        };
      }
      case "codex":
        return { cmd: "npx", args: ["-y", "openai/codex", "--acp", "--stdio"], passPromptViaStdin: true, name: "codex" };
      case "kiro":
        return { cmd: "kiro", args: ["--acp", "--stdio"], passPromptViaStdin: true, name: "kiro" };
      default:
        return { cmd: agentType, args: ["--acp", "--stdio"], passPromptViaStdin: true, name: agentType };
    }
  }

  spawn(agentType: AgentType, prompt: string, opts?: { name?: string; workdir?: string; parentId?: string }): Task {
    const id = generateId();
    const task: Task = {
      id,
      name: opts?.name ?? `${agentType}-${id.slice(0, 6)}`,
      agentType,
      prompt,
      status: "pending",
      createdAt: new Date(),
      progress: [],
      workdir: opts?.workdir,
      parentId: opts?.parentId,
    };

    // Establish parent-child relationship
    if (opts?.parentId) {
      const parent = this.tasks.get(opts.parentId);
      if (parent) {
        if (!parent.children) parent.children = [];
        parent.children.push(id);
      }
    }

    this.tasks.set(id, task);

    const { cmd, args, passPromptViaStdin, name } = this.buildCommand(agentType, opts?.workdir);

    // Filter WSL-only environment variables
    const filteredEnv: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (k.startsWith("HERMES_") || k === "WSL_DISTRO_NAME" || k === "WSLENV") continue;
      if (v !== undefined) filteredEnv[k] = v;
    }

    // Ensure workdir exists
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
      // Prompt is passed as positional arg — stdin not needed
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

    // Stream stdout line by line for real-time progress
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

    // Stream stderr with prefix
    child.stderr?.on("data", (data: Buffer) => {
      const chunk = data.toString();
      const lines = (rt.stderrBuffer + chunk).split("\n");
      rt.stderrBuffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.trim()) {
          task.progress.push(`[${name} stderr] ${line}`);
          taskHistory.appendLog(task.id, [`[${name} stderr] ${line}`]);
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
      taskHistory.add(task);
    });

    child.on("close", (code) => {
      // Flush remaining buffers
      if (rt.stdoutBuffer.trim()) {
        task.progress.push(rt.stdoutBuffer);
        taskHistory.appendLog(task.id, [rt.stdoutBuffer]);
      }
      if (rt.stderrBuffer.trim()) {
        task.progress.push(`[${name} stderr] ${rt.stderrBuffer}`);
        taskHistory.appendLog(task.id, [`[${name} stderr] ${rt.stderrBuffer}`]);
      }
      
      if (this.running.has(id)) {
        task.exitCode = code ?? undefined;
        this.handleProcessExit(id);
      }
      this.emitUpdate(task);
    });

    task.status = "running";
    task.startedAt = new Date();
    this.emitUpdate(task);

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

  /**
   * Spawn a child task under a parent task. Automatically establishes
   * the parent-child relationship via parentId / children fields.
   */
  spawnChild(parentId: string, agentType: AgentType, prompt: string, opts?: { name?: string; workdir?: string }): Task {
    const parent = this.tasks.get(parentId);
    if (!parent) {
      throw new Error(`Parent task not found: ${parentId}`);
    }
    const task = this.spawn(agentType, prompt, { ...opts, parentId });

    // Track in children map
    if (!this.children.has(parentId)) {
      this.children.set(parentId, []);
    }
    this.children.get(parentId)!.push(task.id);

    return task;
  }

  /**
   * Check if all children of a task have completed (done/failed/cancelled).
   * If so, emit "children_done" event with the parent task id and child results.
   */
  private checkChildrenDone(taskId: string): void {
    const task = this.tasks.get(taskId);
    if (!task?.parentId) return;

    const parent = this.tasks.get(task.parentId);
    if (!parent?.children || parent.children.length === 0) return;

    const allDone = parent.children.every((childId) => {
      const child = this.tasks.get(childId);
      return child && (child.status === "done" || child.status === "failed" || child.status === "cancelled");
    });

    if (allDone) {
      const childResults = parent.children.map((childId) => {
        const child = this.tasks.get(childId)!;
        return { id: child.id, name: child.name, status: child.status, error: child.error };
      });
      this.emit("task_complete", { parentId: parent.id, children: childResults });
    }
  }

  kill(taskId: string): void {
    const task = this.tasks.get(taskId);
    const rt = this.running.get(taskId);
    
    if (!rt) {
      // Task already finished
      if (task) {
        task.status = "cancelled";
        task.endedAt = new Date();
        this.emitUpdate(task);
      }
      this.emit("kill", { taskId });
      return;
    }

    if (task) {
      task.status = "cancelled";
      task.endedAt = new Date();
    }

    try {
      // Try SIGTERM first, then SIGKILL after 3 seconds
      rt.child.kill("SIGTERM");
      setTimeout(() => {
        if (this.running.has(taskId)) {
          try {
            rt.child.kill("SIGKILL");
          } catch {
            // Process already dead
          }
        }
      }, 3000);
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
