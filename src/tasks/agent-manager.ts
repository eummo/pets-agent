import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import { randomBytes } from "crypto";
import type { Task, TaskStatus, AgentType } from "./task.js";

function generateId(): string {
  return randomBytes(8).toString("hex");
}

interface RunningTask {
  task: Task;
  child: ChildProcess;
  stdoutBuffer: string;
  stderrBuffer: string;
}

type TaskUpdate = Pick<Task, "id" | "status" | "progress" | "error" | "exitCode" | "startedAt" | "endedAt">;

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
  }

  private broadcastUpdate(task: Task): TaskUpdate {
    return {
      id: task.id,
      status: task.status,
      progress: [...task.progress],
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
        try { cb(update); } catch { /* ignore */ }
      }
    }
  }

  private emitUpdate(task: Task): void {
    const update = this.broadcastUpdate(task);
    this.emit("update", update);
    this.notifySubscribers(task.id, update);
  }

  private buildCommand(agentType: AgentType): { cmd: string; args: string[] } {
    switch (agentType) {
      case "claude-code":
        return { cmd: "claude", args: ["--acp", "--stdio"] };
      case "codex":
        return { cmd: "codex", args: ["--acp", "--stdio"] };
      case "kiro":
        return { cmd: "kiro", args: ["--acp", "--stdio"] };
      default:
        return { cmd: agentType, args: ["--acp", "--stdio"] };
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

    const { cmd, args } = this.buildCommand(agentType);

    const child = spawn(cmd, args, {
      stdio: ["pipe", "pipe", "pipe", "ipc"],
      cwd: opts?.workdir ?? process.cwd(),
      env: { ...process.env },
    });

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
    });

    child.on("close", (code) => {
      if (rt.stdoutBuffer.trim()) {
        task.progress.push(rt.stdoutBuffer);
        this.emitUpdate(task);
      }
      if (this.running.has(id)) {
        task.exitCode = code ?? undefined;
        this.handleProcessExit(id);
      }
    });

    // ACP handshake: wait for start ACK then send prompt
    const handshakeTimeout = setTimeout(() => {
      if (task.status === "pending") {
        task.status = "running";
        task.startedAt = new Date();
        this.sendPrompt(id, prompt);
      }
    }, 2000);

    child.stdout?.once("data", (data: Buffer) => {
      clearTimeout(handshakeTimeout);
      const msg = this.parseAcpMessage(data.toString());
      if (msg && msg.type === "start") {
        task.status = "running";
        task.startedAt = new Date();
      } else {
        task.status = "running";
        task.startedAt = new Date();
        this.sendPrompt(id, prompt);
      }
      this.emitUpdate(task);
    });

    // IPC channel for control messages
    child.on("message", (msg: unknown) => {
      try {
        const m = msg as { type?: string; taskId?: string };
        if (m.type === "ping") {
          child.send?.({ type: "pong", taskId: m.taskId });
        }
      } catch { /* ignore */ }
    });

    // Kick off status check
    setTimeout(() => {
      if (task.status === "pending") {
        task.status = "running";
        task.startedAt = new Date();
        this.emitUpdate(task);
        this.sendPrompt(id, prompt);
      }
    }, 3000);

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
    const rt = this.running.get(taskId);
    if (!rt) return;

    const task = this.tasks.get(taskId);
    if (task) {
      task.status = "cancelled";
      task.endedAt = new Date();
    }

    try {
      rt.child.kill("SIGTERM");
    } catch { /* ignore */ }

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

    // Emit current state immediately
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
    this.removeAllListeners();
  }
}

// Singleton instance
export const agentManager = new AgentManager();
export { AgentManager };
