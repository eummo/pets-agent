import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import { randomBytes } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
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
  /** Token used for external kill signal (e.g. AbortController) */
  token?: string;
}

export const MAX_PROGRESS_LINES = 500;
export type TaskUpdate = Pick<Task, "id" | "status" | "progress" | "error" | "exitCode" | "startedAt" | "endedAt">;

class AgentManager extends EventEmitter {
  private tasks = new Map<string, Task>();
  private running = new Map<string, RunningTask>();
  private subscriptions = new Map<string, Set<(update: TaskUpdate) => void>>();
  /** Token → taskId map for external killByToken */
  private tokenToTask = new Map<string, string>();
  /** Heartbeat interval handle */
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  /** Heartbeat interval in ms (default 30s) */
  private heartbeatIntervalMs = 30_000;

  constructor() {
    super();
    this.startHeartbeat();
  }

  // ─── Heartbeat ─────────────────────────────────────────────────────────────

  private startHeartbeat(): void {
    if (this.heartbeatTimer) return;
    this.heartbeatTimer = setInterval(() => {
      for (const [taskId, rt] of this.running) {
        const task = rt.task;
        // Check if process is still alive
        if (!rt.child.kill(0)) {
          // Process is dead but we haven't processed the exit yet
          console.warn(`[AgentManager] Zombie task detected: ${taskId.slice(0, 8)}, forcing exit`);
          task.error = "Process became unresponsive (zombie)";
          task.status = "failed";
          task.endedAt = new Date();
          this.running.delete(taskId);
          this.emitUpdate(task);
          taskHistory.add(task);
          this.checkChildrenDone(taskId);
          this.emit("exit", { taskId, exitCode: null });
        } else {
          // Send ping over IPC if supported
          rt.child.send?.({ type: "ping", taskId });
        }
      }
    }, this.heartbeatIntervalMs);
  }

  // ─── Progress ─────────────────────────────────────────────────────────────

  private pushProgress(task: Task, line: string): void {
    if (task.progress.length >= MAX_PROGRESS_LINES) {
      task.progress.splice(0, task.progress.length - MAX_PROGRESS_LINES + 1);
    }
    task.progress.push(line);
  }

  // ─── Exit handling ─────────────────────────────────────────────────────────

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
      this.pushProgress(task, rt.stdoutBuffer);
      taskHistory.appendLog(task.id, [rt.stdoutBuffer]);
    }
    if (rt.stderrBuffer.trim()) {
      this.pushProgress(task, `[stderr] ${rt.stderrBuffer}`);
      taskHistory.appendLog(task.id, [`[stderr] ${rt.stderrBuffer}`]);
    }

    if (rt.token) this.tokenToTask.delete(rt.token);

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

  // ─── WSL detection ────────────────────────────────────────────────────────

  /** More robust WSL path detection */
  private static isWslPath(p: string): boolean {
    if (!p || p.length < 7) return false;
    // /mnt/x/ — drive letter at index 5, slash at index 6
    if (!p.startsWith("/mnt/")) return false;
    const ch = p[5];
    if (!ch) return false;
    const afterSlash = p[6];
    return afterSlash === "/" && ch >= "a" && ch <= "z" && (p.match(/^\/mnt\/[a-z]\//)?.length ?? 0) > 0;
  }

  /** Convert WSL path to Windows absolute path */
  private static wslToWindowsPath(wslPath: string): string {
    if (!AgentManager.isWslPath(wslPath)) return wslPath;
    const drive = wslPath[5]!.toUpperCase();
    const rest = wslPath.slice(7).replace(/\//g, "\\");
    return `${drive}:\\${rest}`;
  }

  /** Detect if we are running inside WSL */
  private static isWsl(): boolean {
    if (process.platform !== "linux") return false;
    const wslDistro = process.env.WSL_DISTRO_NAME;
    const wslEnv = process.env.WSLENV;
    // Also check /proc/version for WSL signature as fallback
    if (wslDistro || wslEnv) return true;
    try {
      const kernel = fs.readFileSync("/proc/version", "utf8").toLowerCase();
      return kernel.includes("microsoft") || kernel.includes("wsl");
    } catch {
      return false;
    }
  }

  private static filteredEnv(): Record<string, string> {
    const filtered: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (k.startsWith("HERMES_") || k === "WSL_DISTRO_NAME" || k === "WSLENV") continue;
      if (v !== undefined) filtered[k] = v;
    }
    return filtered;
  }

  // ─── Command building ───────────────────────────────────────────────────────

  private buildCommand(
    agentType: AgentType,
    _workdir?: string,
  ): { cmd: string; args: string[]; passPromptViaStdin: boolean; name: string } {
    switch (agentType) {
      case "claude-code": {
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
          cmd: candidates[0]!,
          args: ["-p", "--dangerously-skip-permissions", "--no-session-persistence", "--bare"],
          passPromptViaStdin: false,
          name: "claude-code",
        };
      }
      case "pi-agent":
        return {
          cmd: "npx",
          args: ["-y", "@earendil-works/pi-coding-agent", "-p", "--no-input"],
          passPromptViaStdin: false,
          name: "pi-agent",
        };
      case "codex":
        return { cmd: "npx", args: ["-y", "openai/codex", "--acp", "--stdio"], passPromptViaStdin: true, name: "codex" };
      case "kiro":
        return { cmd: "kiro", args: ["--acp", "--stdio"], passPromptViaStdin: true, name: "kiro" };
      default:
        return { cmd: agentType, args: ["--acp", "--stdio"], passPromptViaStdin: true, name: agentType };
    }
  }

  // ─── Spawn ─────────────────────────────────────────────────────────────────

  spawn(
    agentType: AgentType,
    prompt: string,
    opts?: {
      name?: string;
      workdir?: string;
      parentId?: string;
      timeoutMs?: number;
      maxRetries?: number;
      priority?: number;
      token?: string;
    },
  ): Task {
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
      priority: opts?.priority,
    };

    if (opts?.parentId) {
      const parent = this.tasks.get(opts.parentId);
      if (parent) {
        if (!parent.children) parent.children = [];
        parent.children.push(id);
      }
    }

    this.tasks.set(id, task);

    const { cmd, args, passPromptViaStdin, name } = this.buildCommand(agentType, opts?.workdir);
    const filteredEnv = AgentManager.filteredEnv();

    // Resolve and normalize workdir, handle WSL paths
    let workdir = opts?.workdir ?? process.cwd();
    if (AgentManager.isWslPath(workdir) && AgentManager.isWsl()) {
      // Running inside WSL, convert path for subprocess running on Windows side
      workdir = AgentManager.wslToWindowsPath(workdir);
    } else if (AgentManager.isWslPath(workdir) && !AgentManager.isWsl()) {
      // Cross-context call from Linux to Windows path — also convert
      workdir = AgentManager.wslToWindowsPath(workdir);
    }

    if (workdir && !fs.existsSync(workdir)) {
      fs.mkdirSync(workdir, { recursive: true });
    }

    const spawnArgs = passPromptViaStdin ? args : [...args, prompt];
    const spawnOpts: Parameters<typeof spawn>[2] = {
      cwd: workdir,
      env: filteredEnv,
      stdio: passPromptViaStdin
        ? ["pipe", "pipe", "pipe", "ipc"]
        : ["ignore", "pipe", "pipe", "ipc"],
    };

    const child = spawn(cmd, spawnArgs, spawnOpts);

    const rt: RunningTask = {
      task,
      child,
      stdoutBuffer: "",
      stderrBuffer: "",
      token: opts?.token,
    };

    if (opts?.token) this.tokenToTask.set(opts.token, id);
    this.running.set(id, rt);

    // Stream stdout
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

    // Stream stderr
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
      if (rt.token) this.tokenToTask.delete(rt.token);
      this.running.delete(id);
      this.emitUpdate(task);
      taskHistory.add(task);
    });

    child.on("close", (code) => {
      if (rt.stdoutBuffer.trim()) {
        this.pushProgress(task, rt.stdoutBuffer);
        taskHistory.appendLog(task.id, [rt.stdoutBuffer]);
      }
      if (rt.stderrBuffer.trim()) {
        this.pushProgress(task, `[${name} stderr] ${rt.stderrBuffer}`);
        taskHistory.appendLog(task.id, [`[${name} stderr] ${rt.stderrBuffer}`]);
      }

      if (this.running.has(id)) {
        task.exitCode = code ?? undefined;
        this.handleProcessExit(id);
      }
    });

    // IPC pong response
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

    task.status = "running";
    task.startedAt = new Date();
    this.emitUpdate(task);

    // Timeout
    if (opts?.timeoutMs && opts.timeoutMs > 0) {
      const timer = setTimeout(() => {
        if (this.running.has(id)) {
          console.warn(`[AgentManager] Task ${id.slice(0, 8)} timed out after ${opts.timeoutMs}ms, killing...`);
          task.error = `Timed out after ${opts.timeoutMs}ms`;
          this.kill(id);
        }
      }, opts.timeoutMs);
      child.on("close", () => clearTimeout(timer));
    }

    return task;
  }

  // ─── Spawn with retry ───────────────────────────────────────────────────────

  /**
   * Spawn a task with automatic retry on failure.
   * Uses a state machine to avoid the previous closure-based race condition.
   * Superseded tasks are linked via supersededBy.
   */
  spawnWithRetry(
    agentType: AgentType,
    prompt: string,
    opts?: {
      name?: string;
      workdir?: string;
      parentId?: string;
      timeoutMs?: number;
      maxRetries?: number;
      priority?: number;
      token?: string;
    },
  ): Task {
    const maxRetries = opts?.maxRetries ?? 0;
    const retryState = {
      currentTaskId: "",
      attempt: 1,
      done: false,
    };

    const spawnCurrent = (): Task => {
      const task = this.spawn(agentType, prompt, {
        ...opts,
        name: opts?.name,
      });
      retryState.currentTaskId = task.id;
      task.attempt = retryState.attempt;
      return task;
    };

    let currentTask = spawnCurrent();

    // Register ONE permanent exit handler that survives across retries
    const handleExit = (payload: { taskId: string; exitCode: number | null }): void => {
      if (payload.taskId !== retryState.currentTaskId) return;
      if (retryState.done) return;

      const task = this.tasks.get(retryState.currentTaskId);
      if (!task) return;

      // Only retry on actual failures
      if (task.status === "failed" && retryState.attempt < maxRetries) {
        const delay = retryState.attempt * 1000;
        console.warn(
          `[AgentManager] Task ${retryState.currentTaskId.slice(0, 8)} failed (attempt ${retryState.attempt}/${maxRetries}), retrying in ${delay}ms...`,
        );

        retryState.attempt++;
        const newTask = this.spawn(agentType, prompt, {
          ...opts,
          name: opts?.name,
        });
        newTask.attempt = retryState.attempt;
        task.supersededBy = newTask.id;
        retryState.currentTaskId = newTask.id;

        // Reschedule exit handler on the new task
        setTimeout(() => {
          this.once("exit", handleExit);
        }, delay);
      } else {
        retryState.done = true;
        // Let the normal flow handle it
      }
    };

    this.once("exit", handleExit);
    return currentTask;
  }

  // ─── Child spawn ────────────────────────────────────────────────────────────

  spawnChild(
    parentId: string,
    agentType: AgentType,
    prompt: string,
    opts?: { name?: string; workdir?: string },
  ): Task {
    const parent = this.tasks.get(parentId);
    if (!parent) {
      throw new Error(`Parent task not found: ${parentId}`);
    }
    return this.spawn(agentType, prompt, { ...opts, parentId });
  }

  // ─── Children tracking ──────────────────────────────────────────────────────

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

  // ─── Kill ──────────────────────────────────────────────────────────────────

  kill(taskId: string): void {
    const task = this.tasks.get(taskId);
    const rt = this.running.get(taskId);

    if (!rt) {
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

    if (rt.token) this.tokenToTask.delete(rt.token);

    try {
      rt.child.kill("SIGTERM");
      setTimeout(() => {
        if (this.running.has(taskId)) {
          try {
            rt.child.kill("SIGKILL");
          } catch {
            // Already dead
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

  /** Kill a task by its token (e.g. from AbortSignal) */
  killByToken(token: string): void {
    const taskId = this.tokenToTask.get(token);
    if (taskId) {
      this.tokenToTask.delete(token);
      this.kill(taskId);
    }
  }

  // ─── Query ─────────────────────────────────────────────────────────────────

  list(includeSuperseded = false): Task[] {
    const all = Array.from(this.tasks.values());
    if (includeSuperseded) return all;

    // Sort by priority (higher first) for non-superseded tasks
    return all
      .filter((t) => !t.supersededBy)
      .sort((a, b) => {
        const pa = a.priority ?? 5;
        const pb = b.priority ?? 5;
        if (pa !== pb) return pb - pa;
        return a.createdAt.getTime() - b.createdAt.getTime();
      });
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
        if (subs.size === 0) this.subscriptions.delete(taskId);
      }
    };
  }

  getActiveTasks(): Task[] {
    return this.list().filter((t) => t.status === "running" || t.status === "pending");
  }

  /**
   * Debug/status snapshot — returns a structured snapshot of all task state
   * for /status HTTP endpoint integration or REPL inspection.
   */
  getStatus(): {
    running: Array<{
      id: string;
      name: string;
      agentType: string;
      status: string;
      age: number;
      progressLines: number;
      hasProgress: boolean;
      zombie: boolean;
    }>;
    subscriptions: number;
    zombieRisk: string[];
    totals: { running: number; done: number; failed: number; cancelled: number };
  } {
    const now = Date.now();
    const running: Array<{
      id: string; name: string; agentType: string; status: string;
      age: number; progressLines: number; hasProgress: boolean; zombie: boolean;
    }> = [];
    const entries = Array.from(this.running.entries());
    for (const [taskId, rt] of entries) {
      const age = rt.task.startedAt ? now - rt.task.startedAt.getTime() : 0;
      const noProgress = rt.task.progress.length === 0;
      const zombie = age > 30 * 60 * 1000 && noProgress;
      running.push({
        id: taskId,
        name: rt.task.name,
        agentType: rt.task.agentType,
        status: rt.task.status,
        age,
        progressLines: rt.task.progress.length,
        hasProgress: !noProgress,
        zombie,
      });
    }
    const zombieRisk = running.filter((r) => r.zombie).map((r) => r.id);

    let subs = 0;
    const subEntries = Array.from(this.subscriptions.entries());
    for (const s of subEntries) subs += s[1].size;

    return {
      running,
      subscriptions: subs,
      zombieRisk,
      totals: {
        running: this.running.size,
        done: 0,
        failed: 0,
        cancelled: 0,
      },
    };
  }

  destroy(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
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
