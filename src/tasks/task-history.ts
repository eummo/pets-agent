/**
 * TaskHistory — disk-backed task execution log with optional in-memory write log.
 *
 * All I/O errors are classified as RECOVERABLE (disk full/permission) or FATAL
 * (corruption/structural). RECOVERABLE errors are logged to stderr and the
 * operation continues gracefully. FATAL errors are also logged and should
 * prompt a reload of the store.
 */

import * as fs from "fs";
import * as path from "path";
import { homedir } from "os";
import type { Task } from "./task.js";

const HISTORY_DIR = path.join(homedir(), ".pets-agent");
const HISTORY_FILE = path.join(HISTORY_DIR, "task-history.json");
const LOGS_DIR = path.join(HISTORY_DIR, "logs");
/** Max log files before oldest are pruned */
const MAX_LOG_FILES = 500;

export interface TaskHistoryEntry {
  id: string;
  name: string;
  agentType: string;
  prompt: string;
  status: string;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  exitCode?: number;
  error?: string;
  workdir?: string;
  progress: string[];
  fileCount?: number;
  logFile: string;
}

export interface TaskHistoryQuery {
  agentType?: string;
  status?: string;
  since?: string;
  until?: string;
  limit?: number;
  /** Number of entries to skip (for pagination). Default: 0. */
  offset?: number;
}

// Re-export Task types for convenience
export type { Task, TaskStatus, AgentType } from "./task.js";

/** Classify an I/O error by severity */
function classifyError(err: unknown): { severity: "recoverable" | "fatal"; message: string } {
  const msg = err instanceof Error ? err.message : String(err);
  const code = (err as NodeJS.ErrnoException).code;
  if (code === "EACCES" || code === "ENOSPC" || code === "EROFS") {
    return { severity: "recoverable", message: msg };
  }
  return { severity: "fatal", message: msg };
}

/** Write a line to stderr, used for structural/log errors */
function warn(msg: string, err: unknown): void {
  const detail = err instanceof Error ? err.message : String(err);
  process.stderr.write(`[TaskHistory] ${msg}: ${detail}\n`);
}

export class TaskHistory {
  private entries: TaskHistoryEntry[] = [];
  private maxEntries = 500;
  private maxProgressLines = 100;
  private saveDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingSave = false;

  constructor() {
    this.load();
  }

  load(): void {
    try {
      if (fs.existsSync(HISTORY_FILE)) {
        const data = fs.readFileSync(HISTORY_FILE, "utf8");
        try {
          this.entries = JSON.parse(data);
        } catch (parseError) {
          warn("Failed to parse history file", parseError);
          this.entries = [];
        }
      }
    } catch (err) {
      warn("Failed to load history", err);
      this.entries = [];
    }
  }

  getLogFile(taskId: string): string {
    return path.join(LOGS_DIR, `${taskId}.log`);
  }

  /**
   * Write log lines to a task's log file (full overwrite).
   * RECOVERABLE: logs error to stderr on failure, returns early.
   */
  writeLog(taskId: string, lines: string[]): void {
    try {
      if (!fs.existsSync(LOGS_DIR)) {
        fs.mkdirSync(LOGS_DIR, { recursive: true });
      }
      const content = lines.join("\n") + "\n";
      fs.writeFileSync(this.getLogFile(taskId), content);
    } catch (err) {
      warn("Failed to write task log", err);
    }
  }

  /**
   * Append log lines to a task's log file.
   * RECOVERABLE: silent on failure (logs grow large; append failures are non-critical).
   */
  appendLog(taskId: string, lines: string[]): void {
    try {
      if (!fs.existsSync(LOGS_DIR)) {
        fs.mkdirSync(LOGS_DIR, { recursive: true });
      }
      const content = lines.join("\n") + "\n";
      fs.appendFileSync(this.getLogFile(taskId), content);
    } catch {
      // silent — append failures on logs are non-critical
    }
  }

  /**
   * Persist the in-memory entries array to HISTORY_FILE.
   * RECOVERABLE: debounced; errors are logged and the in-memory state is retained.
   */
  save(): void {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    this.saveDebounceTimer = setTimeout(() => {
      this.pendingSave = true;
      try {
        if (!fs.existsSync(HISTORY_DIR)) {
          fs.mkdirSync(HISTORY_DIR, { recursive: true });
        }
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(this.entries, null, 2));
      } catch (err) {
        warn("Failed to save history", err);
      } finally {
        this.pendingSave = false;
      }
    }, 1000);
  }

  add(task: Task): void {
    const logFile = this.getLogFile(task.id);
    const entry: TaskHistoryEntry = {
      id: task.id,
      name: task.name,
      agentType: task.agentType,
      prompt: task.prompt,
      status: task.status,
      createdAt: task.createdAt.toISOString(),
      startedAt: task.startedAt?.toISOString(),
      endedAt: task.endedAt?.toISOString(),
      exitCode: task.exitCode,
      error: task.error,
      workdir: task.workdir,
      progress: task.progress.slice(-this.maxProgressLines),
      logFile,
    };
    // Estimate file count from progress lines like "Created N file"
    const fileMatch = task.progress.join("\n").match(/Created (\d+) file/);
    if (fileMatch) {
      entry.fileCount = parseInt(fileMatch[1]!, 10);
    }

    this.writeLog(task.id, task.progress);

    this.entries.unshift(entry);
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(0, this.maxEntries);
    }
    this.pruneLogFiles();
    this.save();
  }

  query(q: TaskHistoryQuery): TaskHistoryEntry[] {
    let results = [...this.entries];
    if (q.agentType) results = results.filter((e) => e.agentType === q.agentType);
    if (q.status) results = results.filter((e) => e.status === q.status);
    if (q.since) {
      const since = new Date(q.since).getTime();
      results = results.filter((e) => new Date(e.createdAt).getTime() >= since);
    }
    if (q.until) {
      const until = new Date(q.until).getTime();
      results = results.filter((e) => new Date(e.createdAt).getTime() <= until);
    }
    return results.slice(q.offset ?? 0, (q.offset ?? 0) + (q.limit ?? 50));
  }

  getAll(): TaskHistoryEntry[] {
    return this.entries;
  }

  /**
   * Read a task's log file from disk.
   * Returns empty array on any error (RECOVERABLE).
   */
  readLog(taskId: string): string[] {
    try {
      const logFile = this.getLogFile(taskId);
      if (fs.existsSync(logFile)) {
        const content = fs.readFileSync(logFile, "utf8");
        return content.split("\n").filter(Boolean);
      }
    } catch (err) {
      warn(`Failed to read log ${taskId}`, err);
    }
    return [];
  }

  /**
   * Prune oldest log files if LOGS_DIR exceeds MAX_LOG_FILES.
   * Called automatically after each new log write.
   * FATAL errors are silently ignored (pruning is best-effort).
   */
  private pruneLogFiles(): void {
    try {
      if (!fs.existsSync(LOGS_DIR)) return;
      const files = fs.readdirSync(LOGS_DIR).filter((f) => f.endsWith(".log"));
      if (files.length <= MAX_LOG_FILES) return;

      // Sort by mtime ascending (oldest first)
      const withMtime = files
        .map((f) => {
          const fp = path.join(LOGS_DIR, f);
          const stat = fs.statSync(fp);
          return { file: f, mtime: stat.mtimeMs };
        })
        .sort((a, b) => a.mtime - b.mtime);

      const toDelete = files.length - MAX_LOG_FILES;
      for (let i = 0; i < toDelete; i++) {
        try {
          fs.unlinkSync(path.join(LOGS_DIR, withMtime[i]!.file));
        } catch {
          /* ignore individual delete failures */
        }
      }
    } catch {
      /* ignore pruning failures */
    }
  }

  /**
   * Force an immediate save, flushing any pending debounced save.
   * Call this before process exit to ensure all history is persisted.
   * FATAL errors are logged to stderr and the in-memory state is retained.
   */
  flush(): void {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
      this.saveDebounceTimer = null;
    }
    if (!fs.existsSync(HISTORY_DIR)) {
      fs.mkdirSync(HISTORY_DIR, { recursive: true });
    }
    try {
      fs.writeFileSync(HISTORY_FILE, JSON.stringify(this.entries, null, 2));
    } catch (err) {
      warn("Failed to flush history", err);
    }
  }

  /**
   * Clean up resources. Call on application shutdown.
   */
  destroy(): void {
    this.flush();
  }
}

export const taskHistory = new TaskHistory();
