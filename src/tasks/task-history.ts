import * as fs from "fs";
import * as path from "path";
import { homedir } from "os";
import type { Task } from "./task.js";

const HISTORY_DIR = path.join(homedir(), ".pets-agent");
const HISTORY_FILE = path.join(HISTORY_DIR, "task-history.json");
const LOGS_DIR = path.join(HISTORY_DIR, "logs");

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
}

export class TaskHistory {
  private entries: TaskHistoryEntry[] = [];
  private maxEntries = 500;
  private maxProgressLines = 100;

  constructor() {
    this.load();
  }

  load(): void {
    try {
      if (fs.existsSync(HISTORY_FILE)) {
        const data = fs.readFileSync(HISTORY_FILE, "utf8");
        this.entries = JSON.parse(data);
      }
    } catch {
      this.entries = [];
    }
  }

  getLogFile(taskId: string): string {
    return path.join(LOGS_DIR, `${taskId}.log`);
  }

  writeLog(taskId: string, lines: string[]): void {
    try {
      if (!fs.existsSync(LOGS_DIR)) {
        fs.mkdirSync(LOGS_DIR, { recursive: true });
      }
      const content = lines.join("\n") + "\n";
      fs.writeFileSync(this.getLogFile(taskId), content);
    } catch (err) {
      console.error("Failed to write task log:", err);
    }
  }

  appendLog(taskId: string, lines: string[]): void {
    try {
      if (!fs.existsSync(LOGS_DIR)) {
        fs.mkdirSync(LOGS_DIR, { recursive: true });
      }
      const content = lines.join("\n") + "\n";
      fs.appendFileSync(this.getLogFile(taskId), content);
    } catch (err) {
      // ignore
    }
  }

  save(): void {
    try {
      if (!fs.existsSync(HISTORY_DIR)) {
        fs.mkdirSync(HISTORY_DIR, { recursive: true });
      }
      fs.writeFileSync(HISTORY_FILE, JSON.stringify(this.entries, null, 2));
    } catch (err) {
      console.error("Failed to save task history:", err);
    }
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
    // 估算创建的文件数（统计 progress 中包含 "Created N file" 的行）
    const fileMatch = task.progress.join("\n").match(/Created (\d+) file/);
    if (fileMatch) {
      entry.fileCount = parseInt(fileMatch[1]);
    }

    this.writeLog(task.id, task.progress);

    this.entries.unshift(entry);
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(0, this.maxEntries);
    }
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
    return results.slice(0, q.limit ?? 50);
  }

  getAll(): TaskHistoryEntry[] {
    return this.entries;
  }

  readLog(taskId: string): string[] {
    try {
      const logFile = this.getLogFile(taskId);
      if (fs.existsSync(logFile)) {
        const content = fs.readFileSync(logFile, "utf8");
        return content.split("\n").filter(Boolean);
      }
    } catch {}
    return [];
  }
}

export const taskHistory = new TaskHistory();
