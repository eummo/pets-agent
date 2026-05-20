export type TaskStatus = "pending" | "running" | "done" | "failed" | "cancelled";
export type AgentType = "claude-code" | "codex" | "kiro" | "pi-agent" | "custom";

export interface Task {
  id: string;
  name: string;
  agentType: AgentType;
  prompt: string;
  status: TaskStatus;
  createdAt: Date;
  startedAt?: Date;
  endedAt?: Date;
  exitCode?: number;
  progress: string[];
  error?: string;
  workdir?: string;
  parentId?: string;
  children?: string[];
  /** Populated when this task was superseded by a retry attempt */
  supersededBy?: string;
  /** Current attempt number (1-based, for spawnWithRetry tasks) */
  attempt?: number;
  /** Task priority 1-10, higher runs first (default: 5) */
  priority?: number;
}
