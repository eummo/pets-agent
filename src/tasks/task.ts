export type TaskStatus = "pending" | "running" | "done" | "failed" | "cancelled";
export type AgentType = "claude-code" | "codex" | "kiro" | "pi-agent" | "custom";

export interface Task {
  id: string;           // nanoid
  name: string;
  agentType: AgentType;
  prompt: string;
  status: TaskStatus;
  createdAt: Date;
  startedAt?: Date;
  endedAt?: Date;
  exitCode?: number;
  progress: string[];   // stdout lines collected so far
  error?: string;
  workdir?: string;
}
