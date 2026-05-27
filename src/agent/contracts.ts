import type { AgentConversationMessage, ChannelUser } from "../core/contracts.js";

export type ContextUsageReport = {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens?: number;
  readonly cacheCreationTokens?: number;
  readonly contextWindow: number;
  readonly usagePercent: number;
};

export type AgentRequest = {
  readonly user: ChannelUser;
  readonly text: string;
  readonly workspacePath: string;
  readonly role?: string;
  readonly history?: readonly AgentConversationMessage[];
  readonly sessionId?: string;
  readonly progress?: AgentProgressPublisher;
  readonly stream?: AgentStreamPublisher;
  readonly onCompact?: (summary: string) => Promise<void>;
};

export type AgentResponse = {
  readonly text: string;
  readonly sessionId?: string;
  readonly contextUsage?: ContextUsageReport;
};

export type AgentStreamEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_use_start"; toolName: string; toolUseId: string; input: Record<string, unknown> }
  | { type: "tool_use_result"; toolUseId: string; result: string; isError?: boolean }
  | { type: "thinking"; text: string }
  | { type: "compact_start" }
  | { type: "compact_complete"; preTokens: number; postTokens?: number; durationMs?: number }
  | { type: "completed"; sessionId: string; costUsd?: number }
  | { type: "error"; message: string };

export type AgentStreamPublisher = (event: AgentStreamEvent) => void;

export type AgentProgressEvent = {
  readonly stage: string;
  readonly message: string;
  readonly data?: Record<string, unknown>;
};

export type AgentProgressPublisher = (event: AgentProgressEvent) => Promise<void>;

export type ProgressReporter = {
  publish(user: ChannelUser, event: AgentProgressEvent): Promise<void>;
};

export type AgentRuntime = {
  readonly name: string;
  run(request: AgentRequest): Promise<AgentResponse>;
  disposeSession(sessionId: string): Promise<void>;
};

export type AgentRuntimeFactory = {
  warmup(): Promise<Record<string, AgentRuntime>>;
  cacheKeyForRole?(role: string): Promise<string | undefined>;
  createRuntime(role: string): Promise<AgentRuntime | undefined>;
};

