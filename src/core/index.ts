export type UserRole = string;

export type ChannelUser = {
  readonly id: string;
  readonly displayName?: string;
};

export type AgentStreamEvent =
  | { readonly type: "text_delta"; readonly text: string }
  | {
      readonly type: "tool_use_start";
      readonly toolName: string;
      readonly toolUseId: string;
      readonly input: Record<string, unknown>;
    }
  | {
      readonly type: "tool_use_result";
      readonly toolUseId: string;
      readonly result: string;
      readonly isError?: boolean;
    }
  | { readonly type: "thinking"; readonly text: string }
  | { readonly type: "compact_start" }
  | {
      readonly type: "compact_complete";
      readonly preTokens: number;
      readonly postTokens?: number;
      readonly durationMs?: number;
    }
  | { readonly type: "completed"; readonly sessionId: string; readonly costUsd?: number }
  | { readonly type: "error"; readonly message: string };

export type AgentStreamPublisher = (event: AgentStreamEvent) => void;

export type InboundMessage = {
  readonly id: string;
  readonly channel: string;
  readonly user: ChannelUser;
  readonly text: string;
  readonly attachments?: readonly InboundAttachment[];
  readonly receivedAt: Date;
  readonly stream?: AgentStreamPublisher;
  readonly chatId?: string;
  readonly chatType?: "single" | "group";
  readonly roleOverride?: UserRole;
};

export type InboundAttachment = {
  readonly type: "document" | "image";
  readonly name: string;
  readonly mimeType: string;
  readonly storagePath: string;
  readonly sizeBytes: number;
};

export type OutboundMessage = {
  readonly text: string;
  readonly sessionId?: string;
};

/**
 * Unified entry point for all user-facing channels.
 *
 * Channel adapters convert browser, HTTP, or WeChat payloads into InboundMessage,
 * call handle(), then translate OutboundMessage back to their own response format.
 */
export type MessageGateway = {
  handle(message: InboundMessage): Promise<OutboundMessage>;
};

export type AgentConversationMessage = {
  readonly role: "user" | "assistant";
  readonly content: string;
};

export type ContextUsageReport = {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens?: number;
  readonly cacheCreationTokens?: number;
  readonly contextWindow: number;
  readonly usagePercent: number;
};

export type AgentProgressEvent = {
  readonly stage: string;
  readonly message: string;
  readonly data?: Record<string, unknown>;
};

export type AgentProgressPublisher = (event: AgentProgressEvent) => Promise<void>;

export type ProgressReporter = {
  publish(user: ChannelUser, event: AgentProgressEvent): Promise<void>;
};

export type AgentRequest = {
  readonly user: ChannelUser;
  readonly text: string;
  readonly attachments?: readonly InboundAttachment[];
  readonly workspacePath: string;
  readonly role?: string;
  readonly history?: readonly AgentConversationMessage[];
  readonly sessionId?: string;
  readonly progress?: AgentProgressPublisher;
  readonly stream?: AgentStreamPublisher;
  readonly onCompact?: (summary: string) => Promise<void>;
  readonly chatType?: "single" | "group";
  readonly chatId?: string;
};

export type AgentResponse = {
  readonly text: string;
  readonly sessionId?: string;
  readonly contextUsage?: ContextUsageReport;
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

// ── Conversation Logger ──────────────────────────────────────────────────────

export type ConversationLogger = {
  write(event: Record<string, unknown>): Promise<void>;
};
