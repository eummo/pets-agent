import type { AgentConversationMessage } from "../core/index.js";

export type ConversationSessionKey = {
  readonly channel: string;
  readonly userId: string;
  readonly workspacePath: string;
  readonly chatId?: string;
};

export type ConversationSessionStore = {
  get(key: ConversationSessionKey): Promise<string | undefined>;
  set(key: ConversationSessionKey, sessionId: string): Promise<void>;
  delete(key: ConversationSessionKey): Promise<void>;
};

export type ConversationHistoryStore = {
  get(key: ConversationSessionKey): Promise<readonly AgentConversationMessage[]>;
  append(key: ConversationSessionKey, messages: readonly AgentConversationMessage[]): Promise<void>;
  compact(key: ConversationSessionKey, summary: string): Promise<void>;
  delete(key: ConversationSessionKey): Promise<void>;
  archive(key: ConversationSessionKey): Promise<void>;
};

export type FeedbackStatus = "pending" | "reviewed" | "resolved";

export type FeedbackEntry = {
  readonly id?: number;
  readonly userId: string;
  readonly channel?: string;
  readonly messageId?: string;
  readonly workspacePath?: string;
  readonly intentType?: "query" | "suggest" | "mutate" | "update_kb";
  readonly roleName?: string;
  readonly userMessage: string;
  readonly conversationContext: string;
  readonly status: FeedbackStatus;
  readonly createdAt?: string;
  readonly updatedAt?: string;
};

export type FeedbackQuery = {
  readonly limit?: number;
  readonly offset?: number;
  readonly status?: FeedbackStatus;
};

export type FeedbackStore = {
  save(entry: FeedbackEntry): Promise<number>;
  updateStatus(id: number, status: FeedbackStatus): Promise<boolean>;
  getAll(query?: FeedbackQuery): Promise<readonly FeedbackEntry[]>;
};
