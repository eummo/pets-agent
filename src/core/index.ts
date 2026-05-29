export type UserRole = string;

export type ChannelUser = {
  readonly id: string;
  readonly displayName?: string;
};

export type InboundMessage = {
  readonly id: string;
  readonly channel: string;
  readonly user: ChannelUser;
  readonly text: string;
  readonly receivedAt: Date;
  readonly stream?: import("../agent/index.js").AgentStreamPublisher;
  readonly chatId?: string;
  readonly chatType?: "single" | "group";
  readonly roleOverride?: UserRole;
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

// ── Conversation Logger ──────────────────────────────────────────────────────

export type ConversationLogger = {
  write(event: Record<string, unknown>): Promise<void>;
};
