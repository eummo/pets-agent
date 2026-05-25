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
  readonly stream?: AgentStreamPublisher;
  readonly chatId?: string;
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

export type AgentStreamEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_use_start"; toolName: string; toolUseId: string; input: Record<string, unknown> }
  | { type: "tool_use_result"; toolUseId: string; result: string; isError?: boolean }
  | { type: "thinking"; text: string }
  | { type: "completed"; sessionId: string; costUsd?: number }
  | { type: "error"; message: string };

export type AgentStreamPublisher = (event: AgentStreamEvent) => void;

export type AgentRequest = {
  readonly user: ChannelUser;
  readonly text: string;
  readonly workspacePath: string;
  readonly history?: readonly AgentConversationMessage[];
  readonly sessionId?: string;
  readonly progress?: AgentProgressPublisher;
  readonly stream?: AgentStreamPublisher;
};

export type AgentResponse = {
  readonly text: string;
  readonly sessionId?: string;
};

export type AgentRuntime = {
  readonly name: string;
  run(request: AgentRequest): Promise<AgentResponse>;
  disposeSession(sessionId: string): Promise<void>;
};

export type AgentRuntimeFactory = {
  cacheKeyForRole?(role: string): Promise<string | undefined>;
  createRuntime(role: string): Promise<AgentRuntime | undefined>;
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

export type AgentConversationMessage = {
  readonly role: "user" | "assistant";
  readonly content: string;
};

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
  delete(key: ConversationSessionKey): Promise<void>;
  archive(key: ConversationSessionKey): Promise<void>;
};

export type KnowledgeWorkspace = {
  readonly kind: "knowledge-base" | "source-repository";
  readonly id: string;
  readonly path: string;
};

export type KnowledgeWorkspaceResolver = {
  resolve(message: InboundMessage): Promise<readonly KnowledgeWorkspace[]>;
};

export type AuthorizationAction = "read" | "suggest" | "mutate";

export type AuthorizationDecision = {
  readonly allowed: boolean;
  readonly reason?: string;
};

export type AuthorizationService = {
  roleFor(user: ChannelUser): Promise<UserRole>;
  can(user: ChannelUser, action: AuthorizationAction, workspace: KnowledgeWorkspace): Promise<AuthorizationDecision>;
  hasCapability(user: ChannelUser, capability: RoleCapability): Promise<boolean>;
  setRole?(userId: string, role: string): void;
};

// ── Role Configuration Store ─────────────────────────────────────────────────

// ── Role Capabilities ────────────────────────────────────────────────────────
// Each capability is an independent, composable unit that a role can possess.
// Adding a new capability only requires extending this union and assigning it
// to the desired roles in the database — no code changes needed elsewhere.

export type RoleCapability =
  | "workspace_read"       // browse and read workspace content
  | "workspace_mutate"     // modify files in the workspace
  | "feedback_view"        // view user feedback entries
  | "feedback_manage"      // review and update feedback status
  | "roles_manage";        // create, update, delete role configurations (future)

export type StoredRoleConfig = {
  readonly name: string;
  readonly systemPrompt: string;
  readonly allowedTools: readonly string[];
  readonly permissionMode: "auto" | "dontAsk" | "acceptEdits" | "bypassPermissions";
  readonly maxTurns?: number;
  readonly model?: string;
  readonly capabilities?: readonly RoleCapability[];
  readonly updatedAt?: string;
};

export const FILE_MUTATION_TOOLS: ReadonlySet<string> = new Set([
  "Edit",
  "MultiEdit",
  "NotebookEdit",
  "Write",
]);

export type RoleConfigStore = {
  getAll(): Promise<readonly StoredRoleConfig[]>;
  getByName(name: string): Promise<StoredRoleConfig | undefined>;
  upsert(config: StoredRoleConfig): Promise<void>;
  deleteByName(name: string): Promise<boolean>;
};

// ── Intent Detection ─────────────────────────────────────────────────────────

export type UserIntent =
  | { readonly type: "query" }
  | { readonly type: "mutate" }
  | { readonly type: "update_kb" };

export type IntentDetectionService = {
  detectIntent(userMessage: string, role: UserRole, history?: readonly AgentConversationMessage[]): Promise<UserIntent>;
};

// ── Feedback Store ───────────────────────────────────────────────────────────

export type FeedbackStatus = "pending" | "reviewed" | "resolved";

export type FeedbackEntry = {
  readonly id?: number;
  readonly userId: string;
  readonly channel?: string;
  readonly messageId?: string;
  readonly workspacePath?: string;
  readonly intentType?: UserIntent["type"];
  readonly roleName?: UserRole;
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

// ── Conversation Logger ──────────────────────────────────────────────────────

export type ConversationLogger = {
  write(event: Record<string, unknown>): Promise<void>;
};
