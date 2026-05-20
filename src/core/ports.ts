export type UserRole = "developer" | "viewer";

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
};

export type OutboundMessage = {
  readonly text: string;
};

export type MessageChannel = {
  readonly name: string;
  send(message: OutboundMessage, replyTo: InboundMessage): Promise<void>;
};

export type MessageHandler = {
  handle(message: InboundMessage): Promise<OutboundMessage>;
};

export type AgentRequest = {
  readonly user: ChannelUser;
  readonly text: string;
  readonly workspacePath: string;
  readonly history?: readonly AgentConversationMessage[];
  readonly sessionId?: string;
  readonly progress?: AgentProgressPublisher;
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

export type WorkspaceKind = "knowledge-base" | "source-repository";

export type KnowledgeWorkspace = {
  readonly kind: WorkspaceKind;
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
};

export type ChangeRequest = {
  readonly workspace: KnowledgeWorkspace;
  readonly summary: string;
};

export type ChangePublication = {
  readonly workspaceId: string;
  readonly branch: string;
  readonly url?: string;
};

export type ChangePublisher = {
  publish(requests: readonly ChangeRequest[]): Promise<readonly ChangePublication[]>;
};
