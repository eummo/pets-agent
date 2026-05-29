import type { ChannelUser } from "../core/index.js";
import type { KnowledgeWorkspace } from "../workspace/index.js";

export type AuthorizationAction = "read" | "suggest" | "mutate" | "update_kb";

export type AuthorizationDecision = {
  readonly allowed: boolean;
  readonly reason?: string;
};

export type AuthorizationService = {
  roleFor(user: ChannelUser): Promise<string>;
  can(user: ChannelUser, action: AuthorizationAction, workspace: KnowledgeWorkspace): Promise<AuthorizationDecision>;
  canRole?(role: string, action: AuthorizationAction, workspace: KnowledgeWorkspace): Promise<AuthorizationDecision>;
  hasCapability(user: ChannelUser, capability: RoleCapability): Promise<boolean>;
  setRole?(userId: string, role: string): void;
};

// ── Role Capabilities ─────────────────────────────────────────────────────────
// Each capability is an independent, composable unit that a role can possess.
// Adding a new capability only requires extending this union and assigning it
// to the desired roles in the database — no code changes needed elsewhere.

export type RoleCapability =
  | "workspace_read"       // browse and read workspace content
  | "workspace_mutate"     // modify files in the workspace
  | "knowledge_base_update" // update curated knowledge-base documentation
  | "feedback_view"        // view user feedback entries
  | "feedback_manage"      // review and update feedback status
  | "roles_manage"         // create, update, delete role configurations (future)
  | "cron_manage";         // create, update, delete, and view cron jobs

export type SettingSource = "user" | "project" | "local";

export type StoredRoleConfig = {
  readonly name: string;
  readonly systemPrompt: string;
  readonly allowedTools: readonly string[];
  readonly permissionMode: "auto" | "dontAsk" | "acceptEdits" | "bypassPermissions";
  readonly maxTurns?: number;
  readonly model?: string;
  readonly capabilities?: readonly RoleCapability[];
  readonly skills?: string[] | "all";
  readonly settingSources?: readonly SettingSource[];
  readonly updatedAt?: string;
};

export type MutationToolName = "Edit" | "MultiEdit" | "NotebookEdit" | "Write";

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
