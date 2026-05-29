import type { ChannelUser, UserRole } from "../core/index.js";
import type { KnowledgeWorkspace } from "../workspace/index.js";
import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService,
  RoleCapability,
  RoleConfigStore,
  StoredRoleConfig
} from "./index.js";
import { FILE_MUTATION_TOOLS } from "./index.js";

// Maps AuthorizationAction to the required RoleCapability
const ACTION_CAPABILITY_MAP: Record<AuthorizationAction, RoleCapability> = {
  read: "workspace_read",
  suggest: "workspace_read",
  mutate: "workspace_mutate",
  update_kb: "knowledge_base_update"
};

export class InMemoryRoleAuthorizationService implements AuthorizationService {
  private readonly roles: Map<string, string>;

  public constructor(
    private readonly roleConfigStore?: RoleConfigStore,
    initialRoles: ReadonlyMap<string, string> = new Map()
  ) {
    this.roles = new Map(initialRoles);
  }

  public roleFor(user: ChannelUser): Promise<UserRole> {
    const role = this.roles.get(user.id) ?? "reviewer";
    return Promise.resolve(normalizeRoleName(role));
  }

  public setRole(userId: string, role: string): void {
    this.roles.set(userId, role);
  }

  public async can(
    user: ChannelUser,
    action: AuthorizationAction,
    workspace: KnowledgeWorkspace
  ): Promise<AuthorizationDecision> {
    void workspace;

    const required = ACTION_CAPABILITY_MAP[action];
    const hasIt = await this.hasCapability(user, required);

    return decisionForCapability(hasIt);
  }

  public async canRole(
    role: string,
    action: AuthorizationAction,
    workspace: KnowledgeWorkspace
  ): Promise<AuthorizationDecision> {
    void workspace;

    const required = ACTION_CAPABILITY_MAP[action];
    const capabilities = await this.resolveCapabilities(normalizeRoleName(role));
    return decisionForCapability(capabilities.includes(required));
  }

  public async hasCapability(user: ChannelUser, capability: RoleCapability): Promise<boolean> {
    const role = await this.roleFor(user);
    const capabilities = await this.resolveCapabilities(role);
    return capabilities.includes(capability);
  }

  private async resolveCapabilities(role: UserRole): Promise<readonly RoleCapability[]> {
    const config = await this.roleConfigStore?.getByName(role);

    // If the role has explicit capabilities, use them
    if (config?.capabilities !== undefined && config.capabilities.length > 0) {
      return config.capabilities;
    }

    // Backwards compat: infer from allowedTools and permissionMode
    if (config !== undefined && configAllowsMutation(config)) {
      return ["workspace_read", "workspace_mutate", "knowledge_base_update"];
    }

    return ["workspace_read"];
  }
}

function normalizeRoleName(role: string): UserRole {
  return role === "viewer" ? "reviewer" : role;
}

function decisionForCapability(hasCapability: boolean): AuthorizationDecision {
  if (hasCapability) {
    return { allowed: true };
  }

  return {
    allowed: false,
    reason: "Insufficient permissions for this action."
  };
}

function configAllowsMutation(config: StoredRoleConfig): boolean {
  if (config.permissionMode !== "acceptEdits" && config.permissionMode !== "bypassPermissions") {
    return false;
  }

  return config.allowedTools.some((tool) => FILE_MUTATION_TOOLS.has(tool));
}
