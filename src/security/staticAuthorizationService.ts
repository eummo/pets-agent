import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService,
  ChannelUser,
  KnowledgeWorkspace,
  RoleCapability,
  RoleConfigStore,
  StoredRoleConfig,
  UserRole
} from "../core/ports.js";
import { FILE_MUTATION_TOOLS } from "../core/ports.js";

export type RoleProvider = {
  getRole(userId: string): string;
};

// Maps AuthorizationAction to the required RoleCapability
const ACTION_CAPABILITY_MAP: Record<AuthorizationAction, RoleCapability> = {
  read: "workspace_read",
  suggest: "workspace_read",
  mutate: "workspace_mutate",
};

export class StaticAuthorizationService implements AuthorizationService {
  public constructor(
    private readonly roleProvider: RoleProvider = mapRoleProvider(new Map()),
    private readonly roleConfigStore?: RoleConfigStore,
  ) {}

  public roleFor(user: ChannelUser): Promise<UserRole> {
    const role = this.roleProvider.getRole(user.id);
    return Promise.resolve(normalizeRoleName(role));
  }

  public async can(
    user: ChannelUser,
    action: AuthorizationAction,
    workspace: KnowledgeWorkspace
  ): Promise<AuthorizationDecision> {
    void workspace;

    const required = ACTION_CAPABILITY_MAP[action];
    const hasIt = await this.hasCapability(user, required);

    if (hasIt) {
      return { allowed: true };
    }

    return {
      allowed: false,
      reason: "Insufficient permissions for this action."
    };
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
      return ["workspace_read", "workspace_mutate"];
    }

    return ["workspace_read"];
  }
}

export function mapRoleProvider(roles: ReadonlyMap<string, string>): RoleProvider {
  return {
    getRole(userId) {
      return roles.get(userId) ?? "reviewer";
    }
  };
}

function normalizeRoleName(role: string): UserRole {
  return role === "viewer" ? "reviewer" : role;
}

function configAllowsMutation(config: StoredRoleConfig): boolean {
  if (config.permissionMode !== "acceptEdits" && config.permissionMode !== "bypassPermissions") {
    return false;
  }

  return config.allowedTools.some((tool) => FILE_MUTATION_TOOLS.has(tool));
}
