import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService,
  ChannelUser,
  KnowledgeWorkspace,
  RoleConfigStore,
  StoredRoleConfig,
  UserRole
} from "../core/ports.js";

export type RoleProvider = {
  getRole(userId: string): string;
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

    const role = await this.roleFor(user);

    if (action === "read" || action === "suggest") {
      return { allowed: true };
    }

    if (await this.roleCanMutate(role)) {
      return { allowed: true };
    }

    return {
      allowed: false,
      reason: "Insufficient permissions for this action."
    };
  }

  private async roleCanMutate(role: UserRole): Promise<boolean> {
    if (role === "developer") {
      return true;
    }

    const config = await this.roleConfigStore?.getByName(role);
    return config === undefined ? false : configAllowsMutation(config);
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
  const mutatingTools = new Set(["Bash", "Edit", "MultiEdit", "NotebookEdit", "Write"]);
  return config.allowedTools.some((tool) => mutatingTools.has(tool));
}
