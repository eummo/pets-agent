import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService,
  ChannelUser,
  KnowledgeWorkspace,
  UserRole
} from "../core/ports.js";

export type RoleProvider = {
  getRole(userId: string): UserRole;
};

export class StaticAuthorizationService implements AuthorizationService {
  public constructor(private readonly roleProvider: RoleProvider = mapRoleProvider(new Map())) {}

  public roleFor(user: ChannelUser): Promise<UserRole> {
    return Promise.resolve(this.roleProvider.getRole(user.id));
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

    if (role === "developer") {
      return { allowed: true };
    }

    return {
      allowed: false,
      reason: "我已识别到这是修改请求，但你当前是普通用户权限，只能查看知识库或生成建议，不能直接修改文件。"
    };
  }
}

export function mapRoleProvider(roles: ReadonlyMap<string, UserRole>): RoleProvider {
  return {
    getRole(userId) {
      return roles.get(userId) ?? "viewer";
    }
  };
}
