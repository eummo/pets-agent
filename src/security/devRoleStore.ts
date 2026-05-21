import type { UserRole } from "../core/ports.js";

export type DevRoleStore = {
  getRole(userId: string): UserRole;
  setRole(userId: string, role: UserRole): void;
};

export function createDevRoleStore(initialRoles: ReadonlyMap<string, UserRole> = new Map()): DevRoleStore {
  const roles = new Map(initialRoles);

  return {
    getRole(userId) {
      const role = roles.get(userId) ?? "reviewer";
      return role === "viewer" ? "reviewer" : role;
    },
    setRole(userId, role) {
      roles.set(userId, role === "viewer" ? "reviewer" : role);
    }
  };
}
