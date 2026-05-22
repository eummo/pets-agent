export type DevRoleStore = {
  getRole(userId: string): string;
  setRole(userId: string, role: string): void;
};

export function createDevRoleStore(initialRoles: ReadonlyMap<string, string> = new Map()): DevRoleStore {
  const roles = new Map(initialRoles);

  return {
    getRole(userId) {
      return roles.get(userId) ?? "reviewer";
    },
    setRole(userId, role) {
      roles.set(userId, role);
    }
  };
}
