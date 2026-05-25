import { describe, expect, it } from "vitest";
import { StaticAuthorizationService } from "./staticAuthorizationService.js";
import type { RoleCapability, RoleConfigStore } from "../core/ports.js";

const reviewerUser = { id: "reviewer-1" };
const developerUser = { id: "dev-1" };
const adminUser = { id: "admin-1" };
const workspace = { kind: "knowledge-base" as const, id: "kb", path: "/kb" };

function makeRoleConfigStore(roles: Record<string, readonly RoleCapability[]>): RoleConfigStore {
  return {
    getAll() { return Promise.resolve([]); },
    getByName(name) {
      const caps = roles[name];
      return Promise.resolve(caps === undefined ? undefined : {
        name,
        systemPrompt: `Prompt for ${name}`,
        allowedTools: ["Read"],
        permissionMode: "dontAsk" as const,
        capabilities: caps,
      });
    },
    upsert() { return Promise.resolve(); },
    deleteByName() { return Promise.resolve(false); },
  };
}

describe("StaticAuthorizationService", () => {
  it("allows read for reviewers with workspace_read capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "read", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows suggest for reviewers with workspace_read capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "suggest", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows mutate for developers with workspace_mutate capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ developer: ["workspace_read", "workspace_mutate"] }),
      new Map([["dev-1", "developer"]]),
    );
    const decision = await service.can(developerUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for reviewers without workspace_mutate capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "mutate", workspace);

    expect(decision.allowed).toBe(false);
    expect(decision.reason).toBe("Insufficient permissions for this action.");
  });

  it("allows mutate for admin with workspace_mutate capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ admin: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"] }),
      new Map([["admin-1", "admin"]]),
    );
    const decision = await service.can(adminUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("resolves role from initial roles", async () => {
    const service = new StaticAuthorizationService(undefined, new Map([["dev-1", "developer"]]));

    await expect(service.roleFor(developerUser)).resolves.toBe("developer");
    await expect(service.roleFor(reviewerUser)).resolves.toBe("reviewer");
  });

  it("maps viewer to reviewer", async () => {
    const service = new StaticAuthorizationService(undefined, new Map([["viewer-1", "viewer"]]));
    const viewerUser = { id: "viewer-1" };

    await expect(service.roleFor(viewerUser)).resolves.toBe("reviewer");
  });

  it("maps unknown roles to reviewer", async () => {
    const service = new StaticAuthorizationService(undefined, new Map([["custom-1", "custom-role"]]));
    const customUser = { id: "custom-1" };

    await expect(service.roleFor(customUser)).resolves.toBe("custom-role");
  });

  it("defaults to reviewer for unknown users", async () => {
    const service = new StaticAuthorizationService();

    await expect(service.roleFor({ id: "unknown" })).resolves.toBe("reviewer");
  });
});

describe("setRole", () => {
  it("sets a role for a user", async () => {
    const service = new StaticAuthorizationService();

    service.setRole("user-1", "developer");
    await expect(service.roleFor({ id: "user-1" })).resolves.toBe("developer");
  });

  it("overwrites an existing role", async () => {
    const service = new StaticAuthorizationService(undefined, new Map([["user-1", "reviewer"]]));

    service.setRole("user-1", "developer");
    await expect(service.roleFor({ id: "user-1" })).resolves.toBe("developer");
  });

  it("stores arbitrary role strings", async () => {
    const service = new StaticAuthorizationService();

    service.setRole("user-1", "custom-role");
    await expect(service.roleFor({ id: "user-1" })).resolves.toBe("custom-role");
  });

  it("returns initial roles from the constructor map", async () => {
    const service = new StaticAuthorizationService(undefined, new Map([["dev-1", "developer"]]));

    await expect(service.roleFor({ id: "dev-1" })).resolves.toBe("developer");
    await expect(service.roleFor({ id: "unknown" })).resolves.toBe("reviewer");
  });
});

describe("hasCapability", () => {
  it("returns true when role has the capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ admin: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"] }),
      new Map([["admin-1", "admin"]]),
    );

    await expect(service.hasCapability(adminUser, "feedback_manage")).resolves.toBe(true);
    await expect(service.hasCapability(adminUser, "feedback_view")).resolves.toBe(true);
    await expect(service.hasCapability(adminUser, "workspace_read")).resolves.toBe(true);
  });

  it("returns false when role lacks the capability", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ developer: ["workspace_read", "workspace_mutate"] }),
      new Map([["dev-1", "developer"]]),
    );

    await expect(service.hasCapability(developerUser, "feedback_manage")).resolves.toBe(false);
    await expect(service.hasCapability(developerUser, "feedback_view")).resolves.toBe(false);
  });

  it("returns false for reviewer without feedback capabilities", async () => {
    const service = new StaticAuthorizationService(
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );

    await expect(service.hasCapability(reviewerUser, "feedback_manage")).resolves.toBe(false);
    await expect(service.hasCapability(reviewerUser, "feedback_view")).resolves.toBe(false);
  });
});

describe("backwards compatibility (no explicit capabilities)", () => {
  it("allows mutate for custom roles that have mutating tools", async () => {
    const service = new StaticAuthorizationService({
      getAll() {
        return Promise.resolve([]);
      },
      getByName(name) {
        return Promise.resolve({
          name,
          systemPrompt: "You can edit.",
          allowedTools: ["Read", "Edit"],
          permissionMode: "acceptEdits",
        });
      },
      upsert() {
        return Promise.resolve();
      },
      deleteByName() {
        return Promise.resolve(false);
      }
    }, new Map([["builder-1", "builder"]]));

    const decision = await service.can({ id: "builder-1" }, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for custom roles without mutating tools", async () => {
    const service = new StaticAuthorizationService({
      getAll() {
        return Promise.resolve([]);
      },
      getByName(name) {
        return Promise.resolve({
          name,
          systemPrompt: "You can read.",
          allowedTools: ["Read", "Grep"],
          permissionMode: "dontAsk",
        });
      },
      upsert() {
        return Promise.resolve();
      },
      deleteByName() {
        return Promise.resolve(false);
      }
    }, new Map([["reader-1", "reader"]]));

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });

  it("denies mutate for read-only roles that can use Bash", async () => {
    const service = new StaticAuthorizationService({
      getAll() {
        return Promise.resolve([]);
      },
      getByName(name) {
        return Promise.resolve({
          name,
          systemPrompt: "You can inspect.",
          allowedTools: ["Read", "Grep", "Bash"],
          permissionMode: "auto",
        });
      },
      upsert() {
        return Promise.resolve();
      },
      deleteByName() {
        return Promise.resolve(false);
      }
    }, new Map([["reader-1", "reader"]]));

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });

  it("denies mutate for custom roles with mutating tools but read-only permission mode", async () => {
    const service = new StaticAuthorizationService({
      getAll() {
        return Promise.resolve([]);
      },
      getByName(name) {
        return Promise.resolve({
          name,
          systemPrompt: "You can read.",
          allowedTools: ["Read", "Edit"],
          permissionMode: "dontAsk",
        });
      },
      upsert() {
        return Promise.resolve();
      },
      deleteByName() {
        return Promise.resolve(false);
      }
    }, new Map([["reader-1", "reader"]]));

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });
});
