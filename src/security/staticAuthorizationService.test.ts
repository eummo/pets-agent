import { describe, expect, it } from "vitest";
import { StaticAuthorizationService, mapRoleProvider } from "./staticAuthorizationService.js";
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
      mapRoleProvider(new Map()),
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "read", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows suggest for reviewers with workspace_read capability", async () => {
    const service = new StaticAuthorizationService(
      mapRoleProvider(new Map()),
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "suggest", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows mutate for developers with workspace_mutate capability", async () => {
    const roles = new Map([["dev-1", "developer"]]);
    const service = new StaticAuthorizationService(
      mapRoleProvider(roles),
      makeRoleConfigStore({ developer: ["workspace_read", "workspace_mutate"] }),
    );
    const decision = await service.can(developerUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for reviewers without workspace_mutate capability", async () => {
    const service = new StaticAuthorizationService(
      mapRoleProvider(new Map()),
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );
    const decision = await service.can(reviewerUser, "mutate", workspace);

    expect(decision.allowed).toBe(false);
    expect(decision.reason).toBe("Insufficient permissions for this action.");
  });

  it("allows mutate for admin with workspace_mutate capability", async () => {
    const roles = new Map([["admin-1", "admin"]]);
    const service = new StaticAuthorizationService(
      mapRoleProvider(roles),
      makeRoleConfigStore({ admin: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"] }),
    );
    const decision = await service.can(adminUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("resolves role from the role provider", async () => {
    const roles = new Map([["dev-1", "developer"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));

    await expect(service.roleFor(developerUser)).resolves.toBe("developer");
    await expect(service.roleFor(reviewerUser)).resolves.toBe("reviewer");
  });

  it("maps viewer to reviewer", async () => {
    const roles = new Map([["viewer-1", "viewer"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const viewerUser = { id: "viewer-1" };

    await expect(service.roleFor(viewerUser)).resolves.toBe("reviewer");
  });

  it("maps unknown roles to reviewer", async () => {
    const roles = new Map([["custom-1", "custom-role"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const customUser = { id: "custom-1" };

    await expect(service.roleFor(customUser)).resolves.toBe("custom-role");
  });
});

describe("hasCapability", () => {
  it("returns true when role has the capability", async () => {
    const roles = new Map([["admin-1", "admin"]]);
    const service = new StaticAuthorizationService(
      mapRoleProvider(roles),
      makeRoleConfigStore({ admin: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"] }),
    );

    await expect(service.hasCapability(adminUser, "feedback_manage")).resolves.toBe(true);
    await expect(service.hasCapability(adminUser, "feedback_view")).resolves.toBe(true);
    await expect(service.hasCapability(adminUser, "workspace_read")).resolves.toBe(true);
  });

  it("returns false when role lacks the capability", async () => {
    const roles = new Map([["dev-1", "developer"]]);
    const service = new StaticAuthorizationService(
      mapRoleProvider(roles),
      makeRoleConfigStore({ developer: ["workspace_read", "workspace_mutate"] }),
    );

    await expect(service.hasCapability(developerUser, "feedback_manage")).resolves.toBe(false);
    await expect(service.hasCapability(developerUser, "feedback_view")).resolves.toBe(false);
  });

  it("returns false for reviewer without feedback capabilities", async () => {
    const service = new StaticAuthorizationService(
      mapRoleProvider(new Map()),
      makeRoleConfigStore({ reviewer: ["workspace_read"] }),
    );

    await expect(service.hasCapability(reviewerUser, "feedback_manage")).resolves.toBe(false);
    await expect(service.hasCapability(reviewerUser, "feedback_view")).resolves.toBe(false);
  });
});

describe("backwards compatibility (no explicit capabilities)", () => {
  it("allows mutate for custom roles that have mutating tools", async () => {
    const roles = new Map([["builder-1", "builder"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles), {
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
    });

    const decision = await service.can({ id: "builder-1" }, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for custom roles without mutating tools", async () => {
    const roles = new Map([["reader-1", "reader"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles), {
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
    });

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });

  it("denies mutate for read-only roles that can use Bash", async () => {
    const roles = new Map([["reader-1", "reader"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles), {
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
    });

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });

  it("denies mutate for custom roles with mutating tools but read-only permission mode", async () => {
    const roles = new Map([["reader-1", "reader"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles), {
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
    });

    const decision = await service.can({ id: "reader-1" }, "mutate", workspace);

    expect(decision.allowed).toBe(false);
  });
});

describe("mapRoleProvider", () => {
  it("returns stored roles for known users", () => {
    const roles = new Map([["user-1", "developer"]]);
    const provider = mapRoleProvider(roles);

    expect(provider.getRole("user-1")).toBe("developer");
  });

  it("defaults to reviewer for unknown users", () => {
    const provider = mapRoleProvider(new Map());

    expect(provider.getRole("unknown")).toBe("reviewer");
  });
});
