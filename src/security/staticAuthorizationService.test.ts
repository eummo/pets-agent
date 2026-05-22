import { describe, expect, it } from "vitest";
import { StaticAuthorizationService, mapRoleProvider } from "./staticAuthorizationService.js";

const reviewerUser = { id: "reviewer-1" };
const developerUser = { id: "dev-1" };
const workspace = { kind: "knowledge-base" as const, id: "kb", path: "/kb" };

describe("StaticAuthorizationService", () => {
  it("allows read for reviewers", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(reviewerUser, "read", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows suggest for reviewers", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(reviewerUser, "suggest", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows mutate for developers", async () => {
    const roles = new Map([["dev-1", "developer"]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const decision = await service.can(developerUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for reviewers with a generic reason", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(reviewerUser, "mutate", workspace);

    expect(decision.allowed).toBe(false);
    expect(decision.reason).toBe("Insufficient permissions for this action.");
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
