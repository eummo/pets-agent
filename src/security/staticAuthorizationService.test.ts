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
    const roles = new Map([["dev-1", "developer" as const]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const decision = await service.can(developerUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for reviewers with a Chinese reason", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(reviewerUser, "mutate", workspace);

    expect(decision.allowed).toBe(false);
    expect(decision.reason).toContain("文档助手权限");
  });

  it("resolves role from the role provider", async () => {
    const roles = new Map([["dev-1", "developer" as const]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));

    await expect(service.roleFor(developerUser)).resolves.toBe("developer");
    await expect(service.roleFor(reviewerUser)).resolves.toBe("reviewer");
  });

  it("maps viewer to reviewer", async () => {
    const roles = new Map([["viewer-1", "viewer" as const]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const viewerUser = { id: "viewer-1" };

    await expect(service.roleFor(viewerUser)).resolves.toBe("reviewer");
  });
});

describe("mapRoleProvider", () => {
  it("returns stored roles for known users", () => {
    const roles = new Map([["user-1", "developer" as const]]);
    const provider = mapRoleProvider(roles);

    expect(provider.getRole("user-1")).toBe("developer");
  });

  it("defaults to reviewer for unknown users", () => {
    const provider = mapRoleProvider(new Map());

    expect(provider.getRole("unknown")).toBe("reviewer");
  });
});
