import { describe, expect, it } from "vitest";
import { StaticAuthorizationService, mapRoleProvider } from "./staticAuthorizationService.js";

const viewerUser = { id: "viewer-1" };
const developerUser = { id: "dev-1" };
const workspace = { kind: "knowledge-base" as const, id: "kb", path: "/kb" };

describe("StaticAuthorizationService", () => {
  it("allows read for viewers", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(viewerUser, "read", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows suggest for viewers", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(viewerUser, "suggest", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("allows mutate for developers", async () => {
    const roles = new Map([["dev-1", "developer" as const]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));
    const decision = await service.can(developerUser, "mutate", workspace);

    expect(decision).toEqual({ allowed: true });
  });

  it("denies mutate for viewers with a Chinese reason", async () => {
    const service = new StaticAuthorizationService();
    const decision = await service.can(viewerUser, "mutate", workspace);

    expect(decision.allowed).toBe(false);
    expect(decision.reason).toContain("普通用户权限");
  });

  it("resolves role from the role provider", async () => {
    const roles = new Map([["dev-1", "developer" as const]]);
    const service = new StaticAuthorizationService(mapRoleProvider(roles));

    await expect(service.roleFor(developerUser)).resolves.toBe("developer");
    await expect(service.roleFor(viewerUser)).resolves.toBe("viewer");
  });
});

describe("mapRoleProvider", () => {
  it("returns stored roles for known users", () => {
    const roles = new Map([["user-1", "developer" as const]]);
    const provider = mapRoleProvider(roles);

    expect(provider.getRole("user-1")).toBe("developer");
  });

  it("defaults to viewer for unknown users", () => {
    const provider = mapRoleProvider(new Map());

    expect(provider.getRole("unknown")).toBe("viewer");
  });
});
