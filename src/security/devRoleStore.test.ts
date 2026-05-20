import { describe, expect, it } from "vitest";
import { createDevRoleStore } from "./devRoleStore.js";

describe("createDevRoleStore", () => {
  it("returns viewer for unknown users by default", () => {
    const store = createDevRoleStore();

    expect(store.getRole("unknown")).toBe("viewer");
  });

  it("returns initial roles from the provided map", () => {
    const store = createDevRoleStore(new Map([["dev-1", "developer"]]));

    expect(store.getRole("dev-1")).toBe("developer");
    expect(store.getRole("unknown")).toBe("viewer");
  });

  it("allows setting and getting roles", () => {
    const store = createDevRoleStore();

    store.setRole("user-1", "developer");

    expect(store.getRole("user-1")).toBe("developer");
  });

  it("allows overwriting existing roles", () => {
    const store = createDevRoleStore(new Map([["user-1", "viewer"]]));

    store.setRole("user-1", "developer");

    expect(store.getRole("user-1")).toBe("developer");
  });
});
