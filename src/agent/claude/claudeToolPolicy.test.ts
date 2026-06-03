import { describe, expect, it } from "vitest";
import type { StoredRoleConfig } from "../../auth/index.js";
import {
  autoAllowedToolsForRole,
  decideToolPermission,
  isToolInputWithinWorkspace
} from "./claudeToolPolicy.js";

const reviewerConfig: StoredRoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Glob", "Grep", "Bash"],
  permissionMode: "dontAsk",
  systemPrompt: "Read only."
};
const developerConfig: StoredRoleConfig = {
  name: "developer",
  allowedTools: ["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
  permissionMode: "bypassPermissions",
  systemPrompt: "Edit selected workspace files."
};

describe("claudeToolPolicy", () => {
  it("does not auto-allow read tools for read-only roles", () => {
    expect(autoAllowedToolsForRole(reviewerConfig)).toEqual([]);
  });

  it("allows read-only tool paths inside the selected workspace", () => {
    expect(isToolInputWithinWorkspace("Read", { file_path: "D:/kb/docs/order.md" }, "D:/kb")).toBe(
      true
    );
    expect(isToolInputWithinWorkspace("Grep", { path: "D:/kb/docs" }, "D:/kb")).toBe(true);
    expect(isToolInputWithinWorkspace("Glob", { path: "docs" }, "D:/kb")).toBe(true);
  });

  it("denies read-only tool paths outside the selected workspace", async () => {
    const result = await decideToolPermission(
      reviewerConfig,
      "Read",
      { file_path: "D:/code/pets-agent/src/core/contracts.ts" },
      undefined,
      "D:/code/pets-agent/.harness/knowledge-base"
    );

    expect(result.behavior).toBe("deny");
    if (result.behavior === "deny") {
      expect(result.message).toContain("outside the selected workspace");
    }
  });

  it("denies mutation tool paths outside the selected workspace", async () => {
    const result = await decideToolPermission(
      developerConfig,
      "Edit",
      { file_path: "D:/code/pets-agent/src/index.ts" },
      undefined,
      "D:/code/pets-agent/.harness/knowledge-base"
    );

    expect(result.behavior).toBe("deny");
    if (result.behavior === "deny") {
      expect(result.message).toContain("outside the selected workspace");
    }
  });

  it("denies bash commands that reference paths outside the selected workspace", async () => {
    const result = await decideToolPermission(
      developerConfig,
      "Bash",
      { command: "cd \"D:\\code\\pets-agent\" && npm run check" },
      undefined,
      "D:/code/pets-agent/.harness/knowledge-base"
    );

    expect(result.behavior).toBe("deny");
    if (result.behavior === "deny") {
      expect(result.message).toContain("outside the selected workspace");
    }
  });

  it("allows WebSearch when role has web_access capability", async () => {
    const webConfig: StoredRoleConfig = {
      name: "reviewer",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
      systemPrompt: "Read only.",
      capabilities: ["workspace_read", "web_access"]
    };
    const result = await decideToolPermission(
      webConfig,
      "WebSearch",
      { query: "latest Node.js LTS version" },
      undefined,
      "D:/kb"
    );

    expect(result.behavior).toBe("allow");
  });

  it("allows WebFetch when role has web_access capability", async () => {
    const webConfig: StoredRoleConfig = {
      name: "developer",
      allowedTools: ["Read"],
      permissionMode: "bypassPermissions",
      systemPrompt: "Edit.",
      capabilities: ["workspace_read", "workspace_mutate", "web_access"]
    };
    const result = await decideToolPermission(
      webConfig,
      "WebFetch",
      { url: "https://nodejs.org/en", prompt: "What is the latest LTS version?" },
      undefined,
      "D:/kb"
    );

    expect(result.behavior).toBe("allow");
  });

  it("denies WebSearch when role lacks web_access capability", async () => {
    const noWebConfig: StoredRoleConfig = {
      name: "reviewer",
      allowedTools: ["Read"],
      permissionMode: "dontAsk",
      systemPrompt: "No web."
    };
    const result = await decideToolPermission(
      noWebConfig,
      "WebSearch",
      { query: "test" },
      undefined,
      "D:/kb"
    );

    expect(result.behavior).toBe("deny");
  });
});
