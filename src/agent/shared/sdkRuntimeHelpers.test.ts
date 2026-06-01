import { describe, expect, it } from "vitest";
import {
  buildSdkQueryOptions,
  serializeQueryOptions,
  type SdkQueryOptionsInput
} from "./sdkRuntimeHelpers.js";
import type { StoredRoleConfig } from "../../auth/index.js";

const baseRoleConfig: StoredRoleConfig = {
  name: "reviewer",
  systemPrompt: "Test prompt",
  allowedTools: ["Read", "Glob"],
  permissionMode: "dontAsk"
};

const baseInput: Omit<SdkQueryOptionsInput, "roleConfig"> = {
  request: {
    text: "test",
    user: { id: "test-user" },
    workspacePath: "/test/workspace"
  },
  contextConfig: {
    autoCompactEnabled: false,
    autoCompactWindow: 80,
    workspaceMaxChars: 8_000,
    historyMaxMessages: 50
  },
  model: undefined,
  canUseTool: async () => Promise.resolve(undefined)
};

describe("buildSdkQueryOptions", () => {
  it("does not include enableWorkflows when undefined", () => {
    const options = buildSdkQueryOptions({ ...baseInput, roleConfig: baseRoleConfig });
    const settings = options["settings"] as Record<string, unknown> | undefined;
    expect(settings?.["enableWorkflows"]).toBeUndefined();
  });

  it("includes enableWorkflows in settings when true", () => {
    const options = buildSdkQueryOptions({
      ...baseInput,
      roleConfig: { ...baseRoleConfig, enableWorkflows: true }
    });
    const settings = options["settings"] as Record<string, unknown>;
    expect(settings).toBeDefined();
    expect(settings["enableWorkflows"]).toBe(true);
  });

  it("merges enableWorkflows with existing autoCompactEnabled settings", () => {
    const options = buildSdkQueryOptions({
      ...baseInput,
      roleConfig: { ...baseRoleConfig, enableWorkflows: true },
      contextConfig: {
        autoCompactEnabled: true,
        autoCompactWindow: 80,
        workspaceMaxChars: 8_000,
        historyMaxMessages: 50
      }
    });
    const settings = options["settings"] as Record<string, unknown>;
    expect(settings["autoCompactEnabled"]).toBe(true);
    expect(settings["enableWorkflows"]).toBe(true);
  });

  it("does not include planModeInstructions when undefined", () => {
    const options = buildSdkQueryOptions({ ...baseInput, roleConfig: baseRoleConfig });
    expect(options["planModeInstructions"]).toBeUndefined();
  });

  it("includes planModeInstructions as top-level option when set", () => {
    const instructions = "Read the codebase first, then propose changes.";
    const options = buildSdkQueryOptions({
      ...baseInput,
      roleConfig: { ...baseRoleConfig, planModeInstructions: instructions }
    });
    expect(options["planModeInstructions"]).toBe(instructions);
  });
});

describe("serializeQueryOptions", () => {
  it("includes planModeInstructions in serialized output", () => {
    const options = {
      cwd: "/test",
      planModeInstructions: "Test instructions"
    };
    const serialized = serializeQueryOptions(options);
    expect(serialized["planModeInstructions"]).toBe("Test instructions");
  });

  it("omits planModeInstructions when not present", () => {
    const options = { cwd: "/test" };
    const serialized = serializeQueryOptions(options);
    expect("planModeInstructions" in serialized).toBe(false);
  });
});
