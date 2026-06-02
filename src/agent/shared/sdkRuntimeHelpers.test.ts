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

  it("adds a pre-tool workspace guard hook that denies outside paths", async () => {
    const options = buildSdkQueryOptions({
      ...baseInput,
      roleConfig: {
        ...baseRoleConfig,
        allowedTools: ["Read", "Edit", "Write", "Bash"],
        permissionMode: "bypassPermissions"
      }
    });

    const hook = workspaceGuardHookFromOptions(options);
    const result = await hook({
      hook_event_name: "PreToolUse",
      tool_name: "Edit",
      tool_input: { file_path: "/outside/file.ts" }
    });

    expect(result).toEqual({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "Tool Edit path is outside the selected workspace."
      }
    });
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

  it("omits endpoint from serialized logs", () => {
    const options = {
      cwd: "/test",
      endpoint: "https://enterprise.example.com/",
      environment: "internal"
    };
    const serialized = serializeQueryOptions(options);
    expect(serialized["endpoint"]).toBeUndefined();
    expect(serialized["environment"]).toBe("internal");
  });
});

type HookEntry = {
  readonly hooks: readonly WorkspaceGuardHook[];
};

type WorkspaceGuardHook = (
  input: Record<string, unknown>
) => Record<string, unknown> | Promise<Record<string, unknown>>;

function workspaceGuardHookFromOptions(options: Record<string, unknown>): WorkspaceGuardHook {
  const hooks = options["hooks"] as Record<string, readonly HookEntry[]>;
  const preToolUseHooks = hooks["PreToolUse"];
  const hook = preToolUseHooks?.[0]?.hooks[0];
  if (hook === undefined) {
    throw new Error("Expected PreToolUse workspace guard hook.");
  }

  return hook;
}
