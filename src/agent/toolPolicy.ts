import path from "node:path";
import type { StoredRoleConfig } from "../core/contracts.js";
import { FILE_MUTATION_TOOLS } from "../core/contracts.js";

// ── Provider-Neutral Permission Result ──────────────────────────────────────
// This type replaces the Claude SDK's PermissionResult so that tool policy
// logic can be reused by Codebuddy, Pi, or future adapters without importing
// any provider SDK types.

export type ToolPermissionResult = {
  readonly behavior: "allow" | "deny";
  readonly message?: string;
  readonly decisionClassification?: "user_temporary" | "user_permanent" | "user_reject";
};

export type ToolPermissionDecider = (
  roleConfig: StoredRoleConfig,
  toolName: string,
  input: Record<string, unknown>,
) => Promise<ToolPermissionResult>;

// ── Tool Availability ───────────────────────────────────────────────────────

export function availableToolsForRole(config: StoredRoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [...config.allowedTools];
  }

  return config.allowedTools.filter((tool) => !FILE_MUTATION_TOOLS.has(tool));
}

export function autoAllowedToolsForRole(config: StoredRoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [...config.allowedTools];
  }

  return [];
}

export function disallowedToolsForRole(config: StoredRoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [];
  }

  return [...FILE_MUTATION_TOOLS].filter((tool) => config.allowedTools.includes(tool));
}

// ── Permission Decision ─────────────────────────────────────────────────────

export async function decideToolPermission(
  config: StoredRoleConfig,
  toolName: string,
  input: Record<string, unknown>,
  toolPermissionDecider: ToolPermissionDecider | undefined,
  workspacePath: string,
): Promise<ToolPermissionResult> {
  if (!canUseConfiguredTool(config, toolName)) {
    return denyTool(config.name, toolName);
  }

  if (!roleCanUseFileMutationTools(config) && !isToolInputWithinWorkspace(toolName, input, workspacePath)) {
    return denyTool(config.name, toolName, `Tool ${toolName} path is outside the selected workspace.`);
  }

  if (toolName === "Bash" && !roleCanUseFileMutationTools(config)) {
    return toolPermissionDecider?.(config, toolName, input)
      ?? denyTool(config.name, toolName);
  }

  return { behavior: "allow" };
}

export function canUseConfiguredTool(config: StoredRoleConfig, toolName: string): boolean {
  if (!config.allowedTools.includes(toolName)) {
    return false;
  }

  return !FILE_MUTATION_TOOLS.has(toolName) || roleCanUseFileMutationTools(config);
}

export function isToolInputWithinWorkspace(toolName: string, input: Record<string, unknown>, workspacePath: string): boolean {
  const pathValue = pathValueForTool(toolName, input);
  if (pathValue === undefined || pathValue.trim().length === 0) {
    return true;
  }

  if (!path.isAbsolute(pathValue)) {
    return true;
  }

  const resolvedWorkspacePath = path.resolve(workspacePath);
  const resolvedToolPath = path.resolve(pathValue);
  const relativePath = path.relative(resolvedWorkspacePath, resolvedToolPath);
  return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

// ── Internal Helpers ────────────────────────────────────────────────────────

function pathValueForTool(toolName: string, input: Record<string, unknown>): string | undefined {
  if (toolName === "Read") {
    return stringField(input, "file_path");
  }

  if (toolName === "Grep" || toolName === "Glob") {
    return stringField(input, "path");
  }

  return undefined;
}

function stringField(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key];
  return typeof value === "string" ? value : undefined;
}

export function roleCanUseFileMutationTools(config: StoredRoleConfig): boolean {
  if (config.permissionMode !== "acceptEdits" && config.permissionMode !== "bypassPermissions") {
    return false;
  }

  return config.allowedTools.some((tool) => FILE_MUTATION_TOOLS.has(tool));
}

function denyTool(roleName: string, toolName: string, message?: string): ToolPermissionResult {
  return {
    behavior: "deny",
    message: message ?? `Tool ${toolName} is not permitted for role ${roleName}.`,
    decisionClassification: "user_reject",
  };
}
