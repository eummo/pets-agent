import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { StoredRoleConfig } from "../core/contracts.js";
import { FILE_MUTATION_TOOLS } from "../core/contracts.js";

export type ToolPermissionDecider = (
  roleConfig: StoredRoleConfig,
  toolName: string,
  input: Record<string, unknown>,
) => Promise<PermissionResult>;

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

  return config.allowedTools.filter((tool) => tool !== "Bash" && !FILE_MUTATION_TOOLS.has(tool));
}

export function disallowedToolsForRole(config: StoredRoleConfig): readonly string[] {
  if (roleCanUseFileMutationTools(config)) {
    return [];
  }

  return [...FILE_MUTATION_TOOLS].filter((tool) => config.allowedTools.includes(tool));
}

export async function decideToolPermission(
  config: StoredRoleConfig,
  toolName: string,
  input: Record<string, unknown>,
  toolPermissionDecider: ToolPermissionDecider | undefined,
): Promise<PermissionResult> {
  if (!canUseConfiguredTool(config, toolName)) {
    return denyTool(config.name, toolName);
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

function roleCanUseFileMutationTools(config: StoredRoleConfig): boolean {
  if (config.permissionMode !== "acceptEdits" && config.permissionMode !== "bypassPermissions") {
    return false;
  }

  return config.allowedTools.some((tool) => FILE_MUTATION_TOOLS.has(tool));
}

function denyTool(roleName: string, toolName: string): PermissionResult {
  return {
    behavior: "deny",
    message: `Tool ${toolName} is not permitted for role ${roleName}.`,
    decisionClassification: "user_reject",
  };
}
