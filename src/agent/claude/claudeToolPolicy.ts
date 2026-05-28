import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type {
  ToolPermissionResult,
  ToolPermissionDecider as NeutralToolPermissionDecider
} from "../policy/toolPolicy.js";

// Re-export provider-neutral types and functions for backward compatibility.
// New code should import directly from toolPolicy.ts.
export type ToolPermissionDecider = NeutralToolPermissionDecider;
export {
  availableToolsForRole,
  autoAllowedToolsForRole,
  disallowedToolsForRole,
  canUseConfiguredTool,
  isToolInputWithinWorkspace,
  roleCanUseFileMutationTools,
  decideToolPermission
} from "../policy/toolPolicy.js";
export type { ToolPermissionResult } from "../policy/toolPolicy.js";

/**
 * Maps a provider-neutral ToolPermissionResult to the Claude SDK's PermissionResult.
 * Use this at the Claude adapter boundary where the SDK expects its own type.
 */
export function toClaudePermissionResult(result: ToolPermissionResult): PermissionResult {
  if (result.behavior === "deny") {
    return {
      behavior: "deny",
      message: result.message ?? "Permission denied.",
      ...(result.decisionClassification !== undefined
        ? { decisionClassification: result.decisionClassification }
        : {})
    };
  }
  return {
    behavior: "allow",
    ...(result.decisionClassification !== undefined
      ? { decisionClassification: result.decisionClassification }
      : {})
  };
}
