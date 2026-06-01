import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type {
  ToolPermissionResult,
  ToolPermissionDecider as NeutralToolPermissionDecider
} from "../../auth/index.js";

export type ToolPermissionDecider = NeutralToolPermissionDecider;
export {
  availableToolsForRole,
  autoAllowedToolsForRole,
  disallowedToolsForRole,
  canUseConfiguredTool,
  isToolInputWithinWorkspace,
  roleCanUseFileMutationTools,
  decideToolPermission
} from "../../auth/index.js";
export type { ToolPermissionResult } from "../../auth/index.js";

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
