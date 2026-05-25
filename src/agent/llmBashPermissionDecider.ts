import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { Api, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import { withRetry } from "../config/retry.js";
import type { StoredRoleConfig } from "../core/ports.js";
import type { ToolPermissionDecider } from "./claudeSdkAgentRuntime.js";

const BASH_PERMISSION_SYSTEM_PROMPT = `You are a Bash command permission classifier.
Decide whether a Bash command is read-only inspection.

Return exactly one label:
- "allow": the command only reads/inspects local state and should not modify files, processes, network state, package state, git state, permissions, or databases.
- "deny": the command may write, delete, move, install, start services, stop processes, change git state, change permissions, make network requests with side effects, or is ambiguous.

Do not explain. Return only "allow" or "deny".`;

const BASH_PERMISSION_TIMEOUT_MS = 5000;

export class LlmBashPermissionDecider {
  public constructor(
    private readonly model: Model<Api>,
    private readonly apiKey: string,
  ) {}

  public readonly decide: ToolPermissionDecider = async (
    roleConfig: StoredRoleConfig,
    toolName: string,
    input: Record<string, unknown>,
  ) => {
    if (toolName !== "Bash") {
      return deny(toolName, "Only Bash commands are classified by this decider.");
    }

    const command = input["command"];
    if (typeof command !== "string" || command.trim().length === 0) {
      return deny(toolName, "Bash command is missing.");
    }

    return await this.classify(roleConfig.name, command);
  };

  private async classify(roleName: string, command: string): Promise<PermissionResult> {
    try {
      const response = await withRetry(async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), BASH_PERMISSION_TIMEOUT_MS);

        return complete(this.model, {
          systemPrompt: BASH_PERMISSION_SYSTEM_PROMPT,
          messages: [{
            role: "user",
            content: `Role: ${roleName}\nCommand: ${command}`,
            timestamp: Date.now(),
          }],
        }, {
          apiKey: this.apiKey,
          signal: controller.signal,
        }).finally(() => clearTimeout(timeout));
      });

      if (response.stopReason === "error") {
        return deny("Bash", "Bash permission classifier failed.");
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();

      return label === "allow"
        ? { behavior: "allow", decisionClassification: "user_temporary" }
        : deny("Bash", "Bash command is not read-only.");
    } catch {
      return deny("Bash", "Bash permission classifier failed.");
    }
  }
}

function deny(toolName: string, message: string): PermissionResult {
  return {
    behavior: "deny",
    message: `Tool ${toolName} denied: ${message}`,
    decisionClassification: "user_reject",
  };
}
