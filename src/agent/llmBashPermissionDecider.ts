import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";
import type { RoleConfig, ToolPermissionDecider } from "./claudeSdkAgentRuntime.js";

const BASH_PERMISSION_PROMPT = `You are a Bash command permission classifier.
Decide whether a Bash command is read-only inspection.

Return exactly one label:
- "allow": the command only reads/inspects local state and should not modify files, processes, network state, package state, git state, permissions, or databases.
- "deny": the command may write, delete, move, install, start services, stop processes, change git state, change permissions, make network requests with side effects, or is ambiguous.

Do not explain. Return only "allow" or "deny".

Role: {role}
Command: {command}`;

const BASH_PERMISSION_TIMEOUT_MS = 5000;
const BASH_PERMISSION_MAX_TOKENS = 128;

export class LlmBashPermissionDecider {
  public constructor(private readonly config: ResolvedLlmConfig) {}

  public readonly decide: ToolPermissionDecider = async (
    roleConfig: RoleConfig,
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
      const prompt = BASH_PERMISSION_PROMPT
        .replace("{role}", roleName)
        .replace("{command}", command);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), BASH_PERMISSION_TIMEOUT_MS);

      const baseUrl = this.config.baseUrl.replace(/\/+$/, "");
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": this.config.apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: this.config.modelId,
          max_tokens: BASH_PERMISSION_MAX_TOKENS,
          messages: [{ role: "user", content: prompt }],
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        return deny("Bash", "Bash permission classifier failed.");
      }

      const data = await response.json() as Record<string, unknown>;
      const content = data["content"] as Record<string, unknown>[] | undefined;
      const text = content
        ?.map((block) => block["text"])
        .find((value): value is string => typeof value === "string");
      const label = text?.trim().toLowerCase() ?? "";

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
