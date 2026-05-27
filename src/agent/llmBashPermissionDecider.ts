import type { Api, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import { withRetry } from "../config/retry.js";
import type { StoredRoleConfig } from "../core/contracts.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import type { ToolPermissionDecider, ToolPermissionResult } from "./toolPolicy.js";

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
    private readonly rawLogger?: JsonlLogger,
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

  private async classify(roleName: string, command: string): Promise<ToolPermissionResult> {
    const startTime = Date.now();
    const userContent = `Role: ${roleName}\nCommand: ${command}`;
    await this.rawLogger?.write({
      type: "llm.request",
      operation: "bash_permission",
      role: roleName,
      command,
      systemPrompt: BASH_PERMISSION_SYSTEM_PROMPT,
      messages: [{
        role: "user",
        content: userContent,
      }],
    });

    try {
      const response = await withRetry(async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), BASH_PERMISSION_TIMEOUT_MS);

        return complete(this.model, {
          systemPrompt: BASH_PERMISSION_SYSTEM_PROMPT,
          messages: [{
            role: "user",
            content: userContent,
            timestamp: Date.now(),
          }],
        }, {
          apiKey: this.apiKey,
          signal: controller.signal,
        }).finally(() => clearTimeout(timeout));
      });

      if (response.stopReason === "error") {
        const result = deny("Bash", "Bash permission classifier failed.");
        await this.logResponseAndDecision(roleName, command, response, result, Date.now() - startTime);
        return result;
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();

      const result: ToolPermissionResult = label === "allow"
        ? { behavior: "allow", decisionClassification: "user_temporary" }
        : deny("Bash", "Bash command is not read-only.");
      await this.logResponseAndDecision(roleName, command, response, result, Date.now() - startTime);
      return result;
    } catch (error) {
      const result = deny("Bash", "Bash permission classifier failed.");
      await this.rawLogger?.write({
        type: "llm.error",
        operation: "bash_permission",
        role: roleName,
        command,
        error: formatUnknownError(error),
        durationMs: Date.now() - startTime,
      });
      await this.logDecision(roleName, command, result, Date.now() - startTime);
      return result;
    }
  }

  private async logResponseAndDecision(
    roleName: string,
    command: string,
    response: Awaited<ReturnType<typeof complete>>,
    result: ToolPermissionResult,
    durationMs: number,
  ): Promise<void> {
    await this.rawLogger?.write({
      type: "llm.response",
      operation: "bash_permission",
      role: roleName,
      command,
      response: {
        stopReason: response.stopReason,
        content: response.content,
      },
      durationMs,
    });
    await this.logDecision(roleName, command, result, durationMs);
  }

  private async logDecision(
    roleName: string,
    command: string,
    result: ToolPermissionResult,
    durationMs: number,
  ): Promise<void> {
    await this.rawLogger?.write({
      type: "tool.permission_result",
      operation: "bash_permission",
      role: roleName,
      command,
      behavior: result.behavior,
      message: "message" in result ? result.message : undefined,
      durationMs,
    });
  }
}

function deny(toolName: string, message: string): ToolPermissionResult {
  return {
    behavior: "deny",
    message: `Tool ${toolName} denied: ${message}`,
    decisionClassification: "user_reject",
  };
}

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

