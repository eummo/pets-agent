import type { Api, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import { withRetry } from "../config/retry.js";
import type {
  StoredRoleConfig,
  ToolPermissionDecider,
  ToolPermissionResult
} from "../auth/index.js";
import type { AgentConversationMessage, UserRole } from "../core/index.js";
import type { UserIntent } from "./index.js";
import { isValidIntentType } from "./index.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";
import { isRecord, stringField, formatUnknownError } from "../core/unknownRecord.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";

const INTENT_SYSTEM_PROMPT = `You are an intent classifier for a knowledge-base assistant.
Given a user message, conversation history (if any), and their current role, classify the intent into exactly one of:

- "query": The user is asking a question, searching for information, or requesting an explanation.
- "mutate": The user wants to modify, create, delete, or update code/files in the workspace.
- "update_kb": The user wants to update, add, or modify knowledge-base content (not code).

Classification rules:
- Use the conversation history to resolve ambiguous short messages.
  If the assistant just suggested adding/updating content and the user says something like
  "补充一下", "好的", "go ahead", "please do", classify as the same intent as the suggestion.
- When the user asks about something that contains modification-related words in its name
  (e.g., "更新日志", "修改记录", "changelog", "update log"), but is clearly asking
  for information, classify as "query".
- When the user asks how or why something is created, updated, changed, or implemented,
  classify as "query" unless they ask you to perform the change.
- Do not grant or deny permission. Only classify the user's intent.
- If in doubt, classify as "query".

Examples (without history):
- "What is the current architecture?" -> query
- "更新日志是什么" -> query
- "修改记录怎么查看" -> query
- "客户订单是怎么创建的" -> query
- "我想修改订单系统" -> mutate
- "更新知识库里的订单流程" -> update_kb

Examples (with history):
- [assistant: "需要补充参数文档"] "补充一下" -> update_kb
- [assistant: "是否需要修改代码？"] "好的" -> mutate
- [assistant: "订单流程分为三步"] "补充一下" -> update_kb
- [assistant: "这是修改记录的内容"] "修改记录是什么" -> query

Respond with ONLY the intent label, nothing else.`;

const INTENT_TIMEOUT_MS = 5000;
const INTENT_MAX_RETRIES = 2;
const BASH_PERMISSION_SYSTEM_PROMPT = `You are a Bash command permission classifier.
Decide whether a Bash command is read-only inspection.

Return exactly one label:
- "allow": the command only reads/inspects local state and should not modify files, processes, network state, package state, git state, permissions, or databases.
- "deny": the command may write, delete, move, install, start services, stop processes, change git state, change permissions, make network requests with side effects, or is ambiguous.

Do not explain. Return only "allow" or "deny".`;
const BASH_PERMISSION_TIMEOUT_MS = 5000;

export class LlmIntentDetectionService {
  public constructor(
    private readonly model: Model<Api>,
    private readonly apiKey: string,
    private readonly rawLogger?: JsonlLogger
  ) {}

  public async detectIntent(
    userMessage: string,
    role: UserRole,
    history?: readonly AgentConversationMessage[]
  ): Promise<UserIntent> {
    const startTime = Date.now();
    const requestContent = buildIntentUserContent(userMessage, role, history);
    await this.rawLogger?.write({
      type: "llm.request",
      operation: "intent_detection",
      role,
      userMessage,
      systemPrompt: INTENT_SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: requestContent
        }
      ]
    });

    try {
      const response = await withRetry(
        async () => {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), INTENT_TIMEOUT_MS);

          return complete(
            this.model,
            {
              systemPrompt: INTENT_SYSTEM_PROMPT,
              messages: [
                {
                  role: "user",
                  content: requestContent,
                  timestamp: Date.now()
                }
              ]
            },
            {
              apiKey: this.apiKey,
              signal: controller.signal
            }
          )
            .then((response) => {
              if (response.stopReason === "error" && isRetryableProviderResponse(response)) {
                throw new Error(errorMessageForResponse(response));
              }
              return response;
            })
            .finally(() => clearTimeout(timeout));
        },
        {
          retries: INTENT_MAX_RETRIES,
          shouldRetry: (error) => isAbortError(error) || isRetryableError(error),
          onRetry: ({ attempt, delayMs, error }) => {
            void this.rawLogger?.write({
              type: "intent.retry",
              role,
              userMessage,
              attempt,
              delayMs,
              error: formatUnknownError(error)
            });
          }
        }
      );

      if (response.stopReason === "error") {
        return await this.logFallbackResult(
          userMessage,
          role,
          startTime,
          "provider_error",
          serializePiResponse(response)
        );
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();

      if (isValidIntentType(label)) {
        const intent = { type: label };
        await this.logResponseAndResult({
          role,
          userMessage,
          response: serializePiResponse(response),
          intent,
          source: "model",
          durationMs: Date.now() - startTime
        });
        return intent;
      }

      return await this.logFallbackResult(
        userMessage,
        role,
        startTime,
        "invalid_label",
        serializePiResponse(response)
      );
    } catch (error) {
      await this.rawLogger?.write({
        type: "llm.error",
        operation: "intent_detection",
        role,
        userMessage,
        error: formatUnknownError(error),
        durationMs: Date.now() - startTime
      });
      return await this.logFallbackResult(userMessage, role, startTime, "exception");
    }
  }

  public readonly decideToolPermission: ToolPermissionDecider = async (
    roleConfig: StoredRoleConfig,
    toolName: string,
    input: Record<string, unknown>
  ) => {
    if (toolName !== "Bash") {
      return denyToolPermission(
        roleConfig.name,
        toolName,
        "Only Bash commands are classified by this decider."
      );
    }

    const command = input["command"];
    if (typeof command !== "string" || command.trim().length === 0) {
      return denyToolPermission(roleConfig.name, toolName, "Bash command is missing.");
    }

    return await this.classifyBashPermission(roleConfig.name, command);
  };

  private async logFallbackResult(
    userMessage: string,
    role: UserRole,
    startTime: number,
    reason: string,
    response?: Record<string, unknown>
  ): Promise<UserIntent> {
    const intent = fallbackIntentFor(userMessage);
    await this.logResponseAndResult({
      role,
      userMessage,
      intent,
      source: "fallback",
      reason,
      durationMs: Date.now() - startTime,
      ...(response !== undefined ? { response } : {})
    });
    return intent;
  }

  private async logResponseAndResult(event: {
    readonly role: UserRole;
    readonly userMessage: string;
    readonly response?: Record<string, unknown>;
    readonly intent: UserIntent;
    readonly source: "model" | "fallback";
    readonly reason?: string;
    readonly durationMs: number;
  }): Promise<void> {
    if (event.response !== undefined) {
      await this.rawLogger?.write({
        type: "llm.response",
        operation: "intent_detection",
        role: event.role,
        userMessage: event.userMessage,
        response: event.response,
        durationMs: event.durationMs
      });
    }

    await this.rawLogger?.write({
      type: "intent.result",
      role: event.role,
      userMessage: event.userMessage,
      intentType: event.intent.type,
      source: event.source,
      ...(event.reason !== undefined ? { reason: event.reason } : {}),
      durationMs: event.durationMs
    });
  }

  private async classifyBashPermission(
    roleName: string,
    command: string
  ): Promise<ToolPermissionResult> {
    const startTime = Date.now();
    const userContent = `Role: ${roleName}\nCommand: ${command}`;
    await this.rawLogger?.write({
      type: "llm.request",
      operation: "bash_permission",
      role: roleName,
      command,
      systemPrompt: BASH_PERMISSION_SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: userContent
        }
      ]
    });

    try {
      const response = await withRetry(async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), BASH_PERMISSION_TIMEOUT_MS);

        return complete(
          this.model,
          {
            systemPrompt: BASH_PERMISSION_SYSTEM_PROMPT,
            messages: [
              {
                role: "user",
                content: userContent,
                timestamp: Date.now()
              }
            ]
          },
          {
            apiKey: this.apiKey,
            signal: controller.signal
          }
        ).finally(() => clearTimeout(timeout));
      });

      if (response.stopReason === "error") {
        const result = denyToolPermission(roleName, "Bash", "Bash permission classifier failed.");
        await this.logBashPermissionResponseAndDecision(
          roleName,
          command,
          response,
          result,
          Date.now() - startTime
        );
        return result;
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();
      const result: ToolPermissionResult =
        label === "allow"
          ? { behavior: "allow", decisionClassification: "user_temporary" }
          : denyToolPermission(roleName, "Bash", "Bash command is not read-only.");
      await this.logBashPermissionResponseAndDecision(
        roleName,
        command,
        response,
        result,
        Date.now() - startTime
      );
      return result;
    } catch (error) {
      const result = denyToolPermission(roleName, "Bash", "Bash permission classifier failed.");
      await this.rawLogger?.write({
        type: "llm.error",
        operation: "bash_permission",
        role: roleName,
        command,
        error: formatUnknownError(error),
        durationMs: Date.now() - startTime
      });
      await this.logBashPermissionDecision(roleName, command, result, Date.now() - startTime);
      return result;
    }
  }

  private async logBashPermissionResponseAndDecision(
    roleName: string,
    command: string,
    response: Awaited<ReturnType<typeof complete>>,
    result: ToolPermissionResult,
    durationMs: number
  ): Promise<void> {
    await this.rawLogger?.write({
      type: "llm.response",
      operation: "bash_permission",
      role: roleName,
      command,
      response: serializePiResponse(response),
      durationMs
    });
    await this.logBashPermissionDecision(roleName, command, result, durationMs);
  }

  private async logBashPermissionDecision(
    roleName: string,
    command: string,
    result: ToolPermissionResult,
    durationMs: number
  ): Promise<void> {
    await this.rawLogger?.write({
      type: "tool.permission_result",
      operation: "bash_permission",
      role: roleName,
      command,
      behavior: result.behavior,
      message: result.behavior === "deny" ? result.message : undefined,
      durationMs
    });
  }
}

function denyToolPermission(
  roleName: string,
  toolName: string,
  message?: string
): ToolPermissionResult {
  return {
    behavior: "deny",
    message: message ?? `Tool ${toolName} is not permitted for role ${roleName}.`,
    decisionClassification: "user_reject"
  };
}

function buildIntentUserContent(
  userMessage: string,
  role: UserRole,
  history?: readonly AgentConversationMessage[]
): string {
  if (history === undefined || history.length === 0) {
    return `User role: ${role}\nUser message: ${userMessage}`;
  }

  return [
    `User role: ${role}`,
    "",
    "Conversation history:",
    ...history.map((m) => `${m.role}: ${m.content}`),
    "",
    "Current user message:",
    userMessage
  ].join("\n");
}

function serializePiResponse(
  response: Awaited<ReturnType<typeof complete>>
): Record<string, unknown> {
  return {
    stopReason: response.stopReason,
    content: response.content
  };
}
function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException) return error.name === "AbortError";
  if (error instanceof Error) return error.name === "AbortError";
  return false;
}

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = error.message.toLowerCase();
  return (
    message.includes("abort") ||
    message.includes("rate") ||
    message.includes("overload") ||
    message.includes("429") ||
    message.includes("503")
  );
}

function isRetryableProviderResponse(response: Awaited<ReturnType<typeof complete>>): boolean {
  return isRetryableError(new Error(errorMessageForResponse(response)));
}

function errorMessageForResponse(response: Awaited<ReturnType<typeof complete>>): string {
  const errorMessage = isRecord(response) ? stringField(response, "errorMessage") : undefined;
  if (errorMessage !== undefined) {
    return errorMessage;
  }
  return response.stopReason;
}
