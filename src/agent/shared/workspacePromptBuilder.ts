import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AgentRequest } from "../index.js";

export const DEFAULT_WORKSPACE_MAX_CHARS = 8_000;

export async function buildWorkspacePrompt(
  request: AgentRequest,
  maxChars = DEFAULT_WORKSPACE_MAX_CHARS
): Promise<string> {
  const workspaceContext = await readWorkspaceContext(request.workspacePath, maxChars);
  const chatContext = buildChatContext(request);

  if (workspaceContext === undefined && chatContext === undefined) {
    return request.text;
  }

  const parts: string[] = [];

  if (workspaceContext !== undefined) {
    parts.push(
      "Selected workspace context:",
      workspaceContext,
      "",
      "Use the selected workspace context above as the primary source of truth.",
      "If the user asks about the current project, architecture, or system, answer about this selected workspace.",
      "Do not answer from the host agent implementation unless the user explicitly asks how this assistant is built.",
      "Do not infer the business domain from names such as Pets Agent unless the workspace context explicitly defines that domain.",
      ""
    );
  }

  if (chatContext !== undefined) {
    parts.push(chatContext, "");
  }

  parts.push("User request:", request.text);
  return parts.join("\n");
}

/**
 * Build chat context instructions for the agent prompt.
 *
 * In group chats, tells the AI the sender's userid and the `<@userid>` syntax
 * for @mentioning group members. In single chats, instructs the AI not to use
 * @mentions. Returns undefined when chatType is not set (non-chat channels).
 */
export function buildChatContext(request: AgentRequest): string | undefined {
  if (request.chatType === undefined) return undefined;

  if (request.chatType === "group") {
    return [
      "Chat context: group chat.",
      `The sender's userid is "${request.user.id}".`,
      "You may @mention the sender or other group members in your reply using the syntax <@userid>.",
      `For example, to mention the sender, include <@${request.user.id}> in your Markdown response.`,
      "Use @mentions sparingly -- only when directly addressing or drawing attention to a specific person.",
      "Do not use @mentions in single chats."
    ].join("\n");
  }

  // Single chat -- no mention capability
  return "Chat context: single chat. Do not use @mentions.";
}

async function readWorkspaceContext(
  workspacePath: string,
  maxChars: number
): Promise<string | undefined> {
  try {
    const content = await readFile(path.join(workspacePath, "CLAUDE.md"), "utf8");
    const normalized = content.trim();
    if (normalized.length === 0) return undefined;
    return normalized.length <= maxChars ? normalized : truncateToBudget(normalized, maxChars);
  } catch {
    return undefined;
  }
}

export function truncateToBudget(content: string, maxChars: number): string {
  const sections = splitAtHeadings(content);
  let result = "";
  for (const section of sections) {
    if (result.length + section.length > maxChars) break;
    result += section;
  }
  return result.length > 0 ? result : content.slice(0, maxChars);
}

export function splitAtHeadings(content: string): string[] {
  const lines = content.split("\n");
  const sections: string[] = [];
  let current = "";
  for (const line of lines) {
    if (/^#{1,3}\s/.test(line) && current.length > 0) {
      sections.push(current);
      current = line + "\n";
    } else {
      current += line + "\n";
    }
  }
  if (current.length > 0) sections.push(current);
  return sections;
}
