import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AgentConversationMessage, AgentRequest, InboundAttachment } from "../index.js";

export const DEFAULT_WORKSPACE_MAX_CHARS = 8_000;
export const DEFAULT_HISTORY_MAX_MESSAGES = 20;
export const DEFAULT_ATTACHMENT_MAX_CHARS = 12_000;

export async function buildWorkspacePrompt(
  request: AgentRequest,
  maxChars = DEFAULT_WORKSPACE_MAX_CHARS,
  historyMaxMessages = DEFAULT_HISTORY_MAX_MESSAGES,
  attachmentMaxChars = DEFAULT_ATTACHMENT_MAX_CHARS
): Promise<string> {
  const workspaceContext = await readWorkspaceContext(request.workspacePath, maxChars);
  const attachmentContext = await buildAttachmentContext(request.attachments, attachmentMaxChars);
  const historyContext = buildHistoryContext(request.history, historyMaxMessages);
  const chatContext = buildChatContext(request);

  if (
    workspaceContext === undefined &&
    attachmentContext === undefined &&
    historyContext === undefined &&
    chatContext === undefined
  ) {
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

  if (attachmentContext !== undefined) {
    parts.push(attachmentContext, "");
  }

  if (historyContext !== undefined) {
    parts.push(historyContext, "");
  }

  if (chatContext !== undefined) {
    parts.push(chatContext, "");
  }

  parts.push("User request:", request.text);
  return parts.join("\n");
}

export async function buildAttachmentContext(
  attachments: readonly InboundAttachment[] | undefined,
  maxChars = DEFAULT_ATTACHMENT_MAX_CHARS
): Promise<string | undefined> {
  if (attachments === undefined || attachments.length === 0) return undefined;

  let remainingChars = maxChars;
  const parts: string[] = ["Uploaded document context:"];
  for (const attachment of attachments) {
    if (remainingChars <= 0) break;

    const normalized = await readAttachmentText(attachment);
    const text =
      normalized.length <= remainingChars
        ? normalized
        : `${truncateToBudget(normalized, remainingChars)}\n[Document truncated for context budget.]`;
    remainingChars -= text.length;

    parts.push(
      "",
      `Document: ${attachment.name}`,
      `Media type: ${attachment.mimeType}`,
      `Size: ${attachment.sizeBytes} bytes`,
      "Content:",
      text
    );
  }

  parts.push(
    "",
    "Use the uploaded document context above to answer this user request when it is relevant."
  );
  return parts.join("\n");
}

async function readAttachmentText(attachment: InboundAttachment): Promise<string> {
  try {
    const content = await readFile(attachment.storagePath, "utf8");
    const normalized = content.trim();
    return normalized.length > 0 ? normalized : "[Uploaded document is empty.]";
  } catch {
    return "[Uploaded document could not be read.]";
  }
}

export function buildHistoryContext(
  history: readonly AgentConversationMessage[] | undefined,
  maxMessages = DEFAULT_HISTORY_MAX_MESSAGES
): string | undefined {
  if (history === undefined || history.length === 0) return undefined;

  const messages = history.slice(-maxMessages);
  const lines = messages.map((message) => {
    const role = message.role === "user" ? "User" : "Assistant";
    return `${role}: ${message.content}`;
  });

  return [
    "Previous conversation:",
    ...lines,
    "",
    "Continue the conversation below. The user may refer to earlier messages above."
  ].join("\n");
}

/**
 * Build chat context instructions for the agent prompt.
 *
 * In group chats, tells the AI the sender's userid for conversational context.
 * The WeChat smart bot channel renders mention markup as literal text, so the
 * prompt explicitly forbids @mentions. Returns undefined when chatType is not
 * set (non-chat channels).
 */
export function buildChatContext(request: AgentRequest): string | undefined {
  if (request.chatType === undefined) return undefined;

  if (request.chatType === "group") {
    return [
      "Chat context: group chat.",
      `The sender's userid is "${request.user.id}".`,
      "Do not use @mentions or angle-bracket userid mention markup; this channel renders them as plain text.",
      "Address people in natural language instead."
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
