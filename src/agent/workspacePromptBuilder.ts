import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AgentRequest } from "../core/contracts.js";

export async function buildWorkspacePrompt(request: AgentRequest): Promise<string> {
  const workspaceContext = await readWorkspaceContext(request.workspacePath);
  if (workspaceContext === undefined) {
    return request.text;
  }

  return [
    "Selected workspace context:",
    workspaceContext,
    "",
    "Use the selected workspace context above as the primary source of truth.",
    "If the user asks about the current project, architecture, or system, answer about this selected workspace.",
    "Do not answer from the host agent implementation unless the user explicitly asks how this assistant is built.",
    "",
    "User request:",
    request.text,
  ].join("\n");
}

async function readWorkspaceContext(workspacePath: string): Promise<string | undefined> {
  try {
    const content = await readFile(path.join(workspacePath, "CLAUDE.md"), "utf8");
    const normalized = content.trim();
    return normalized.length > 0 ? normalized.slice(0, 4_000) : undefined;
  } catch {
    return undefined;
  }
}
