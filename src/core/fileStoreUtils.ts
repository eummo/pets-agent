import path from "node:path";
import type { ConversationSessionKey } from "./ports.js";

export function serializeSessionKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath)]);
}

export function isFileNotFound(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
