import path from "node:path";
import type { ConversationSessionKey } from "./index.js";
import { AsyncMutex } from "../core/asyncMutex.js";

export function serializeSessionKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath), key.chatId ?? ""]);
}

export function isFileNotFound(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

export class FileMutex {
  private readonly mutex = new AsyncMutex();

  public async acquire(filePath: string): Promise<() => void> {
    return this.mutex.acquire(path.resolve(filePath));
  }

  public activeLockCount(): number {
    return this.mutex.activeLockCount();
  }
}

