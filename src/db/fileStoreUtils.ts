import path from "node:path";
import type { ConversationSessionKey } from "../core/ports.js";

export function serializeSessionKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath)]);
}

export function isFileNotFound(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

type Release = () => void;

export class FileMutex {
  private readonly locks = new Map<string, Promise<void>>();

  public async acquire(filePath: string): Promise<Release> {
    const key = path.resolve(filePath);
    const previous = this.locks.get(key);
    let resolve!: Release;
    const next = new Promise<void>((r) => { resolve = r; });
    this.locks.set(key, previous ? previous.then(() => next) : next);
    if (previous !== undefined) {
      await previous;
    }
    return resolve;
  }
}
