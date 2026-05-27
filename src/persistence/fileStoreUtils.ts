import path from "node:path";
import type { ConversationSessionKey } from "./index.js";

export function serializeSessionKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath), key.chatId ?? ""]);
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
    const queued = previous ? previous.then(() => next) : next;
    this.locks.set(key, queued);
    if (previous !== undefined) {
      await previous;
    }
    let released = false;
    return () => {
      if (released) {
        return;
      }
      released = true;
      resolve();
      void queued.finally(() => {
        if (this.locks.get(key) === queued) {
          this.locks.delete(key);
        }
      });
    };
  }

  public activeLockCount(): number {
    return this.locks.size;
  }
}

