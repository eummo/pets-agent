import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import type { ConversationSessionKey, ConversationSessionStore } from "./ports.js";

type StoredSession = {
  readonly sessionId: string;
  readonly createdAt: string;
  readonly updatedAt: string;
};

type SessionStoreFile = {
  readonly sessions?: Record<string, StoredSession>;
};

export class FileConversationSessionStore implements ConversationSessionStore {
  private readonly filePath: string;

  public constructor(filePathInput: string) {
    this.filePath = path.resolve(filePathInput);
  }

  public async get(key: ConversationSessionKey): Promise<string | undefined> {
    const file = await this.readStore();
    return file.sessions?.[serializeKey(key)]?.sessionId;
  }

  public async set(key: ConversationSessionKey, sessionId: string): Promise<void> {
    const file = await this.readStore();
    const sessions = { ...(file.sessions ?? {}) };
    const keyText = serializeKey(key);
    const now = new Date().toISOString();
    sessions[keyText] = {
      sessionId,
      createdAt: sessions[keyText]?.createdAt ?? now,
      updatedAt: now
    };
    await this.writeStore({ sessions });
  }

  public async delete(key: ConversationSessionKey): Promise<void> {
    const file = await this.readStore();
    const keyText = serializeKey(key);
    const sessions = Object.fromEntries(
      Object.entries(file.sessions ?? {}).filter(([storedKey]) => storedKey !== keyText)
    );
    await this.writeStore({ sessions });
  }

  private async readStore(): Promise<SessionStoreFile> {
    try {
      const raw = await readFile(this.filePath, "utf8");
      return JSON.parse(raw) as SessionStoreFile;
    } catch (error) {
      if (isFileNotFound(error)) {
        return {};
      }
      throw error;
    }
  }

  private async writeStore(file: SessionStoreFile): Promise<void> {
    await mkdir(path.dirname(this.filePath), { recursive: true });
    const tempPath = `${this.filePath}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(tempPath, `${JSON.stringify(file, null, 2)}\n`, "utf8");
    await rename(tempPath, this.filePath);
  }
}

function serializeKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath)]);
}

function isFileNotFound(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
