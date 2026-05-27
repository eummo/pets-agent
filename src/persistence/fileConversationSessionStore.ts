import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type { ConversationSessionKey, ConversationSessionStore } from "./index.js";
import { toLocalIsoString } from "../logging/jsonlLogger.js";
import { FileMutex, isFileNotFound, serializeSessionKey } from "./fileStoreUtils.js";

type StoredSession = {
  readonly sessionId: string;
  readonly createdAt: string;
  readonly updatedAt: string;
};

type SessionStoreFile = {
  readonly sessions?: Record<string, StoredSession>;
};

const storedSessionSchema = z.object({
  sessionId: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

const sessionStoreFileSchema = z.object({
  sessions: z.record(z.string(), storedSessionSchema).optional(),
});

export class FileConversationSessionStore implements ConversationSessionStore {
  private readonly filePath: string;
  private readonly mutex = new FileMutex();

  public constructor(filePathInput: string) {
    this.filePath = path.resolve(filePathInput);
  }

  public async get(key: ConversationSessionKey): Promise<string | undefined> {
    const file = await this.readStore();
    return file.sessions?.[serializeSessionKey(key)]?.sessionId;
  }

  public async set(key: ConversationSessionKey, sessionId: string): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const sessions = { ...(file.sessions ?? {}) };
      const keyText = serializeSessionKey(key);
      const now = toLocalIsoString(new Date());
      sessions[keyText] = {
        sessionId,
        createdAt: sessions[keyText]?.createdAt ?? now,
        updatedAt: now
      };
      await this.writeStore({ sessions });
    } finally {
      release();
    }
  }

  public async delete(key: ConversationSessionKey): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const keyText = serializeSessionKey(key);
      const sessions = Object.fromEntries(
        Object.entries(file.sessions ?? {}).filter(([storedKey]) => storedKey !== keyText)
      );
      await this.writeStore({ sessions });
    } finally {
      release();
    }
  }

  private async readStore(): Promise<SessionStoreFile> {
    try {
      const raw = await readFile(this.filePath, "utf8");
      const parsed = sessionStoreFileSchema.parse(JSON.parse(raw));
      return toSessionStoreFile(parsed);
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

function toSessionStoreFile(parsed: z.infer<typeof sessionStoreFileSchema>): SessionStoreFile {
  return {
    ...(parsed.sessions !== undefined ? { sessions: parsed.sessions } : {}),
  };
}
