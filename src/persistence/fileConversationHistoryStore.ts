import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import type { AgentConversationMessage, ConversationHistoryStore, ConversationSessionKey } from "../core/contracts.js";
import { FileMutex, isFileNotFound, serializeSessionKey } from "./fileStoreUtils.js";

type StoredHistory = {
  readonly messages: readonly AgentConversationMessage[];
  readonly createdAt: string;
  readonly updatedAt: string;
};

type ArchivedHistory = StoredHistory & {
  readonly archivedAt: string;
};

type HistoryStoreFile = {
  readonly histories?: Record<string, StoredHistory>;
  readonly archives?: Record<string, readonly ArchivedHistory[]>;
};

export type FileConversationHistoryStoreOptions = {
  readonly maxMessages?: number;
};

export class FileConversationHistoryStore implements ConversationHistoryStore {
  private readonly filePath: string;
  private readonly maxMessages: number;
  private readonly mutex = new FileMutex();

  public constructor(filePathInput: string, options: FileConversationHistoryStoreOptions = {}) {
    this.filePath = path.resolve(filePathInput);
    this.maxMessages = options.maxMessages ?? 20;
  }

  public async get(key: ConversationSessionKey): Promise<readonly AgentConversationMessage[]> {
    const file = await this.readStore();
    return file.histories?.[serializeSessionKey(key)]?.messages ?? [];
  }

  public async append(
    key: ConversationSessionKey,
    messages: readonly AgentConversationMessage[]
  ): Promise<void> {
    if (messages.length === 0) {
      return;
    }

    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const histories = { ...(file.histories ?? {}) };
      const keyText = serializeSessionKey(key);
      const now = new Date().toISOString();
      const previous = histories[keyText];
      histories[keyText] = {
        messages: [...(previous?.messages ?? []), ...messages].slice(-this.maxMessages),
        createdAt: previous?.createdAt ?? now,
        updatedAt: now
      };
      await this.writeStore(withExistingArchives({ histories }, file));
    } finally {
      release();
    }
  }

  public async compact(key: ConversationSessionKey, summary: string): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const keyText = serializeSessionKey(key);
      const existing = file.histories?.[keyText];

      if (existing === undefined || existing.messages.length === 0) {
        return;
      }

      const compactSummary: AgentConversationMessage = {
        role: "assistant",
        content: `[Previous conversation summary]\n${summary}`,
      };
      const recentMessages = existing.messages.slice(-2);
      const messages = [compactSummary, ...recentMessages].slice(-this.maxMessages);

      const histories = { ...(file.histories ?? {}) };
      histories[keyText] = {
        messages,
        createdAt: existing.createdAt,
        updatedAt: new Date().toISOString(),
      };
      await this.writeStore(withExistingArchives({ histories }, file));
    } finally {
      release();
    }
  }

  public async delete(key: ConversationSessionKey): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const keyText = serializeSessionKey(key);
      const histories = Object.fromEntries(
        Object.entries(file.histories ?? {}).filter(([storedKey]) => storedKey !== keyText)
      );
      await this.writeStore(withExistingArchives({ histories }, file));
    } finally {
      release();
    }
  }

  public async archive(key: ConversationSessionKey): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const file = await this.readStore();
      const keyText = serializeSessionKey(key);
      const current = file.histories?.[keyText];

      if (current === undefined || current.messages.length === 0) {
        return;
      }

      const histories = Object.fromEntries(
        Object.entries(file.histories ?? {}).filter(([storedKey]) => storedKey !== keyText)
      );
      const archives = { ...(file.archives ?? {}) };
      archives[keyText] = [
        ...(archives[keyText] ?? []),
        {
          ...current,
          archivedAt: new Date().toISOString()
        }
      ];
      await this.writeStore({ histories, archives });
    } finally {
      release();
    }
  }

  private async readStore(): Promise<HistoryStoreFile> {
    try {
      const raw = await readFile(this.filePath, "utf8");
      return JSON.parse(raw) as HistoryStoreFile;
    } catch (error) {
      if (isFileNotFound(error)) {
        return {};
      }
      throw error;
    }
  }

  private async writeStore(file: HistoryStoreFile): Promise<void> {
    await mkdir(path.dirname(this.filePath), { recursive: true });
    const tempPath = `${this.filePath}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(tempPath, `${JSON.stringify(file, null, 2)}\n`, "utf8");
    await rename(tempPath, this.filePath);
  }
}

function withExistingArchives(file: HistoryStoreFile, existingFile: HistoryStoreFile): HistoryStoreFile {
  if (existingFile.archives === undefined) {
    return file;
  }

  return {
    ...file,
    archives: existingFile.archives
  };
}

