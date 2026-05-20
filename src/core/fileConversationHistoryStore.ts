import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import type { AgentConversationMessage, ConversationHistoryStore, ConversationSessionKey } from "./ports.js";

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

  public constructor(filePathInput: string, options: FileConversationHistoryStoreOptions = {}) {
    this.filePath = path.resolve(filePathInput);
    this.maxMessages = options.maxMessages ?? 20;
  }

  public async get(key: ConversationSessionKey): Promise<readonly AgentConversationMessage[]> {
    const file = await this.readStore();
    return file.histories?.[serializeKey(key)]?.messages ?? [];
  }

  public async append(
    key: ConversationSessionKey,
    messages: readonly AgentConversationMessage[]
  ): Promise<void> {
    if (messages.length === 0) {
      return;
    }

    const file = await this.readStore();
    const histories = { ...(file.histories ?? {}) };
    const keyText = serializeKey(key);
    const now = new Date().toISOString();
    const previous = histories[keyText];
    histories[keyText] = {
      messages: [...(previous?.messages ?? []), ...messages].slice(-this.maxMessages),
      createdAt: previous?.createdAt ?? now,
      updatedAt: now
    };
    await this.writeStore(withExistingArchives({ histories }, file));
  }

  public async delete(key: ConversationSessionKey): Promise<void> {
    const file = await this.readStore();
    const keyText = serializeKey(key);
    const histories = Object.fromEntries(
      Object.entries(file.histories ?? {}).filter(([storedKey]) => storedKey !== keyText)
    );
    await this.writeStore(withExistingArchives({ histories }, file));
  }

  public async archive(key: ConversationSessionKey): Promise<void> {
    const file = await this.readStore();
    const keyText = serializeKey(key);
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

function serializeKey(key: ConversationSessionKey): string {
  return JSON.stringify([key.channel, key.userId, path.resolve(key.workspacePath)]);
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

function isFileNotFound(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
