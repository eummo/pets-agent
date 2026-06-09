import type Database from "better-sqlite3";
import { z } from "zod";
import type { AgentConversationMessage } from "../core/index.js";
import { toLocalIsoString } from "../logging/jsonlLogger.js";
import type {
  ConversationHistoryStore,
  ConversationSessionKey,
  ConversationSessionStore
} from "./index.js";
import { serializeSessionKey } from "./fileStoreUtils.js";

const conversationMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string()
});

type SessionRow = {
  readonly session_id: string;
};

type HistoryRow = {
  readonly messages_json: string;
  readonly created_at: string;
  readonly updated_at: string;
};

type KeyColumns = {
  readonly sessionKey: string;
  readonly channel: string;
  readonly userId: string;
  readonly workspacePath: string;
  readonly chatId: string | null;
};

export class SqliteConversationSessionStore implements ConversationSessionStore {
  public constructor(private readonly db: Database.Database) {}

  public get(key: ConversationSessionKey): Promise<string | undefined> {
    const row = this.db
      .prepare("SELECT session_id FROM conversation_sessions WHERE session_key = ?")
      .get(serializeSessionKey(key)) as SessionRow | undefined;
    return Promise.resolve(row?.session_id);
  }

  public set(key: ConversationSessionKey, sessionId: string): Promise<void> {
    const keyColumns = toKeyColumns(key);
    this.db
      .prepare(
        `
      INSERT INTO conversation_sessions (
        session_key, channel, user_id, workspace_path, chat_id, session_id, created_at, updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))
      ON CONFLICT(session_key) DO UPDATE SET
        session_id = excluded.session_id,
        channel = excluded.channel,
        user_id = excluded.user_id,
        workspace_path = excluded.workspace_path,
        chat_id = excluded.chat_id,
        updated_at = datetime('now', 'localtime')
    `
      )
      .run(
        keyColumns.sessionKey,
        keyColumns.channel,
        keyColumns.userId,
        keyColumns.workspacePath,
        keyColumns.chatId,
        sessionId
      );
    return Promise.resolve();
  }

  public delete(key: ConversationSessionKey): Promise<void> {
    this.db
      .prepare("DELETE FROM conversation_sessions WHERE session_key = ?")
      .run(serializeSessionKey(key));
    return Promise.resolve();
  }
}

export type SqliteConversationHistoryStoreOptions = {
  readonly maxMessages?: number;
};

export class SqliteConversationHistoryStore implements ConversationHistoryStore {
  private readonly maxMessages: number;

  public constructor(
    private readonly db: Database.Database,
    options: SqliteConversationHistoryStoreOptions = {}
  ) {
    this.maxMessages = options.maxMessages ?? 20;
  }

  public get(key: ConversationSessionKey): Promise<readonly AgentConversationMessage[]> {
    const row = this.db
      .prepare("SELECT messages_json FROM conversation_histories WHERE session_key = ?")
      .get(serializeSessionKey(key)) as Pick<HistoryRow, "messages_json"> | undefined;
    if (row === undefined) {
      return Promise.resolve([]);
    }
    return Promise.resolve(parseMessages(row.messages_json));
  }

  public append(
    key: ConversationSessionKey,
    messages: readonly AgentConversationMessage[]
  ): Promise<void> {
    if (messages.length === 0) {
      return Promise.resolve();
    }

    const keyColumns = toKeyColumns(key);
    const now = toLocalIsoString(new Date());
    const transaction = this.db.transaction(() => {
      const existing = this.getHistoryRow(keyColumns.sessionKey);
      const previousMessages = existing === undefined ? [] : parseMessages(existing.messages_json);
      const nextMessages = [...previousMessages, ...messages].slice(-this.maxMessages);
      this.upsertHistory({
        keyColumns,
        messages: nextMessages,
        createdAt: existing?.created_at ?? now
      });
    });
    transaction();
    return Promise.resolve();
  }

  public compact(key: ConversationSessionKey, summary: string): Promise<void> {
    const keyColumns = toKeyColumns(key);
    const existing = this.getHistoryRow(keyColumns.sessionKey);
    if (existing === undefined) {
      return Promise.resolve();
    }

    const existingMessages = parseMessages(existing.messages_json);
    if (existingMessages.length === 0) {
      return Promise.resolve();
    }

    const compactSummary: AgentConversationMessage = {
      role: "assistant",
      content: `[Previous conversation summary]\n${summary}`
    };
    const recentMessages = existingMessages.slice(-2);
    const messages = [compactSummary, ...recentMessages].slice(-this.maxMessages);
    this.upsertHistory({ keyColumns, messages, createdAt: existing.created_at });
    return Promise.resolve();
  }

  public delete(key: ConversationSessionKey): Promise<void> {
    this.db
      .prepare("DELETE FROM conversation_histories WHERE session_key = ?")
      .run(serializeSessionKey(key));
    return Promise.resolve();
  }

  public archive(key: ConversationSessionKey): Promise<void> {
    const keyColumns = toKeyColumns(key);
    const transaction = this.db.transaction(() => {
      const existing = this.getHistoryRow(keyColumns.sessionKey);
      if (existing === undefined) {
        return;
      }

      const messages = parseMessages(existing.messages_json);
      if (messages.length === 0) {
        return;
      }

      this.db
        .prepare(
          `
        INSERT INTO conversation_history_archives (
          session_key, channel, user_id, workspace_path, chat_id, messages_json, created_at, updated_at, archived_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
      `
        )
        .run(
          keyColumns.sessionKey,
          keyColumns.channel,
          keyColumns.userId,
          keyColumns.workspacePath,
          keyColumns.chatId,
          JSON.stringify(messages),
          existing.created_at,
          existing.updated_at
        );
      this.db
        .prepare("DELETE FROM conversation_histories WHERE session_key = ?")
        .run(keyColumns.sessionKey);
    });
    transaction();
    return Promise.resolve();
  }

  private getHistoryRow(sessionKey: string): HistoryRow | undefined {
    return this.db
      .prepare(
        "SELECT messages_json, created_at, updated_at FROM conversation_histories WHERE session_key = ?"
      )
      .get(sessionKey) as HistoryRow | undefined;
  }

  private upsertHistory(options: {
    readonly keyColumns: KeyColumns;
    readonly messages: readonly AgentConversationMessage[];
    readonly createdAt: string;
  }): void {
    this.db
      .prepare(
        `
      INSERT INTO conversation_histories (
        session_key, channel, user_id, workspace_path, chat_id, messages_json, created_at, updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
      ON CONFLICT(session_key) DO UPDATE SET
        channel = excluded.channel,
        user_id = excluded.user_id,
        workspace_path = excluded.workspace_path,
        chat_id = excluded.chat_id,
        messages_json = excluded.messages_json,
        updated_at = datetime('now', 'localtime')
    `
      )
      .run(
        options.keyColumns.sessionKey,
        options.keyColumns.channel,
        options.keyColumns.userId,
        options.keyColumns.workspacePath,
        options.keyColumns.chatId,
        JSON.stringify(options.messages),
        options.createdAt
      );
  }
}

function toKeyColumns(key: ConversationSessionKey): KeyColumns {
  return {
    sessionKey: serializeSessionKey(key),
    channel: key.channel,
    userId: key.userId,
    workspacePath: key.workspacePath,
    chatId: key.chatId ?? null
  };
}

function parseMessages(raw: string): readonly AgentConversationMessage[] {
  return z.array(conversationMessageSchema).parse(JSON.parse(raw));
}
