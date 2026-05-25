import type Database from "better-sqlite3";
import type { FeedbackEntry, FeedbackStore, FeedbackStatus, UserIntent } from "../core/ports.js";

type FeedbackRow = {
  readonly id: number;
  readonly user_id: string;
  readonly channel: string | null;
  readonly message_id: string | null;
  readonly workspace_path: string | null;
  readonly intent_type: UserIntent["type"] | null;
  readonly role_name: string | null;
  readonly user_message: string;
  readonly conversation_context: string;
  readonly status: FeedbackStatus;
  readonly created_at: string;
  readonly updated_at: string;
};

function rowToEntry(row: FeedbackRow): FeedbackEntry {
  return {
    id: row.id,
    userId: row.user_id,
    ...(row.channel !== null && { channel: row.channel }),
    ...(row.message_id !== null && { messageId: row.message_id }),
    ...(row.workspace_path !== null && { workspacePath: row.workspace_path }),
    ...(row.intent_type !== null && { intentType: row.intent_type }),
    ...(row.role_name !== null && { roleName: row.role_name }),
    userMessage: row.user_message,
    conversationContext: row.conversation_context,
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export class SqliteFeedbackStore implements FeedbackStore {
  public constructor(private readonly db: Database.Database) {}

  public save(entry: FeedbackEntry): Promise<number> {
    const result = this.db.prepare(`
      INSERT INTO feedback (
        user_id,
        channel,
        message_id,
        workspace_path,
        intent_type,
        role_name,
        user_message,
        conversation_context,
        status
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      entry.userId,
      entry.channel ?? null,
      entry.messageId ?? null,
      entry.workspacePath ?? null,
      entry.intentType ?? null,
      entry.roleName ?? null,
      entry.userMessage,
      entry.conversationContext,
      entry.status,
    );
    return Promise.resolve(Number(result.lastInsertRowid));
  }

  public updateStatus(id: number, status: FeedbackStatus): Promise<boolean> {
    const result = this.db.prepare(`
      UPDATE feedback SET status = ?, updated_at = datetime('now', 'localtime') WHERE id = ?
    `).run(status, id);
    return Promise.resolve(result.changes > 0);
  }

  public getAll(): Promise<readonly FeedbackEntry[]> {
    const rows = this.db.prepare(`
      SELECT
        id,
        user_id,
        channel,
        message_id,
        workspace_path,
        intent_type,
        role_name,
        user_message,
        conversation_context,
        status,
        created_at,
        updated_at
      FROM feedback
      ORDER BY id DESC
    `).all() as FeedbackRow[];
    return Promise.resolve(rows.map(rowToEntry));
  }
}
