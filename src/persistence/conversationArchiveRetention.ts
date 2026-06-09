import type Database from "better-sqlite3";
import type { ConversationLogger } from "../core/index.js";
import { toLocalIsoString } from "../logging/jsonlLogger.js";

export type ConversationArchiveRetentionOptions = {
  readonly db: Database.Database;
  readonly retentionDays: number;
  readonly cleanupIntervalMs: number;
  readonly logger?: ConversationLogger;
  now?(): Date;
};

export type ConversationArchiveRetentionHandle = {
  stop(): void;
};

const MS_PER_DAY = 86_400_000;

export function cleanupExpiredConversationArchives(options: {
  readonly db: Database.Database;
  readonly retentionDays: number;
  now?(): Date;
}): number {
  const now = options.now?.() ?? new Date();
  const cutoff = toLocalIsoString(new Date(now.getTime() - options.retentionDays * MS_PER_DAY));
  const result = options.db
    .prepare("DELETE FROM conversation_history_archives WHERE archived_at < ?")
    .run(cutoff);
  return result.changes;
}

export function startConversationArchiveRetention(
  options: ConversationArchiveRetentionOptions
): ConversationArchiveRetentionHandle {
  void runCleanup(options);
  const handle = setInterval(() => {
    void runCleanup(options);
  }, options.cleanupIntervalMs);

  return {
    stop() {
      clearInterval(handle);
    }
  };
}

async function runCleanup(options: ConversationArchiveRetentionOptions): Promise<void> {
  try {
    const deletedCount = cleanupExpiredConversationArchives(options);
    await options.logger?.write({
      type: "conversation.archive_retention.cleaned",
      retentionDays: options.retentionDays,
      deletedCount
    });
  } catch (error) {
    await options.logger?.write({
      type: "conversation.archive_retention.error",
      retentionDays: options.retentionDays,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}
