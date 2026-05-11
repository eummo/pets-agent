/**
 * Base MemoryStore — file-backed persistent memory with frozen snapshot.
 *
 * Inspired by Hermes MemoryStore design:
 * - load_from_disk() captures a frozen snapshot for session use
 * - Mutations persist to disk immediately but do NOT update the snapshot
 * - Snapshot refreshes on next load (new session / explicit reload)
 *
 * Entry delimiter: § (section sign). Entries are multiline-capable.
 * Character limits enforced per store type.
 */

import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import * as os from "os";
import { randomBytes } from "crypto";

export const MEMORY_DIR = path.join(os.homedir(), ".pets-agent", "memory");
export const ENTRY_DELIMITER = "\n§\n";

export interface MemoryEntry {
  id: string;
  content: string;
  tags: string[];
  createdAt: string;
  source?: "task" | "user" | "auto";
}

export interface MemoryStoreOptions {
  filename: string;
  charLimit: number;
}

/**
 * Unified error result type — replaces console.error + silent swallow patterns.
 * All public-facing store methods return this instead of throwing.
 */
export interface StoreResult<T = void> {
  ok: true;
  value: T;
  /** True if the operation touched disk (useful for callers batching writes) */
  persisted?: boolean;
}
export interface StoreError {
  ok: false;
  error: string;
  /**
   * Severity: 'recoverable' = operation failed but store is still usable
   *           'fatal'       = store data may be corrupted, reload advised
   */
  severity: "recoverable" | "fatal";
}

export type PersistResult = StoreResult<void> | StoreError;

export abstract class MemoryStore {
  protected entries: MemoryEntry[] = [];
  protected snapshot: string = "";
  protected charLimit: number;
  protected filename: string;
  protected dirty: boolean = false;
  private _loaded: boolean = false;

  constructor(opts: MemoryStoreOptions) {
    this.filename = opts.filename;
    const envLimit = parseInt(process.env.MEMORY_CHAR_LIMIT ?? "", 10);
    this.charLimit = !isNaN(envLimit) && envLimit > 0 ? envLimit : opts.charLimit;
  }

  // -- Disk I/O ------------------------------------------------------------

  protected filepath(): string {
    return path.join(MEMORY_DIR, this.filename);
  }

  async loadAsync(): Promise<void> {
    if (this._loaded) return;
    try {
      if (!fs.existsSync(MEMORY_DIR)) {
        fs.mkdirSync(MEMORY_DIR, { recursive: true });
        this.entries = [];
      }

      const file = this.filepath();
      if (!fs.existsSync(file)) {
        this.entries = [];
      } else {
        const raw = await fsp.readFile(file, "utf-8");
        if (raw.trim()) {
          const items = raw.split(ENTRY_DELIMITER).filter(Boolean);
          this.entries = items.map((item) => this.parseEntry(item)).filter(Boolean) as MemoryEntry[];
        } else {
          this.entries = [];
        }
      }
    } catch {
      this.entries = [];
    }

    this.captureSnapshot();
    this._loaded = true;
  }

  load(): void {
    // Synchronous fallback for contexts that must block
    if (this._loaded) return;
    try {
      if (!fs.existsSync(MEMORY_DIR)) {
        fs.mkdirSync(MEMORY_DIR, { recursive: true });
        this.entries = [];
      }

      const file = this.filepath();
      if (!fs.existsSync(file)) {
        this.entries = [];
      } else {
        const raw = fs.readFileSync(file, "utf-8");
        if (raw.trim()) {
          const items = raw.split(ENTRY_DELIMITER).filter(Boolean);
          this.entries = items.map((item) => this.parseEntry(item)).filter(Boolean) as MemoryEntry[];
        } else {
          this.entries = [];
        }
      }
    } catch {
      this.entries = [];
    }

    this.captureSnapshot();
    this._loaded = true;
  }

  /**
   * Persist entries to disk (async). Returns a typed result instead of console.error.
   *
   * RECOVERABLE errors (disk full, permission): log is written, caller can retry.
   * FATAL errors (write error that may corrupt file): returned as fatal so caller
   * can trigger a reload of the store.
   */
  protected persistAsync(): Promise<PersistResult> {
    try {
      if (!fs.existsSync(MEMORY_DIR)) {
        fs.mkdirSync(MEMORY_DIR, { recursive: true });
      }

      const content = this.entries
        .map((e) => this.serializeEntry(e))
        .join(ENTRY_DELIMITER);

      const tmp = this.filepath() + `.tmp.${randomBytes(4).toString("hex")}`;
      return fsp.writeFile(tmp, content, "utf-8")
        .then(() => fsp.rename(tmp, this.filepath()))
        .then(() => {
          this.dirty = false;
          return { ok: true as const, value: undefined as void, persisted: true };
        })
        .catch((err: NodeJS.ErrnoException) => {
          const recoverable =
            err.code === "EACCES" || err.code === "ENOSPC" || err.code === "EROFS";
          const result: StoreError = {
            ok: false,
            error:
              err.code === "ENOENT"
                ? `Directory not found and could not be created: ${MEMORY_DIR}`
                : `Write failed: ${err.message}`,
            severity: recoverable ? "recoverable" : "fatal",
          };
          process.stderr.write(
            `[MemoryStore] persistAsync failed (${result.severity}): ${result.error}\n`
          );
          this.dirty = false;
          return result;
        });
    } catch (err) {
      const error = err instanceof Error ? err : String(err);
      const result: StoreError = { ok: false, error: `Unexpected: ${error}`, severity: "fatal" };
      process.stderr.write(`[MemoryStore] persistAsync unexpected error: ${error}\n`);
      this.dirty = false;
      return Promise.resolve(result);
    }
  }

  /**
   * Persist entries to disk (sync). Returns a typed result instead of console.error.
   */
  protected persist(): PersistResult {
    try {
      if (!fs.existsSync(MEMORY_DIR)) {
        fs.mkdirSync(MEMORY_DIR, { recursive: true });
      }

      const content = this.entries
        .map((e) => this.serializeEntry(e))
        .join(ENTRY_DELIMITER);

      const tmp = this.filepath() + `.tmp.${randomBytes(4).toString("hex")}`;
      fs.writeFileSync(tmp, content, "utf-8");
      fs.renameSync(tmp, this.filepath());
      this.dirty = false;
      return { ok: true, value: undefined as void, persisted: true };
    } catch (err) {
      const error = err instanceof Error ? err : String(err);
      const result: StoreError = { ok: false, error: `Write failed: ${error}`, severity: "fatal" };
      process.stderr.write(`[MemoryStore] persist failed: ${result.error}\n`);
      this.dirty = false;
      return result;
    }
  }

  // -- Snapshot (frozen at load time, used for injection) -----------------

  captureSnapshot(): void {
    this.snapshot = this.renderSnapshot();
  }

  getSnapshot(): string {
    return this.snapshot;
  }

  protected abstract renderSnapshot(): string;

  protected abstract serializeEntry(e: MemoryEntry): string;
  protected abstract parseEntry(raw: string): Partial<MemoryEntry> | null;

  // -- CRUD ----------------------------------------------------------------

  /**
   * Add a new memory entry. Persists synchronously; callers can check the
   * `persisted` flag on the return value to know whether the write succeeded.
   *
   * Returns `{ success: true }` on OK.
   * Returns `{ success: false, error: string }` on validation/dedup/limit errors.
   * Persist I/O errors are logged to stderr but do not cause false returns —
   * the entry is still held in memory for the session.
   */
  add(
    content: string,
    opts?: { tags?: string[]; source?: MemoryEntry["source"] }
  ): { success: true; persisted: boolean } | { success: false; error: string } {
    content = content.trim();
    if (!content) return { success: false, error: "Content cannot be empty." };

    // Dedup
    if (this.entries.some((e) => e.content === content)) {
      return { success: false, error: "Entry already exists." };
    }

    const total = this.totalChars();
    if (total + content.length > this.charLimit) {
      return {
        success: false,
        error: `Memory at ${total}/${this.charLimit} chars. Would exceed limit by ${
          total + content.length - this.charLimit
        } chars.`,
      };
    }

    const entry: MemoryEntry = {
      id: randomBytes(8).toString("hex"),
      content,
      tags: opts?.tags ?? [],
      createdAt: new Date().toISOString(),
      source: opts?.source ?? "auto",
    };

    this.entries.unshift(entry);
    const persisted = this.persist();
    this.captureSnapshot();

    return { success: true, persisted: persisted.ok };
  }

  /**
   * Remove an entry by id or content substring.
   * Returns `{ success: true }` on OK.
   * Returns `{ success: false, error: string }` if no match found.
   */
  remove(idOrSubstring: string): { success: true; persisted: boolean } | { success: false; error: string } {
    const idx = this.entries.findIndex(
      (e) => e.id === idOrSubstring || e.content.includes(idOrSubstring)
    );

    if (idx === -1) {
      return { success: false, error: `No entry matched '${idOrSubstring}'.` };
    }

    this.entries.splice(idx, 1);
    const persisted = this.persist();
    this.captureSnapshot();

    return { success: true, persisted: persisted.ok };
  }

  query(searchText: string): MemoryEntry[] {
    if (!searchText.trim()) return this.entries.slice(0, 50);
    const q = searchText.toLowerCase();
    return this.entries.filter(
      (e) =>
        e.content.toLowerCase().includes(q) ||
        e.tags.some((t) => t.toLowerCase().includes(q))
    );
  }

  all(): MemoryEntry[] {
    return this.entries;
  }

  protected totalChars(): number {
    return this.entries.map((e) => e.content).join(ENTRY_DELIMITER).length;
  }

  usage(): { current: number; limit: number; pct: number } {
    const current = this.totalChars();
    return { current, limit: this.charLimit, pct: Math.min(100, Math.round((current / this.charLimit) * 100)) };
  }
}
