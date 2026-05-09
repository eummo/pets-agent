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

export abstract class MemoryStore {
  protected entries: MemoryEntry[] = [];
  protected snapshot: string = "";
  protected charLimit: number;
  protected filename: string;
  protected dirty: boolean = false;

  constructor(opts: MemoryStoreOptions) {
    this.filename = opts.filename;
    this.charLimit = opts.charLimit;
    this.load();
  }

  // -- Disk I/O ------------------------------------------------------------

  protected filepath(): string {
    return path.join(MEMORY_DIR, this.filename);
  }

  load(): void {
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
  }

  protected persist(): void {
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
    } catch (err) {
      console.error("[MemoryStore] persist failed:", err);
    }

    this.dirty = false;
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

  add(content: string, opts?: { tags?: string[]; source?: MemoryEntry["source"] }): { success: boolean; error?: string } {
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
        error: `Memory at ${total}/${this.charLimit} chars. Would exceed limit by ${(total + content.length) - this.charLimit} chars.`,
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
    this.persist();
    this.captureSnapshot();

    return { success: true };
  }

  remove(idOrSubstring: string): { success: boolean; error?: string } {
    const idx = this.entries.findIndex(
      (e) => e.id === idOrSubstring || e.content.includes(idOrSubstring)
    );

    if (idx === -1) {
      return { success: false, error: `No entry matched '${idOrSubstring}'.` };
    }

    this.entries.splice(idx, 1);
    this.persist();
    this.captureSnapshot();

    return { success: true };
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
