/**
 * PatternMemory — stores successful command/code patterns discovered across tasks.
 *
 * Auto-learned from task output when a task succeeds.
 * Useful for: build commands, shell idioms, tool combos that worked before.
 */

import { MemoryStore, type MemoryEntry } from "./store.js";

export interface PatternEntry extends MemoryEntry {
  tags: string[]; // e.g. ["npm", "build", "vite", "error-fix"]
}

export class PatternMemory extends MemoryStore {
  constructor() {
    super({ filename: "patterns.md", charLimit: 3000 });
  }

  protected renderSnapshot(): string {
    if (this.entries.length === 0) return "";
    const usage = this.usage();
    const lines = [
      "═══════════════════════════════════════════════",
      `PATTERNS (successful commands/workflows) [${usage.pct}% — ${usage.current}/${usage.limit} chars]`,
      "═══════════════════════════════════════════════",
      ...this.entries.map((e) => {
        const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
        const date = new Date(e.createdAt).toLocaleDateString("zh-CN");
        return `[${date}]${tags}\n${e.content}`;
      }),
    ];
    return lines.join("\n");
  }

  protected serializeEntry(e: MemoryEntry): string {
    const pe = e as PatternEntry;
    const header = `ID:${pe.id}|TAGS:${pe.tags.join(",")}|DATE:${pe.createdAt}|SOURCE:${pe.source ?? "auto"}`;
    return `${header}\n${pe.content}`;
  }

  protected parseEntry(raw: string): Partial<PatternEntry> | null {
    const [header, ...body] = raw.split("\n");
    if (!header || body.length === 0) return null;
    const content = body.join("\n");

    const idMatch = header.match(/ID:([^|]+)/);
    const tagsMatch = header.match(/TAGS:([^|]+)/);
    const dateMatch = header.match(/DATE:([^|]+)/);
    const sourceMatch = header.match(/SOURCE:([^|]+)/);

    return {
      id: idMatch?.[1] ?? "",
      tags: tagsMatch?.[1] ? tagsMatch[1].split(",").filter(Boolean) : [],
      createdAt: dateMatch?.[1] ?? new Date().toISOString(),
      source: (sourceMatch?.[1] as MemoryEntry["source"]) ?? "auto",
      content,
    };
  }

  /**
   * Try to auto-learn patterns from task output.
   * Extracts: shell commands, file paths, error fixes.
   */
  learnFromOutput(outputLines: string[]): void {
    for (const line of outputLines) {
      // Shell command pattern (lines starting with $ or that look like commands)
      if (line.match(/^\s*\$\s+(.+)/)) {
        const cmd = line.replace(/^\s*\$\s+/, "").trim();
        if (cmd.length > 5 && cmd.length < 200) {
          this.add(cmd, { tags: ["command"], source: "task" });
        }
      }

      // Error fix patterns: "Error X → fixed with Y"
      const fixMatch = line.match(/(?:error|failed|exception)[:\s].*(?:fix|resolved|solved)[:\s]+(.+)/i);
      if (fixMatch) {
        this.add(fixMatch[1].trim(), { tags: ["error-fix"], source: "task" });
      }

      // Build/test success lines
      if (line.match(/^(✓|✔|done|success|passed|build successful)/i) && line.length > 3) {
        const prevIdx = outputLines.indexOf(line) - 1;
        if (prevIdx >= 0) {
          const prev = outputLines[prevIdx].replace(/^\s*\$\s+/, "").trim();
          if (prev && prev.length < 200) {
            this.add(prev, { tags: ["workflow"], source: "task" });
          }
        }
      }
    }
  }

  search(query: string): PatternEntry[] {
    return this.query(query) as PatternEntry[];
  }
}

export const patternMemory = new PatternMemory();
