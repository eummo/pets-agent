/**
 * PreferenceMemory — learns which agent types / approaches are preferred per task type.
 *
 * Tracks:
 * - Which agent type succeeds most for given task patterns
 * - User feedback on task outcomes
 * - Output format preferences
 */

import { MemoryStore, type MemoryEntry } from "./store.js";

export interface PreferenceEntry extends MemoryEntry {
  tags: string[]; // e.g. ["agent:claude-code", "task:coding", "success"]
  quality?: number; // 1-5 rating if provided
}

export class PreferenceMemory extends MemoryStore {
  constructor() {
    super({ filename: "preferences.md", charLimit: 2000 });
  }

  protected renderSnapshot(): string {
    if (this.entries.length === 0) return "";
    const usage = this.usage();
    const lines = [
      "═══════════════════════════════════════════════",
      `PREFERENCES (learned agent preferences) [${usage.pct}% — ${usage.current}/${usage.limit} chars]`,
      "═══════════════════════════════════════════════",
      ...this.entries.map((e) => {
        const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
        return `${tags}\n${e.content}`;
      }),
    ];
    return lines.join("\n");
  }

  protected serializeEntry(e: MemoryEntry): string {
    const pe = e as PreferenceEntry;
    const header = `ID:${pe.id}|TAGS:${pe.tags.join(",")}|DATE:${pe.createdAt}|SOURCE:${pe.source ?? "auto"}|Q:${pe.quality ?? ""}`;
    return `${header}\n${pe.content}`;
  }

  protected parseEntry(raw: string): Partial<PreferenceEntry> | null {
    const [header, ...body] = raw.split("\n");
    if (!header || body.length === 0) return null;
    const content = body.join("\n");

    const idMatch = header.match(/ID:([^|]+)/);
    const tagsMatch = header.match(/TAGS:([^|]+)/);
    const dateMatch = header.match(/DATE:([^|]+)/);
    const sourceMatch = header.match(/SOURCE:([^|]+)/);
    const qMatch = header.match(/Q:([^|]+)/);

    return {
      id: idMatch?.[1] ?? "",
      tags: tagsMatch?.[1] ? tagsMatch[1].split(",").filter(Boolean) : [],
      createdAt: dateMatch?.[1] ?? new Date().toISOString(),
      source: (sourceMatch?.[1] as MemoryEntry["source"]) ?? "auto",
      quality: qMatch?.[1] ? parseInt(qMatch[1]) : undefined,
      content,
    };
  }

  /**
   * Record outcome of a task. Used to build success statistics.
   */
  recordOutcome(params: {
    agentType: string;
    taskPrompt: string;
    success: boolean;
    exitCode?: number;
    durationSec?: number;
    fileCount?: number;
  }): void {
    const tags = [
      `agent:${params.agentType}`,
      params.success ? "outcome:success" : "outcome:failure",
    ];

    const lines = [
      `${params.success ? "✓" : "✗"} ${params.agentType} for: ${params.taskPrompt.slice(0, 80)}`,
      params.durationSec !== undefined ? `  duration: ${params.durationSec}s` : "",
      params.fileCount !== undefined ? `  files: ${params.fileCount}` : "",
      params.exitCode !== undefined ? `  exit: ${params.exitCode}` : "",
    ].filter(Boolean);

    this.add(lines.join("\n"), { tags, source: "task" });
  }

  /**
   * Get best agent type for a task prompt (simple keyword matching).
   */
  suggestAgentType(taskPrompt: string): string | null {
    const entries = this.query(taskPrompt);
    const successful = entries.filter(
      (e) => e.tags.includes("outcome:success")
    );

    if (successful.length === 0) return null;

    // Count wins per agent type
    const wins: Record<string, number> = {};
    for (const e of successful) {
      const agentTag = e.tags.find((t) => t.startsWith("agent:"));
      if (agentTag) {
        wins[agentTag.replace("agent:", "")] = (wins[agentTag.replace("agent:", "")] ?? 0) + 1;
      }
    }

    const best = Object.entries(wins).sort((a, b) => b[1] - a[1]);
    return best.length > 0 ? best[0][0] : null;
  }
}

export const preferenceMemory = new PreferenceMemory();
