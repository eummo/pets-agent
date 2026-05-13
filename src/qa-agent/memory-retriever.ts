/**
 * MemoryRetriever — queries all memory stores and formats results
 * as structured context for the QA Agent.
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { PatternMemory, type PatternEntry } from "../memory/pattern-memory.js";
import { PreferenceMemory, type PreferenceEntry } from "../memory/preference-memory.js";
import { ProjectMemory, ProjectMemoryStore, type ProjectEntry } from "../memory/project-memory.js";
import { MEMORY_DIR, ENTRY_DELIMITER, type MemoryEntry } from "../memory/store.js";

const PROJECTS_DIR = path.join(MEMORY_DIR, "projects");

export class MemoryRetriever {
  private patterns: PatternMemory;
  private preferences: PreferenceMemory;
  private projectEntries: { entry: MemoryEntry; projectKey: string }[] = [];

  constructor() {
    this.patterns = new PatternMemory();
    this.preferences = new PreferenceMemory();
  }

  async init(): Promise<void> {
    await this.patterns.loadAsync();
    await this.preferences.loadAsync();

    // Load project entries by reading and parsing files directly
    // (ProjectMemoryStore constructor has a path bug — it prepends MEMORY_DIR to an already-absolute path)
    if (fs.existsSync(PROJECTS_DIR)) {
      const files = fs.readdirSync(PROJECTS_DIR).filter((f) => f.endsWith(".md"));
      for (const file of files) {
        const projectKey = file.replace(".md", "");
        const fp = path.join(PROJECTS_DIR, file);
        try {
          const raw = fs.readFileSync(fp, "utf-8");
          if (!raw.trim()) continue;
          const items = raw.split(ENTRY_DELIMITER).filter(Boolean);
          for (const item of items) {
            const [header, ...body] = item.split("\n");
            if (!header || body.length === 0) continue;
            const content = body.join("\n");
            const tagsMatch = header.match(/TAGS:([^|]+)/);
            const idMatch = header.match(/ID:([^|]+)/);
            const dateMatch = header.match(/DATE:([^|]+)/);
            this.projectEntries.push({
              projectKey,
              entry: {
                id: idMatch?.[1] ?? "",
                content,
                tags: tagsMatch?.[1] ? tagsMatch[1].split(",").filter(Boolean) : [],
                createdAt: dateMatch?.[1] ?? new Date().toISOString(),
                source: "auto",
              },
            });
          }
        } catch { /* ignore unreadable files */ }
      }
    }
  }

  /**
   * Retrieve relevant memory entries for a query and format as context.
   */
  retrieve(query: string): string {
    const sections: string[] = [];

    // 1. Pattern search (has relevance scoring)
    const patternResults = this.patterns.search(query).slice(0, 10);
    if (patternResults.length > 0) {
      const lines = patternResults.map((e: PatternEntry) => {
        const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
        const date = new Date(e.createdAt).toLocaleDateString("zh-CN");
        return `- (${date})${tags} ${e.content}`;
      });
      sections.push("### 命令/模式知识\n" + lines.join("\n"));
    }

    // 2. Preference query
    const prefResults = this.preferences.query(query).slice(0, 10);
    if (prefResults.length > 0) {
      const lines = prefResults.map((e) => {
        const pe = e as PreferenceEntry;
        const tags = pe.tags.length > 0 ? ` [${pe.tags.join(", ")}]` : "";
        return `- ${tags} ${pe.content}`;
      });
      sections.push("### Agent 偏好知识\n" + lines.join("\n"));
    }

    // 3. Project memory query
    const projectResults: string[] = [];
    const q = query.toLowerCase();
    for (const { entry } of this.projectEntries) {
      if (
        entry.content.toLowerCase().includes(q) ||
        entry.tags.some((t) => t.toLowerCase().includes(q))
      ) {
        const tags = entry.tags.length > 0 ? ` [${entry.tags.join(", ")}]` : "";
        projectResults.push(`- ${tags} ${entry.content}`);
        if (projectResults.length >= 5) break;
      }
    }
    if (projectResults.length > 0) {
      sections.push("### 项目知识\n" + projectResults.join("\n"));
    }

    if (sections.length === 0) {
      return "";
    }

    return "以下是从知识库中检索到的相关信息：\n\n" + sections.join("\n\n");
  }

  /**
   * List all memory as a full overview (for open-ended questions).
   */
  listAll(): string {
    const sections: string[] = [];

    // Patterns overview
    const allPatterns = this.patterns.all();
    if (allPatterns.length > 0) {
      const lines = allPatterns.map((e) => {
        const pe = e as PatternEntry;
        const tags = pe.tags.length > 0 ? ` [${pe.tags.join(", ")}]` : "";
        return `- ${tags} ${e.content}`;
      });
      sections.push(`### 命令/模式知识 (${allPatterns.length} 条)\n` + lines.join("\n"));
    }

    // Preferences overview
    const allPrefs = this.preferences.all();
    if (allPrefs.length > 0) {
      const lines = allPrefs.map((e) => {
        const pe = e as PreferenceEntry;
        const tags = pe.tags.length > 0 ? ` [${pe.tags.join(", ")}]` : "";
        return `- ${tags} ${e.content}`;
      });
      sections.push(`### Agent 偏好知识 (${allPrefs.length} 条)\n` + lines.join("\n"));
    }

    // Projects overview
    if (this.projectEntries.length > 0) {
      const lines = this.projectEntries.map(({ entry }) => {
        const tags = entry.tags.length > 0 ? ` [${entry.tags.join(", ")}]` : "";
        return `- ${tags} ${entry.content}`;
      });
      sections.push(`### 项目知识 (${this.projectEntries.length} 条)\n` + lines.join("\n"));
    }

    if (sections.length === 0) {
      return "知识库当前为空，暂无任何记录。";
    }

    return "以下是知识库的完整概览：\n\n" + sections.join("\n\n");
  }
}
