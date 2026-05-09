/**
 * ProjectMemory — per-project context that auto-learns from task history.
 *
 * Stores in: ~/.pets-agent/memory/projects/<project-hash>.md
 *
 * Auto-captures:
 * - Tech stack (package.json, tsconfig.json, etc.)
 * - Build/test/dev commands
 * - Key file purposes
 * - Agent outcomes per project
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { createHash } from "crypto";
import { MemoryStore, type MemoryEntry } from "./store.js";

export interface ProjectEntry extends MemoryEntry {
  tags: string[];
  projectKey: string;
}

export class ProjectMemory {
  private stores = new Map<string, ProjectMemoryStore>();
  private cacheDir: string;

  constructor() {
    this.cacheDir = path.join(os.homedir(), ".pets-agent", "memory", "projects");
  }

  private projectKey(workdir: string): string {
    return createHash("sha256").update(workdir).digest("hex").slice(0, 12);
  }

  store(workdir: string): ProjectMemoryStore {
    const key = this.projectKey(workdir);
    if (!this.stores.has(key)) {
      this.stores.set(key, new ProjectMemoryStore(key, path.join(this.cacheDir, `${key}.md`)));
    }
    return this.stores.get(key)!;
  }

  /**
   * Try to detect tech stack from project files.
   */
  detectTechStack(workdir: string): string[] {
    const indicators: string[] = [];
    const files: Record<string, string[]> = {
      "package.json": ["node", "npm"],
      "Cargo.toml": ["rust", "cargo"],
      "go.mod": ["go"],
      "requirements.txt": ["python", "pip"],
      "pyproject.toml": ["python", "pdm", "poetry"],
      "tsconfig.json": ["typescript", "tsc"],
      "vite.config.ts": ["vite", "frontend"],
      "next.config.js": ["nextjs", "react"],
      "astro.config.mjs": ["astro"],
      "bun.lockb": ["bun"],
      "pnpm-lock.yaml": ["pnpm"],
    };

    for (const [file, tags] of Object.entries(files)) {
      if (fs.existsSync(path.join(workdir, file))) {
        indicators.push(...tags);
      }
    }

    return Array.from(new Set<string>(indicators));
  }

  /**
   * List all known project keys.
   */
  listProjects(): string[] {
    return Array.from(this.stores.keys());
  }
}

export class ProjectMemoryStore extends MemoryStore {
  private projectKey: string;

  constructor(projectKey: string, filepath: string) {
    super({ filename: filepath, charLimit: 3000 });
    this.projectKey = projectKey;
  }

  protected renderSnapshot(): string {
    if (this.entries.length === 0) return "";
    const usage = this.usage();
    const lines = [
      "═══════════════════════════════════════════════",
      `PROJECT MEMORY [${usage.pct}% — ${usage.current}/${usage.limit} chars]`,
      "═══════════════════════════════════════════════",
      ...this.entries.map((e) => {
        const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
        return `${tags}\n${e.content}`;
      }),
    ];
    return lines.join("\n");
  }

  protected serializeEntry(e: MemoryEntry): string {
    const pe = e as ProjectEntry;
    const header = `ID:${pe.id}|TAGS:${pe.tags.join(",")}|DATE:${pe.createdAt}|SOURCE:${pe.source ?? "auto"}|PKEY:${pe.projectKey}`;
    return `${header}\n${pe.content}`;
  }

  protected parseEntry(raw: string): Partial<ProjectEntry> | null {
    const [header, ...body] = raw.split("\n");
    if (!header || body.length === 0) return null;
    const content = body.join("\n");

    const idMatch = header.match(/ID:([^|]+)/);
    const tagsMatch = header.match(/TAGS:([^|]+)/);
    const dateMatch = header.match(/DATE:([^|]+)/);
    const sourceMatch = header.match(/SOURCE:([^|]+)/);
    const pkeyMatch = header.match(/PKEY:([^|]+)/);

    return {
      id: idMatch?.[1] ?? "",
      tags: tagsMatch?.[1] ? tagsMatch[1].split(",").filter(Boolean) : [],
      createdAt: dateMatch?.[1] ?? new Date().toISOString(),
      source: (sourceMatch?.[1] as MemoryEntry["source"]) ?? "auto",
      projectKey: pkeyMatch?.[1] ?? "",
      content,
    };
  }

  addToProject(content: string, opts?: { tags?: string[]; source?: MemoryEntry["source"] }): { success: boolean; error?: string } {
    const pe: { tags?: string[]; source?: MemoryEntry["source"]; projectKey?: string } = opts ?? {};
    pe.projectKey = this.projectKey;
    return this.add(content, pe);
  }
}

export const projectMemory = new ProjectMemory();
