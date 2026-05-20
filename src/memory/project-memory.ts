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
  /** workdir → { stack: string[], mtime: number } */
  private techStackCache = new Map<string, { stack: string[]; mtime: number }>();

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
   * Detect tech stack from project files, with in-memory cache keyed on workdir.
   * Cache is invalidated if any indicator file has a newer mtime than the cached mtime.
   */
  detectTechStack(workdir: string): string[] {
    // Indicator files we track for cache invalidation
    const indicatorFiles = [
      "package.json", "Cargo.toml", "go.mod", "requirements.txt", "pyproject.toml",
      "Pipfile", "tsconfig.json", "vite.config.ts", "vite.config.js", "next.config.js",
      "next.config.ts", "astro.config.mjs", "bun.lockb", "pnpm-lock.yaml", "yarn.lock",
      "package-lock.json", "android/app/build.gradle", "ios/Podfile", "pubspec.yaml",
      "pubspec.lock", "CMakeLists.txt", "Makefile", "setup.py", "manage.py",
      "Gemfile", "Rakefile", "composer.json", "Stack.yaml", "go.sum", ".nvmrc",
      ".python-version", "Dockerfile", "docker-compose.yml", "kubernetes.yaml",
      "kustomization.yaml", "terraform.tf", ".tfvars", "renovate.json",
      "biome.json", "eslint.config.js", ".eslintrc.js", "prettier.config.js",
      ".prettierrc", "stylelint.config.js", "vitest.config.ts", "vitest.config.js",
      "jest.config.js", "playwright.config.ts", "webpack.config.js",
      "rollup.config.js", "turbo.json", "nx.json", ".github/workflows",
    ];

    // Get max mtime of all indicator files
    let maxMtime = 0;
    for (const file of indicatorFiles) {
      try {
        const fp = path.join(workdir, file);
        const stat = fs.statSync(fp);
        if (stat.mtimeMs > maxMtime) maxMtime = stat.mtimeMs;
      } catch { /* ignore */ }
    }

    const cached = this.techStackCache.get(workdir);
    if (cached && cached.mtime >= maxMtime) {
      return cached.stack;
    }

    // Perform full detection
    const indicators: string[] = [];
    const files: Record<string, string[]> = {
      "package.json": ["node", "npm"],
      "Cargo.toml": ["rust", "cargo"],
      "go.mod": ["go"],
      "requirements.txt": ["python", "pip"],
      "pyproject.toml": ["python", "pdm", "poetry"],
      "Pipfile": ["python", "pipenv"],
      "tsconfig.json": ["typescript", "tsc"],
      "vite.config.ts": ["vite", "frontend"],
      "vite.config.js": ["vite", "frontend"],
      "next.config.js": ["nextjs", "react"],
      "next.config.ts": ["nextjs", "react"],
      "astro.config.mjs": ["astro"],
      "bun.lockb": ["bun"],
      "pnpm-lock.yaml": ["pnpm"],
      "yarn.lock": ["yarn", "node"],
      "package-lock.json": ["npm", "node"],
      "android/app/build.gradle": ["android", "java", "gradle"],
      "ios/Podfile": ["ios", "cocoapods", "swift"],
      "pubspec.yaml": ["flutter", "dart"],
      "pubspec.lock": ["flutter", "dart"],
      "CMakeLists.txt": ["cmake", "c++"],
      "Makefile": ["make", "c"],
      "setup.py": ["python", "setuptools"],
      "manage.py": ["django", "python"],
      "Gemfile": ["ruby", "rails"],
      "Rakefile": ["ruby"],
      "composer.json": ["php", "composer"],
      "Stack.yaml": ["haskell", "stack"],
      "go.sum": ["go"],
      ".nvmrc": ["node", "nvm"],
      ".python-version": ["python"],
      "Dockerfile": ["docker"],
      "docker-compose.yml": ["docker", "docker-compose"],
      "kubernetes.yaml": ["kubernetes", "k8s"],
      "kustomization.yaml": ["kubernetes", "kustomize"],
      "terraform.tf": ["terraform"],
      ".tfvars": ["terraform"],
      "renovate.json": ["renovate"],
      "biome.json": ["biome", "linter"],
      "eslint.config.js": ["eslint"],
      ".eslintrc.js": ["eslint"],
      "prettier.config.js": ["prettier"],
      ".prettierrc": ["prettier"],
      "stylelint.config.js": ["stylelint"],
      "vitest.config.ts": ["vitest"],
      "vitest.config.js": ["vitest"],
      "jest.config.js": ["jest"],
      "playwright.config.ts": ["playwright"],
      "webpack.config.js": ["webpack"],
      "rollup.config.js": ["rollup"],
      "turbo.json": ["turborepo"],
      "nx.json": ["nx"],
      ".github/workflows": ["github-actions", "ci"],
    };

    for (const [file, tags] of Object.entries(files)) {
      if (fs.existsSync(path.join(workdir, file))) {
        indicators.push(...tags);
      }
    }

    const stack = Array.from(new Set<string>(indicators));
    this.techStackCache.set(workdir, { stack, mtime: maxMtime });
    return stack;
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
