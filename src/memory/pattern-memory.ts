/**
 * PatternMemory — stores successful command/code patterns discovered across tasks.
 *
 * Auto-learned from task output when a task succeeds.
 * Useful for: build commands, shell idioms, tool combos that worked before.
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
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
   * Extracts: shell commands, file paths, error fixes, build success, watch patterns.
   *
   * Deduplication strategy (two-layer):
   * - Layer 1 — MemoryStore.add() dedup: same content can't be added twice in-session
   * - Layer 2 — learned_patterns.json: cross-session dedup for raw commands only
   *   (prevents re-learning commands across restarts; other pattern types always
   *    go through MemoryStore dedup so no cross-session dedup needed for them)
   *
   * Only raw shell commands ($ prefix) use Layer 2 dedup to avoid polluting
   * learned_patterns.json with error-fix/worklow content that may legitimately
   * overlap across sessions.
   */
  learnFromOutput(outputLines: string[]): void {
    const learnedFile = path.join(os.homedir(), ".pets-agent", "memory", "learned_patterns.json");
    const LEARNED_MAX = 2000; // cap entries to prevent unbounded growth

    interface LearnedEntry { cmd: string; learnedAt: number; }
    let learned: LearnedEntry[] = [];
    try {
      if (fs.existsSync(learnedFile)) {
        learned = JSON.parse(fs.readFileSync(learnedFile, "utf8")) as LearnedEntry[];
      }
    } catch { /* ignore */ }

    const seen = new Set<string>();

    for (let i = 0; i < outputLines.length; i++) {
      const line = outputLines[i];
      const trimmed = line.trim();

      // ─────────────────────────────────────────────────────────────
      // 1. Shell command pattern — $ prefix (classic)
      // ─────────────────────────────────────────────────────────────
      const dollarMatch = trimmed.match(/^\$\s+(.+)/);
      if (dollarMatch) {
        const cmd = dollarMatch[1].trim();
        if (cmd.length > 5 && cmd.length < 200 && !seen.has(cmd)) {
          seen.add(cmd);
          if (!learned.some((e) => e.cmd === cmd)) {
            // Evict oldest if at cap
            if (learned.length >= LEARNED_MAX) {
              learned.sort((a, b) => a.learnedAt - b.learnedAt);
              learned = learned.slice(1);
            }
            learned.push({ cmd, learnedAt: Date.now() });
            this.add(cmd, { tags: ["command"], source: "task" });
          }
        }
        continue;
      }

      // ─────────────────────────────────────────────────────────────
      // 2. CLI tool patterns — check against ALL patterns
      // ─────────────────────────────────────────────────────────────
      const cliCommandPatterns = [
        /^(npm|npx|yarn|pnpm|git|docker|docker-compose|kubectl|make|cargo|go|py|python|pip|pip3|rbenv|composer|apt|apt-get|yum|dnf|brew|choco)\s+/,
        /^(cd|ls|ll|la|mkdir|rm|rmrf|cp|mv|chmod|chown|cat|echo|export|source|alias|unalias|which|where|type|command)\s+/,
        /^(curl|wget|rsync|scp|sftp|ssh|ftp|telnet|netstat|ping|traceroute|dig|nslookup)\s+/,
        /^(vim|vi|nano|emacs|less|more|head|tail|grep|rg|fdfind|find|xargs|sort|uniq|wc|awk|sed|cut|tr|base64|openssl)\s+/,
        /^(tsc|ts-node|bundle|browserify|webpack|vite|esbuild|rollup|parcel|grunt|gulp)\s+/,
        /^(jest|vitest|mocha|jasmine|cypress|playwright|puppeteer|selenium)\s+/,
        /^(clang|gcc|g\+\+|swift|rustc|dart|flutter|java|javac|scala|groovy)\s+/,
        /^(terraform|terragrunt|pulumi|ansible|chef|puppet|vagrant|helm)\s+/,
        /^(node|ruby|perl|php|lua|haskell|elixir|erlang|bash|sh|zsh|fish)\s+/,
      ];
      const isCliCommand = cliCommandPatterns.some((p) => p.test(trimmed));
      if (isCliCommand) {
        const cmdLine = trimmed.split(/\s{2,}/)[0];
        if (cmdLine.length > 5 && cmdLine.length < 200 && !seen.has(cmdLine)) {
          const prevLine = i > 0 ? outputLines[i - 1].trim() : "";
          const isContinuation = /:\s*$|→\s*$/.test(prevLine);
          if (!isContinuation) {
            seen.add(cmdLine);
            if (!learned.some((e) => e.cmd === cmdLine)) {
              if (learned.length >= LEARNED_MAX) {
                learned.sort((a, b) => a.learnedAt - b.learnedAt);
                learned = learned.slice(1);
              }
              learned.push({ cmd: cmdLine, learnedAt: Date.now() });
              this.add(cmdLine, { tags: ["command", "claude-code"], source: "task" });
            }
          }
        }
        continue;
      }

      // ─────────────────────────────────────────────────────────────
      // 3. Error-fix pairs — "Error X → fixed with Y" / "failed...resolved..."
      // ─────────────────────────────────────────────────────────────
      const fixMatch = trimmed.match(/(?:error|failed|exception|panicked)[:\s].*(?:fix|resolved|solved|handled)[:\s]+(.+)/i);
      if (fixMatch) {
        const fix = fixMatch[1].trim();
        if (fix.length > 3 && fix.length < 200 && !seen.has(fix)) {
          seen.add(fix);
          this.add(fix, { tags: ["error-fix"], source: "task" });
        }
        continue;
      }

      // Error → solution inline pattern: "Error: X at Y → Z"
      const errorArrowMatch = trimmed.match(/(?:error|failed|exception)[:\s].*?→\s*(.+)/i);
      if (errorArrowMatch) {
        const fix = errorArrowMatch[1].trim();
        if (fix.length > 3 && fix.length < 200 && !seen.has(fix)) {
          seen.add(fix);
          this.add(fix, { tags: ["error-fix"], source: "task" });
        }
        continue;
      }

      // ─────────────────────────────────────────────────────────────
      // 4. Build success patterns
      // ─────────────────────────────────────────────────────────────
      const buildSuccessPatterns = [
        /^(✓|✔|✅|done|success|passed|build successful|compiled successfully|build complete|build succeeded)/i,
        /^(✓|✔|✅)\s*\S+\s+in\s+\d+ms/i,
        /^(✓|✔|✅)\s+\S+\s+(\S+\s+)?done/i,
        /^(DONE|ALL TESTS PASSED|BUILD SUCCESSFUL|COMPILED SUCCESSFULLY)/i,
        /^(✓|✔|✅)\s*\d+\s+(tests?|files?|modules?)\s+(passed|built|compiled)/i,
      ];
      const isBuildSuccess = buildSuccessPatterns.some((p) => trimmed.match(p));
      if (isBuildSuccess && trimmed.length > 3) {
        if (i > 0) {
          const prev = outputLines[i - 1].replace(/^\s*\$\s+/, "").trim();
          if (prev && prev.length < 200 && !seen.has(prev)) {
            seen.add(prev);
            this.add(prev, { tags: ["workflow", "build-success"], source: "task" });
          }
        }
        continue;
      }

      // "Build at X" or "built in Y ms" patterns
      const builtInMatch = trimmed.match(/built\s+in\s+(\d+ms|s\d+(\.\d+)?s)/i);
      if (builtInMatch) {
        if (i > 0) {
          const prev = outputLines[i - 1].replace(/^\s*\$\s+/, "").trim();
          if (prev && prev.length < 200 && !seen.has(prev)) {
            seen.add(prev);
            this.add(prev, { tags: ["workflow", "build-success"], source: "task" });
          }
        }
        continue;
      }

      // ─────────────────────────────────────────────────────────────
      // 5. Watch / serve patterns
      // ─────────────────────────────────────────────────────────────
      const watchPatterns = [
        /^(npm|npx|yarn|pnpm)\s+(run\s+)?(dev|serve|watch|start|live|hot)\b/i,
        /^(vite|webpack|rollup|parcel|esbuild)\s+(--\S+\s+)*(dev|serve|watch|build)/i,
        /^(nodemon|node-dev|ts-node-dev)\s+/i,
        /^(watchexec|watchman|fswatch)\s+/i,
        /^(python|mphp)\s+.*\s+-m\s+(http\.server|flask|django)/i,
        /^(live-server|serve|http-server|mini-http)\s+/i,
        /^(concurrently|npm-run-all)\s+/i,
        /listening\s+on\s+(https?:\/\/)?[\w\-\.:]+\//i,
        /server\s+running\s+at\s+(https?:\/\/)?[\w\-\.:]+\//i,
        /ready\s+in\s+\d+ms/i,
        /(Local|Network):\s+(https?:\/\/)?[\w\-\.:]+\//i,
        /^\s*( dév| dev | watch | serve )\s*:?\s*$/i,
      ];
      const watchMatch = watchPatterns.some((p) => trimmed.match(p));
      if (watchMatch) {
        if (trimmed.length < 200 && !seen.has(trimmed)) {
          seen.add(trimmed);
          this.add(trimmed, { tags: ["watch", "serve"], source: "task" });
        }
        continue;
      }

      // ─────────────────────────────────────────────────────────────
      // 6. File:line and file:line:col references (e.g., src/index.ts:12:5)
      // ─────────────────────────────────────────────────────────────
      const fileLinePatterns = [
        /^(\.\.?\/)[\w\-\/.\\]+\.(ts|tsx|js|jsx|mjs|cjs|cts|mts|py|rb|go|rs|java|cpp|c|h|hpp|swift|kt|scala|php|rb|sh|bash|zsh|yaml|yml|json|toml|xml|html|css|scss|sass|less|md|sql):(\d+)(:\d+)?/,
        /^(\.\.?\/)[\w\-\/.\\]+\.(ts|tsx|js|jsx|mjs|cjs|cts|mts):(\d+):(\d+)/,
        /^(\.\.?\/)[\w\-\/.\\]+\.(ts|tsx|js|jsx|mjs|cjs|cts|mts):(\d+):(\d+):/,
      ];
      const fileLineMatch = trimmed.match(fileLinePatterns[0]) || trimmed.match(fileLinePatterns[1]) || trimmed.match(fileLinePatterns[2]);
      if (fileLineMatch) {
        const ref = fileLineMatch[0];
        if (ref.length < 300 && !seen.has(ref)) {
          seen.add(ref);
          this.add(ref, { tags: ["file-reference"], source: "task" });
        }
        continue;
      }

      // Linter/formatter shorthand: "file.ts:10:5" (no leading path, just filename)
      const shortFileLineMatch = trimmed.match(/^[\w\-\.]+\.(ts|tsx|js|jsx|mjs|cjs|py|rb|go|rs|java):(\d+)(:\d+)?/);
      if (shortFileLineMatch) {
        const ref = shortFileLineMatch[0];
        if (!seen.has(ref)) {
          seen.add(ref);
          this.add(ref, { tags: ["file-reference"], source: "task" });
        }
        continue;
      }
    }

    // Persist learned set with LRU eviction (oldest by learnedAt removed first)
    try {
      const dir = path.dirname(learnedFile);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(learnedFile, JSON.stringify(learned), "utf8");
    } catch { /* ignore */ }
  }

  /**
   * Search with basic relevance scoring and ranking.
   * Scores by: exact match > start-of-line match > tag match > recency.
   * Returns top 20 results sorted by score descending.
   */
  search(query: string): PatternEntry[] {
    if (!query.trim()) return this.all() as PatternEntry[];

    const q = query.toLowerCase();
    const results = this.entries
      .map((e): PatternEntry & { _score: number } => {
        const entry = e as PatternEntry;
        const content = entry.content.toLowerCase();
        const tags = entry.tags.map((t) => t.toLowerCase());

        let score = 0;

        // Exact match (case-insensitive)
        if (content === q) score += 100;
        // Starts with query
        else if (content.startsWith(q)) score += 60;
        // Contains query as word boundary
        else if (content.includes(` ${q}`) || content.includes(`${q} `)) score += 40;
        // Contains query anywhere
        else if (content.includes(q)) score += 20;

        // Tag match bonus
        if (tags.some((t) => t === q)) score += 30;
        else if (tags.some((t) => t.includes(q))) score += 10;

        // Shorter content = more specific = slight bonus (normalized)
        score += Math.max(0, 10 - Math.floor(entry.content.length / 100));

        // Recency bonus (entries created in last 7 days get a bump)
        const ageMs = Date.now() - new Date(entry.createdAt).getTime();
        if (ageMs < 7 * 24 * 60 * 60 * 1000) score += 5;

        return { ...entry, _score: score };
      })
      .filter((e) => e._score > 0)
      .sort((a, b) => b._score - a._score)
      .slice(0, 20);

    // Strip internal score field
    return results.map(({ _score: _, ...rest }) => rest);
  }

  /**
   * Legacy query — simple substring filter (kept for backward compatibility).
   */
  query(searchText: string): MemoryEntry[] {
    if (!searchText.trim()) return this.entries.slice(0, 50);
    const q = searchText.toLowerCase();
    return this.entries.filter(
      (e) =>
        e.content.toLowerCase().includes(q) ||
        e.tags.some((t) => t.toLowerCase().includes(q))
    );
  }
}

export const patternMemory = new PatternMemory();
