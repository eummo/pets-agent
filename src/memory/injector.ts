/**
 * MemoryInjector — assembles memory snapshots into a system prompt block.
 *
 * Frozen snapshot model (per Hermes):
 * - Snapshots captured at load time, NOT mutated during session
 * - Real-time writes go to disk but do NOT affect the in-session snapshot
 * - Next session gets fresh snapshot with all persisted entries
 *
 * Caching strategy:
 * - buildBlock() results are cached per workdir with a configurable TTL
 * - Cache is invalidated when the underlying memory files change (mtime-based)
 * - Callers can bypass the cache with { forceRefresh: true }
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { createHash } from "crypto";
import { patternMemory } from "./pattern-memory.js";
import { preferenceMemory } from "./preference-memory.js";
import { projectMemory } from "./project-memory.js";
import {
  DefaultResourceLoader,
  getAgentDir,
} from "@earendil-works/pi-coding-agent";

// ─── Shared DefaultResourceLoader singleton ────────────────────────────────────

let _sharedLoader: DefaultResourceLoader | null = null;

/**
 * Returns the module-level singleton DefaultResourceLoader.
 * All call sites (memory-tools, injector) share the same instance to avoid
 * duplicated skill discovery and inconsistent discovery paths.
 */
export function getSharedResourceLoader(): DefaultResourceLoader {
  if (!_sharedLoader) {
    _sharedLoader = new DefaultResourceLoader({
      cwd: process.cwd(),
      agentDir: getAgentDir(),
    });
  }
  return _sharedLoader;
}

/**
 * Compute the same project-key that ProjectMemory.store() uses internally.
 * Must stay in sync with project-memory.ts projectKey() logic.
 */
function projectKey(workdir: string): string {
  return createHash("sha256").update(workdir).digest("hex").slice(0, 12);
}

export interface InjectorOptions {
  workdir?: string;
  includePatterns?: boolean;
  includePreferences?: boolean;
  includeProject?: boolean;
  /** Include available skills (false by default to keep prompt lean) */
  includeSkills?: boolean;
  /** Bypass the snapshot cache and rebuild immediately (default: false) */
  forceRefresh?: boolean;
}

interface CacheEntry {
  /** Built block text */
  block: string;
  /** TTL in ms */
  ttl: number;
  /** When this entry expires (Date.now() + ttl) */
  expiresAt: number;
  /** mtime of pattern memory file at time of cache creation */
  patternMtime: number;
  /** mtime of preference memory file */
  prefMtime: number;
  /** mtime of project memory file (0 if no project) */
  projectMtime: number;
}

/** Default TTL for buildBlock cache (ms). Default: 60 seconds. */
const DEFAULT_CACHE_TTL_MS = 60_000;

export class MemoryInjector {
  private _cache = new Map<string, CacheEntry>();
  /** Default TTL in ms */
  private _defaultTtlMs: number;

  constructor({ ttlMs = DEFAULT_CACHE_TTL_MS }: { ttlMs?: number } = {}) {
    this._defaultTtlMs = ttlMs;
  }

  /**
   * Cache key for a given workdir. Changes when workdir changes.
   * Note: patterns/prefs have no per-workdir key so their mtimes are baked
   * into the cache entry for invalidation.
   */
  private cacheKey(workdir?: string): string {
    return workdir ?? "__global__";
  }

  /**
   * Get the current mtime of the memory directory's entry file, or 0 if absent.
   * Used to detect when a memory store has been written to since the last cache.
   */
  private static memoryMtime(filename: string): number {
    try {
      const fp = path.join(os.homedir(), ".pets-agent", "memory", filename);
      if (fs.existsSync(fp)) {
        return fs.statSync(fp).mtimeMs;
      }
    } catch {
      /* ignore */
    }
    return 0;
  }

  /**
   * Check whether a cache entry is still valid based on TTL and memory mtimes.
   */
  private isCacheHit(key: string): CacheEntry | null {
    const entry = this._cache.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this._cache.delete(key);
      return null;
    }
    // Invalidate if any backing memory file changed since caching.
    // Note: for project memory, the file is stored as projects/<projectKey>.md
    // where projectKey = sha256(workdir).slice(0,12). We use the same key (workdir
    // or "__global__") which matches how cacheKey() stores it.
    const patMtime = MemoryInjector.memoryMtime("patterns.md");
    const prefMtime = MemoryInjector.memoryMtime("preferences.md");
    const projMtime = key !== "__global__"
      ? MemoryInjector.memoryMtime(`projects/${projectKey(key)}.md`)
      : 0;

    if (
      patMtime !== entry.patternMtime ||
      prefMtime !== entry.prefMtime ||
      projMtime !== entry.projectMtime
    ) {
      this._cache.delete(key);
      return null;
    }
    return entry;
  }

  /**
   * Build a system prompt block with all available memory snapshots.
   * Results are cached per workdir; cache is automatically invalidated when
   * the backing memory files change or the TTL expires.
   */
  buildBlock(opts: InjectorOptions = {}): string {
    const {
      workdir,
      includePatterns = true,
      includePreferences = true,
      includeProject = true,
      includeSkills = false,
      forceRefresh = false,
    } = opts;

    const key = this.cacheKey(workdir);

    if (!forceRefresh) {
      const hit = this.isCacheHit(key);
      if (hit) return hit.block;
    } else {
      this._cache.delete(key);
    }

    const block = this._buildBlock(opts);

    const patMtime = MemoryInjector.memoryMtime("patterns.md");
    const prefMtime = MemoryInjector.memoryMtime("preferences.md");
    // Use the same project-key filename as isCacheHit for consistency
    const projMtime = workdir ? MemoryInjector.memoryMtime(`projects/${projectKey(workdir)}.md`) : 0;

    this._cache.set(key, {
      block,
      ttl: this._defaultTtlMs,
      expiresAt: Date.now() + this._defaultTtlMs,
      patternMtime: patMtime,
      prefMtime,
      projectMtime: projMtime,
    });

    return block;
  }

  /**
   * Actually build the block string (called by buildBlock after cache check).
   */
  private _buildBlock(opts: InjectorOptions): string {
    const {
      workdir,
      includePatterns = true,
      includePreferences = true,
      includeProject = true,
      includeSkills = false,
    } = opts;

    const parts: string[] = [];

    if (includePatterns) {
      const snap = patternMemory.getSnapshot();
      if (snap) parts.push(snap);
    }

    if (includePreferences) {
      const snap = preferenceMemory.getSnapshot();
      if (snap) parts.push(snap);
    }

    if (includeProject && workdir) {
      const store = projectMemory.store(workdir);
      const snap = store.getSnapshot();
      const stack = projectMemory.detectTechStack(workdir);
      const stackLine =
        stack.length > 0 ? `\n[Auto-detected stack: ${stack.join(", ")}]` : "";

      if (snap || stackLine) {
        parts.push(
          `═══════════════════════════════════════════════\n` +
            `PROJECT (${workdir})${stackLine}\n` +
            `═══════════════════════════════════════════════\n` +
            (snap ?? "")
        );
      }
    }

    if (includeSkills) {
      const skillBlock = this._buildSkillBlock();
      if (skillBlock) parts.push(skillBlock);
    }

    if (parts.length === 0) return "";

    return (
      "\n\n" +
      "╔══════════════════════════════════════════════════════╗\n" +
      "║  MEMORY — pets-agent persistent context              ║\n" +
      "╚══════════════════════════════════════════════════════╝\n" +
      parts.join("\n\n") +
      "\n"
    );
  }

  private _skillLoader: DefaultResourceLoader | null = null;

  /** Get (or create) the shared DefaultResourceLoader singleton */
  private _getLoader(): DefaultResourceLoader {
    if (!this._skillLoader) {
      this._skillLoader = getSharedResourceLoader();
    }
    return this._skillLoader;
  }

  /**
   * Build a skill summary block from pi-mono's skill loader.
   * Shows skill names + one-line descriptions for quick discovery.
   * SILENT FAILURE: returns "" on any error so it never breaks prompt injection.
   */
  private _buildSkillBlock(): string {
    try {
      const loader = this._getLoader();
      const { skills, diagnostics } = loader.getSkills();

      if (skills.length === 0) return "";

      const lines = [
        `═══════════════════════════════════════════════`,
        `SKILLS (${skills.length} available)`,
        `═══════════════════════════════════════════════`,
        ``,
        `Available skills — use view_skill(name) for full content:`,
        ``,
      ];

      for (const s of skills) {
        lines.push(`[${s.name}]  ${s.description}`);
      }

      if (diagnostics.length > 0) {
        lines.push(``);
        lines.push(`⚠ Skill warnings:`);
        for (const d of diagnostics) {
          lines.push(`  ${d.message}${d.path ? ` (${d.path})` : ""}`);
        }
      }

      return lines.join("\n");
    } catch {
      // Non-fatal: if skill loading fails, skip the block silently
      return "";
    }
  }

  /**
   * Quick status summary for debugging / "memory" tool output.
   */
  status(): {
    patterns: { count: number; usage: { current: number; limit: number; pct: number } };
    preferences: { count: number; usage: { current: number; limit: number; pct: number } };
  } {
    return {
      patterns: {
        count: patternMemory.all().length,
        usage: patternMemory.usage(),
      },
      preferences: {
        count: preferenceMemory.all().length,
        usage: preferenceMemory.usage(),
      },
    };
  }

  /**
   * Clear the snapshot cache. Useful after bulk memory operations.
   */
  clearCache(): void {
    this._cache.clear();
  }
}

export const memoryInjector = new MemoryInjector();
