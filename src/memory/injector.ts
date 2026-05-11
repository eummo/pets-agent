/**
 * MemoryInjector — assembles memory snapshots into a system prompt block.
 *
 * Frozen snapshot model (per Hermes):
 * - Snapshots captured at load time, NOT mutated during session
 * - Real-time writes go to disk but do NOT affect the in-session snapshot
 * - Next session gets fresh snapshot with all persisted entries
 */

import { patternMemory } from "./pattern-memory.js";
import { preferenceMemory } from "./preference-memory.js";
import { projectMemory } from "./project-memory.js";
import { DefaultResourceLoader, getAgentDir } from "@earendil-works/pi-coding-agent";

export interface InjectorOptions {
  workdir?: string;
  includePatterns?: boolean;
  includePreferences?: boolean;
  includeProject?: boolean;
  /** Include available skills (false by default to keep prompt lean) */
  includeSkills?: boolean;
}

export class MemoryInjector {
  /**
   * Build a system prompt block with all available memory snapshots.
   */
  buildBlock(opts: InjectorOptions = {}): string {
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
      const stackLine = stack.length > 0
        ? `\n[Auto-detected stack: ${stack.join(", ")}]`
        : "";

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
      const skillBlock = this.#buildSkillBlock();
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

  #skillLoader: DefaultResourceLoader | null = null;

  #getLoader(): DefaultResourceLoader {
    if (!this.#skillLoader) {
      this.#skillLoader = new DefaultResourceLoader({
        cwd: process.cwd(),
        agentDir: getAgentDir(),
      });
    }
    return this.#skillLoader;
  }

  /**
   * Build a skill summary block from pi-mono's skill loader.
   * Shows skill names + one-line descriptions for quick discovery.
   */
  #buildSkillBlock(_workdir?: string): string {
    try {
      const loader = this.#getLoader();
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
}

export const memoryInjector = new MemoryInjector();
