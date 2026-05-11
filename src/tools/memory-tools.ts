/**
 * Memory Tools — remember_pattern, remember_prefs, remember_project,
 * get_memory, refresh_memory, forget_memory, list_skills, view_skill
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { DefaultResourceLoader, getAgentDir } from "@earendil-works/pi-coding-agent";
import * as fs from "fs";
import { patternMemory } from "../memory/pattern-memory.js";
import { preferenceMemory } from "../memory/preference-memory.js";
import { projectMemory } from "../memory/project-memory.js";
import { memoryInjector } from "../memory/injector.js";

export function registerMemoryTools(pi: ExtensionAPI): void {
  const RememberPatternParams = Type.Object({
    pattern: Type.String({ description: "Command or workflow pattern to remember" }),
    tags: Type.Optional(Type.String({ description: "Comma-separated tags, e.g. 'npm,build,vite'" })),
  });

  const RememberPrefsParams = Type.Object({
    agentType: Type.String({ description: "Agent type: claude-code, pi-agent, codex, kiro" }),
    taskPrompt: Type.String({ description: "Brief description of the task type" }),
    success: Type.Boolean({ description: "Whether the task succeeded" }),
    exitCode: Type.Optional(Type.Number()),
    durationSec: Type.Optional(Type.Number()),
  });

  const RememberProjectParams = Type.Object({
    workdir: Type.String({ description: "Project directory path" }),
    content: Type.String({ description: "Context to remember about this project" }),
    tags: Type.Optional(Type.String({ description: "Comma-separated tags" })),
  });

  const GetMemoryParams = Type.Object({
    type: Type.Optional(Type.String({ description: "Memory type: patterns, preferences, project. Default: all" })),
    workdir: Type.Optional(Type.String({ description: "Project directory for project memory" })),
    query: Type.Optional(Type.String({ description: "Search query for patterns/preferences" })),
  });

  const RefreshMemoryParams = Type.Object({
    workdir: Type.Optional(Type.String({ description: "Project directory to refresh project memory for" })),
  });

  const ForgetMemoryParams = Type.Object({
    type: Type.String({ description: "Memory type: patterns, preferences", enum: ["patterns", "preferences"] }),
    idOrText: Type.String({ description: "Entry ID or substring to match and remove" }),
  });

  // ─── remember_pattern ──────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "remember_pattern",
    label: "Remember Pattern",
    description: "Save a successful command or workflow pattern for future reference.",
    parameters: RememberPatternParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const tags = params.tags ? params.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];
      const result = patternMemory.add(params.pattern, { tags, source: "user" });

      if (result.success) {
        const usage = patternMemory.usage();
        return {
          content: [{ type: "text", text: `Pattern saved [${usage.pct}% — ${usage.current}/${usage.limit} chars]\n${params.pattern}` }],
          details: { saved: true },
        };
      }
      return {
        content: [{ type: "text", text: `Failed to save pattern: ${result.error}` }],
        details: { saved: false, error: result.error },
      };
    },
  }));

  // ─── remember_prefs ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "remember_prefs",
    label: "Remember Preference",
    description: "Record which agent type succeeded for a given task type.",
    parameters: RememberPrefsParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      preferenceMemory.recordOutcome({
        agentType: params.agentType,
        taskPrompt: params.taskPrompt,
        success: params.success,
        exitCode: params.exitCode,
        durationSec: params.durationSec,
      });
      const suggested = preferenceMemory.suggestAgentType(params.taskPrompt);
      const msg = suggested
        ? `Recorded. Suggested agent for '${params.taskPrompt.slice(0, 50)}': ${suggested}`
        : "Recorded. No strong pattern yet.";
      return { content: [{ type: "text", text: msg }], details: { suggested } };
    },
  }));

  // ─── remember_project ──────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "remember_project",
    label: "Remember Project",
    description: "Save context about a specific project (tech stack, key files, conventions).",
    parameters: RememberProjectParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const tags = params.tags ? params.tags.split(",").map((t) => t.trim()).filter(Boolean) : [];
      const store = projectMemory.store(params.workdir);
      const result = store.addToProject(params.content, { tags, source: "user" });

      if (result.success) {
        const usage = store.usage();
        return {
          content: [{ type: "text", text: `Project context saved for ${params.workdir} [${usage.pct}%]\n${params.content}` }],
          details: { saved: true },
        };
      }
      return {
        content: [{ type: "text", text: `Failed: ${result.error}` }],
        details: { saved: false, error: result.error },
      };
    },
  }));

  // ─── get_memory ───────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "get_memory",
    label: "Get Memory",
    description: "View or search memory stores. Shows status summary by default.",
    parameters: GetMemoryParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const type = params.type ?? "all";

      if (type === "patterns" || type === "all") {
        const query = params.query ?? "";
        const results = patternMemory.search(query);
        const usage = patternMemory.usage();
        const lines = [
          `--- Patterns [${usage.pct}% — ${results.length} entries] ---`,
          ...results.map((e) => {
            const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
            return `${tags}\n${e.content}`;
          }),
        ];
        if (type === "patterns") return { content: [{ type: "text", text: lines.join("\n") }], details: { results } };
      }

      if (type === "preferences" || type === "all") {
        const query = params.query ?? "";
        const results = preferenceMemory.query(query);
        const usage = preferenceMemory.usage();
        const lines = [
          `--- Preferences [${usage.pct}% — ${results.length} entries] ---`,
          ...results.map((e) => {
            const tags = e.tags.length > 0 ? ` [${e.tags.join(", ")}]` : "";
            return `${tags}\n${e.content}`;
          }),
        ];
        if (type === "preferences") return { content: [{ type: "text", text: lines.join("\n") }], details: { results } };
      }

      if (type === "project" && params.workdir) {
        const store = projectMemory.store(params.workdir);
        const snap = store.getSnapshot();
        const stack = projectMemory.detectTechStack(params.workdir);
        const lines = [
          `--- Project: ${params.workdir} [${stack.join(", ") || "unknown stack"}] ---`,
          snap || "(no entries)",
        ];
        return { content: [{ type: "text", text: lines.join("\n") }], details: {} };
      }

      // Default: status overview
      const status = memoryInjector.status();
      return {
        content: [{
          type: "text",
          text: [
            "Memory Status:",
            `  Patterns: ${status.patterns.count} entries, ${status.patterns.usage.pct}%`,
            `  Preferences: ${status.preferences.count} entries, ${status.preferences.usage.pct}%`,
            "",
            "Use get_memory(type='patterns'|'preferences'|'project', workdir='...') to view details.",
          ].join("\n"),
        }],
        details: { status },
      };
    },
  }));

  // ─── refresh_memory ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "refresh_memory",
    label: "Refresh Memory",
    description: "Force reload of all memory snapshots from disk. Use after remember_* calls to see updated memory in current session.",
    parameters: RefreshMemoryParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      patternMemory.load();
      preferenceMemory.load();
      if (params.workdir) {
        projectMemory.store(params.workdir).load();
      }
      const status = memoryInjector.status();
      return {
        content: [{ type: "text", text: "Memory snapshots refreshed." }],
        details: { status },
      };
    },
  }));

  // ─── forget_memory ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "forget_memory",
    label: "Forget Memory",
    description: "Remove a memory entry by ID or content match.",
    parameters: ForgetMemoryParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const result =
        params.type === "patterns"
          ? patternMemory.remove(params.idOrText)
          : preferenceMemory.remove(params.idOrText);

      if (result.success) {
        return { content: [{ type: "text", text: `Removed from ${params.type}: '${params.idOrText}'` }], details: { removed: true } };
      }
      return { content: [{ type: "text", text: result.error ?? "Not found" }], details: { removed: false } };
    },
  }));

  // ─── list_skills ───────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "list_skills",
    label: "List Skills",
    description: "List all available skills loaded from ~/.pi/agent/skills, ./.pi/skills, etc.",
    parameters: Type.Object({
      query: Type.Optional(Type.String({ description: "Filter by name or description" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate) {
      const loader = new DefaultResourceLoader({ cwd: process.cwd(), agentDir: getAgentDir() });
      const { skills, diagnostics } = loader.getSkills();

      let filtered = skills;
      if (params.query) {
        const q = params.query.toLowerCase();
        filtered = skills.filter(
          (s) =>
            s.name.toLowerCase().includes(q) ||
            s.description.toLowerCase().includes(q),
        );
      }

      if (filtered.length === 0) {
        return {
          content: [{ type: "text", text: "No skills found." }],
          details: { skills: [], total: skills.length },
        };
      }

      const lines: string[] = [
        `${filtered.length} skill(s) found (${skills.length} total):`,
        "",
      ];
      for (const s of filtered) {
        lines.push(`[${s.name}]`);
        lines.push(`  ${s.description}`);
        lines.push(`  path: ${s.filePath}`);
        lines.push("");
      }

      if (diagnostics.length > 0) {
        lines.push(`⚠ ${diagnostics.length} warning(s):`);
        for (const d of diagnostics) {
          lines.push(`  ${d.message}${d.path ? ` (${d.path})` : ""}`);
        }
      }

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { skills: filtered, total: skills.length, diagnostics },
      };
    },
  }));

  // ─── view_skill ────────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "view_skill",
    label: "View Skill",
    description: "View the full content of a skill by name.",
    parameters: Type.Object({
      name: Type.String({ description: "Skill name to view" }),
      section: Type.Optional(Type.String({ description: "Section to highlight (optional)" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate) {
      const loader = new DefaultResourceLoader({ cwd: process.cwd(), agentDir: getAgentDir() });
      const { skills } = loader.getSkills();
      const skill = skills.find((s) => s.name === params.name);

      if (!skill) {
        const available = skills.map((s) => s.name).join(", ");
        return {
          content: [
            {
              type: "text",
              text: `Skill not found: "${params.name}"\n\nAvailable: ${available || "(none)"}`,
            },
          ],
          details: { error: "not_found", available },
        };
      }

      let content: string;
      try {
        content = fs.readFileSync(skill.filePath, "utf-8");
      } catch {
        content = `(Could not read file: ${skill.filePath})`;
      }

      return {
        content: [
          {
            type: "text",
            text: [
              `=== ${skill.name} ===`,
              `path: ${skill.filePath}`,
              `description: ${skill.description}`,
              "",
              content,
            ].join("\n"),
          },
        ],
        details: { skill },
      };
    },
  }));
}
