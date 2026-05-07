import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { homedir } from "os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "path";
import { parse as parseYaml } from "yaml";

/** Max name length per spec */
const MAX_NAME_LENGTH = 64;
/** Max description length per spec */
const MAX_DESCRIPTION_LENGTH = 1024;

const IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"];

function toPosixPath(p: string): string {
  return p.split(sep).join("/");
}

/**
 * Minimal .gitignore pattern matcher (subset of gitignore spec).
 * Returns true if the path matches any ignore pattern.
 */
function matchesIgnorePattern(patterns: string[], path: string, isDir: boolean): boolean {
  for (const pattern of patterns) {
    const negated = pattern.startsWith("!");
    const p = negated ? pattern.slice(1) : pattern;

    // Directory-only pattern
    const dirOnly = p.endsWith("/");
    const cleanP = dirOnly ? p.slice(0, -1) : p;

    if (dirOnly && !isDir) continue;

    // Glob: **/foo or foo/** or **/foo/**
    if (cleanP === "**") {
      if (!negated) return true;
      continue;
    }

    if (cleanP.startsWith("**/")) {
      const rest = cleanP.slice(3);
      if (path.endsWith(rest) || path.includes("/" + rest)) {
        if (!negated) return true;
      }
      continue;
    }

    if (cleanP.endsWith("/**")) {
      const prefix = cleanP.slice(0, -3);
      if (path.startsWith(prefix) || path.startsWith(prefix + "/")) {
        if (!negated) return true;
      }
      continue;
    }

    // Normalize leading slash: /foo means root-only
    if (cleanP.startsWith("/")) {
      if (path === cleanP.slice(1) || path.startsWith(cleanP.slice(1) + "/")) {
        if (!negated) return true;
      }
      continue;
    }

    // General: match anywhere
    if (path === cleanP || path.endsWith("/" + cleanP) || path.startsWith(cleanP + "/") || path.includes("/" + cleanP + "/")) {
      if (!negated) return true;
    }
  }
  return false;
}

interface IgnorePatterns {
  add(patterns: string[]): void;
  ignores(path: string, isDir: boolean): boolean;
}

function createIgnorePatterns(): IgnorePatterns {
  const patterns: string[] = [];
  return {
    add(pats: string[]) { patterns.push(...pats); },
    ignores(path: string, isDir: boolean) {
      return matchesIgnorePattern(patterns, path, isDir);
    },
  };
}

function addIgnoreRules(ig: IgnorePatterns, dir: string, rootDir: string): void {
  const relativeDir = relative(rootDir, dir);
  const prefix = relativeDir ? `${toPosixPath(relativeDir)}/` : "";

  for (const filename of IGNORE_FILE_NAMES) {
    const ignorePath = join(dir, filename);
    if (!existsSync(ignorePath)) continue;
    try {
      const content = readFileSync(ignorePath, "utf-8");
      const patterns = content
        .split(/\r?\n/)
        .map((line) => {
          const trimmed = line.trim();
          if (!trimmed) return null;
          if (trimmed.startsWith("#") && !trimmed.startsWith("\\#")) return null;
          let pattern = trimmed.startsWith("!") ? trimmed.slice(1) : trimmed;
          if (pattern.startsWith("\\!")) pattern = pattern.slice(1);
          if (pattern.startsWith("/")) pattern = pattern.slice(1);
          const prefixed = prefix ? `${prefix}${pattern}` : pattern;
          return trimmed.startsWith("!") ? `!${prefixed}` : prefixed;
        })
        .filter((line): line is string => Boolean(line));
      if (patterns.length > 0) ig.add(patterns);
    } catch (err) {
      console.warn(`[SkillLoader] Failed to read ignore file ${ignorePath}:`, err);
    }
  }
}

export interface SkillFrontmatter {
  name?: string;
  description?: string;
  "disable-model-invocation"?: boolean;
  compatibility?: string;
  license?: string;
  metadata?: Record<string, unknown>;
  "allowed-tools"?: string;
  [key: string]: unknown;
}

export interface Skill {
  name: string;
  description: string;
  filePath: string;
  baseDir: string;
  disableModelInvocation: boolean;
  source: string;
}

export interface LoadSkillsResult {
  skills: Skill[];
  diagnostics: SkillDiagnostic[];
}

export type SkillDiagnostic = {
  type: "warning" | "collision" | "error";
  message: string;
  path: string;
  collision?: {
    resourceType: string;
    name: string;
    winnerPath: string;
    loserPath: string;
  };
};

function validateName(name: string, parentDirName: string): string[] {
  const errors: string[] = [];
  if (name !== parentDirName) {
    errors.push(`name "${name}" does not match parent directory "${parentDirName}"`);
  }
  if (name.length > MAX_NAME_LENGTH) {
    errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
  }
  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push(`name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)`);
  }
  if (name.startsWith("-") || name.endsWith("-")) {
    errors.push(`name must not start or end with a hyphen`);
  }
  if (name.includes("--")) {
    errors.push(`name must not contain consecutive hyphens`);
  }
  return errors;
}

function validateDescription(description: string | undefined): string[] {
  const errors: string[] = [];
  if (!description || description.trim() === "") {
    errors.push("description is required");
  } else if (description.length > MAX_DESCRIPTION_LENGTH) {
    errors.push(`description exceeds ${MAX_DESCRIPTION_LENGTH} characters (${description.length})`);
  }
  return errors;
}

function parseFrontmatter(content: string): { frontmatter: SkillFrontmatter; body: string } {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!normalized.startsWith("---")) return { frontmatter: {}, body: normalized };

  const endIndex = normalized.indexOf("\n---", 3);
  if (endIndex === -1) return { frontmatter: {}, body: normalized };

  const yamlString = normalized.slice(4, endIndex);
  const body = normalized.slice(endIndex + 4).trim();
  try {
    const parsed = parseYaml(yamlString);
    return { frontmatter: (parsed ?? {}) as SkillFrontmatter, body };
  } catch (error) {
    console.warn(`[SkillLoader] Failed to parse frontmatter: ${error instanceof Error ? error.message : String(error)}`);
    return { frontmatter: {}, body };
  }
}

function loadSkillFromFile(
  filePath: string,
  source: string,
): { skill: Skill | null; diagnostics: SkillDiagnostic[] } {
  const diagnostics: SkillDiagnostic[] = [];

  try {
    const rawContent = readFileSync(filePath, "utf-8");
    const { frontmatter } = parseFrontmatter(rawContent);
    const skillDir = dirname(filePath);
    const parentDirName = basename(skillDir);

    for (const error of validateDescription(frontmatter.description)) {
      diagnostics.push({ type: "warning", message: error, path: filePath });
    }

    const name = frontmatter.name || parentDirName;

    for (const error of validateName(name, parentDirName)) {
      diagnostics.push({ type: "warning", message: error, path: filePath });
    }

    if (!frontmatter.description || frontmatter.description.trim() === "") {
      return { skill: null, diagnostics };
    }

    return {
      skill: {
        name,
        description: frontmatter.description,
        filePath,
        baseDir: skillDir,
        disableModelInvocation: frontmatter["disable-model-invocation"] === true,
        source,
      },
      diagnostics,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "failed to parse skill file";
    diagnostics.push({ type: "error", message, path: filePath });
    return { skill: null, diagnostics };
  }
}

function loadSkillsFromDirInternal(
  dir: string,
  source: string,
  includeRootFiles: boolean,
  ignoreInstance?: IgnorePatterns,
  rootDir?: string,
): LoadSkillsResult {
  const skills: Skill[] = [];
  const diagnostics: SkillDiagnostic[] = [];

  if (!existsSync(dir)) return { skills, diagnostics };

  const root = rootDir ?? dir;
  const ig = ignoreInstance ?? createIgnorePatterns();
  addIgnoreRules(ig, dir, root);

  try {
    const entries = readdirSync(dir, { withFileTypes: true });

    // Phase 1: SKILL.md files (stop after finding = skill root, don't recurse)
    for (const entry of entries) {
      if (entry.name !== "SKILL.md") continue;

      const fullPath = join(dir, entry.name);
      let isFile = entry.isFile();
      if (entry.isSymbolicLink()) {
        try {
          isFile = statSync(fullPath).isFile();
        } catch {
          continue;
        }
      }

      const relPath = toPosixPath(relative(root, fullPath));
      if (!isFile || ig.ignores(relPath, false)) continue;

      const result = loadSkillFromFile(fullPath, source);
      if (result.skill) skills.push(result.skill);
      diagnostics.push(...result.diagnostics);
      return { skills, diagnostics };
    }

    // Phase 2: directories + root .md files
    for (const entry of entries) {
      if (entry.name.startsWith(".")) continue;
      if (entry.name === "node_modules") continue;

      const fullPath = join(dir, entry.name);

      let isDirectory = entry.isDirectory();
      let isFile = entry.isFile();
      if (entry.isSymbolicLink()) {
        try {
          const stats = statSync(fullPath);
          isDirectory = stats.isDirectory();
          isFile = stats.isFile();
        } catch {
          continue;
        }
      }

      const relPath = toPosixPath(relative(root, fullPath));
      if (ig.ignores(relPath, isDirectory)) continue;

      if (isDirectory) {
        const subResult = loadSkillsFromDirInternal(fullPath, source, false, ig, root);
        skills.push(...subResult.skills);
        diagnostics.push(...subResult.diagnostics);
        continue;
      }

      if (!isFile || !includeRootFiles || !entry.name.endsWith(".md")) continue;
      const result = loadSkillFromFile(fullPath, source);
      if (result.skill) skills.push(result.skill);
      diagnostics.push(...result.diagnostics);
    }
  } catch {}

  return { skills, diagnostics };
}

function canonicalizePath(p: string): string {
  try {
    return resolve(p);
  } catch {
    return p;
  }
}

export function loadSkillsFromDir(dir: string, source: string = "path"): LoadSkillsResult {
  return loadSkillsFromDirInternal(dir, source, true, createIgnorePatterns());
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * Format skills for system prompt using Anthropic Agent Skills XML format.
 * Per https://agentskills.io/integrate-skills
 */
export function formatSkillsForPrompt(skills: Skill[]): string {
  const visible = skills.filter((s) => !s.disableModelInvocation);
  if (visible.length === 0) return "";

  const lines = [
    "\n\nThe following skills provide specialized instructions for specific tasks.",
    "Use the read tool to load a skill's file when the task matches its description.",
    "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
    "",
    "<available_skills>",
  ];

  for (const skill of visible) {
    lines.push("  <skill>");
    lines.push(`    <name>${escapeXml(skill.name)}</name>`);
    lines.push(`    <description>${escapeXml(skill.description)}</description>`);
    lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
    lines.push("  </skill>");
  }

  lines.push("</available_skills>");
  return lines.join("\n");
}

/**
 * Load all skills from configured directories.
 */
export function loadSkills(skillDirs: string[]): LoadSkillsResult {
  const skillMap = new Map<string, Skill>();
  const realPathSet = new Set<string>();
  const allDiagnostics: SkillDiagnostic[] = [];

  for (const rawPath of skillDirs) {
    const trimmed = rawPath.trim();
    let resolved: string;
    if (trimmed === "~") resolved = homedir();
    else if (trimmed.startsWith("~/")) resolved = join(homedir(), trimmed.slice(2));
    else resolved = isAbsolute(trimmed) ? trimmed : resolve(process.cwd(), trimmed);

    if (!existsSync(resolved)) {
      allDiagnostics.push({ type: "warning", message: "skill dir does not exist", path: resolved });
      continue;
    }

    const result = loadSkillsFromDirInternal(resolved, "path", true, createIgnorePatterns());
    for (const diag of result.diagnostics) allDiagnostics.push(diag);

    for (const skill of result.skills) {
      const realPath = canonicalizePath(skill.filePath);
      if (realPathSet.has(realPath)) continue;

      const existing = skillMap.get(skill.name);
      if (existing) {
        allDiagnostics.push({
          type: "collision",
          message: `name "${skill.name}" collision`,
          path: skill.filePath,
          collision: {
            resourceType: "skill",
            name: skill.name,
            winnerPath: existing.filePath,
            loserPath: skill.filePath,
          },
        });
      } else {
        skillMap.set(skill.name, skill);
        realPathSet.add(realPath);
      }
    }
  }

  return {
    skills: Array.from(skillMap.values()),
    diagnostics: allDiagnostics,
  };
}
