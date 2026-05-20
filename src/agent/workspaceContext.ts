import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

export type WorkspaceContextOptions = {
  readonly workspacePath: string;
  readonly query?: string;
  readonly maxFiles?: number;
  readonly maxBytesPerFile?: number;
};

const preferredFiles = new Set([
  "CLAUDE.md",
  "README.md",
  "README.zh-CN.md",
  "package.json",
  "docs",
  "requirements"
]);

const supportedExtensions = new Set([".md", ".json", ".txt"]);

export async function buildWorkspaceContext(options: WorkspaceContextOptions): Promise<string> {
  const maxFiles = options.maxFiles ?? 12;
  const maxBytesPerFile = options.maxBytesPerFile ?? 4000;
  const workspacePath = path.resolve(options.workspacePath);
  const files = await discoverContextFiles(workspacePath, maxFiles, options.query ?? "");
  const sections: string[] = [];

  for (const filePath of files) {
    const relativePath = path.relative(workspacePath, filePath);
    const content = await readFile(filePath, "utf8");
    sections.push(
      [`--- ${relativePath} ---`, truncateByBytes(content.trim(), maxBytesPerFile)].join("\n")
    );
  }

  if (sections.length === 0) {
    return "No readable workspace context files were found.";
  }

  return sections.join("\n\n");
}

async function discoverContextFiles(
  workspacePath: string,
  maxFiles: number,
  query: string
): Promise<string[]> {
  const files: string[] = [];
  await collectFiles(workspacePath, workspacePath, files, maxFiles, query);
  return files;
}

async function collectFiles(
  rootPath: string,
  currentPath: string,
  files: string[],
  maxFiles: number,
  query: string
): Promise<void> {
  if (files.length >= maxFiles) {
    return;
  }

  const entries = await readdir(currentPath, { withFileTypes: true });
  const sortedEntries = entries.sort((left, right) => scoreEntry(left.name) - scoreEntry(right.name));

  for (const entry of sortedEntries) {
    if (files.length >= maxFiles || shouldSkip(entry.name)) {
      continue;
    }

    const entryPath = path.join(currentPath, entry.name);
    if (entry.isDirectory()) {
      if (isContextDirectory(entry.name) || currentPath !== rootPath) {
        await collectFiles(rootPath, entryPath, files, maxFiles, query);
      }
      continue;
    }

    if (entry.isFile() && isContextFile(entry.name) && isRelevantFile(entryPath, query)) {
      const fileStat = await stat(entryPath);
      if (fileStat.size > 0) {
        files.push(entryPath);
      }
    }
  }
}

function isRelevantFile(filePath: string, query: string): boolean {
  const normalizedPath = filePath.toLowerCase();
  const normalizedQuery = query.toLowerCase();
  const asksForTesting =
    /\b(test|tests|testing|entrypoint|entrypoints|health|browser|wechat|callback)\b/.test(normalizedQuery);

  if (normalizedPath.includes("test-entrypoints")) {
    return asksForTesting;
  }

  return true;
}

function scoreEntry(name: string): number {
  if (preferredFiles.has(name)) {
    return 0;
  }
  if (isContextDirectory(name)) {
    return 1;
  }
  if (isContextFile(name)) {
    return 2;
  }
  return 3;
}

function isContextDirectory(name: string): boolean {
  return name === "docs" || name === "requirements" || name === ".claude";
}

function isContextFile(name: string): boolean {
  return preferredFiles.has(name) || supportedExtensions.has(path.extname(name));
}

function shouldSkip(name: string): boolean {
  return name === ".git" || name === "node_modules" || name === "dist" || name === "coverage";
}

function truncateByBytes(value: string, maxBytes: number): string {
  const buffer = Buffer.from(value, "utf8");
  if (buffer.byteLength <= maxBytes) {
    return value;
  }

  return `${buffer.subarray(0, maxBytes).toString("utf8")}\n[truncated]`;
}
