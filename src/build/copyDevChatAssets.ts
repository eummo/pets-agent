import { copyFile, mkdir, readdir } from "node:fs/promises";
import path from "node:path";

const projectRoot = process.cwd();
const sourceDir = path.join(projectRoot, "src", "server", "dev-chat");
const targetDir = path.join(projectRoot, "dist", "server", "dev-chat");

async function copyDirectory(source: string, target: string): Promise<void> {
  await mkdir(target, { recursive: true });

  const entries = await readdir(source, { withFileTypes: true });
  for (const entry of entries) {
    const sourcePath = path.join(source, entry.name);
    const targetPath = path.join(target, entry.name);

    if (entry.isDirectory()) {
      await copyDirectory(sourcePath, targetPath);
    } else if (entry.isFile()) {
      await copyFile(sourcePath, targetPath);
    }
  }
}

try {
  await copyDirectory(sourceDir, targetDir);
} catch (error) {
  console.error("Failed to copy dev chat assets.", error);
  process.exitCode = 1;
}
