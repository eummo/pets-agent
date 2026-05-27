import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { FastifyInstance } from "fastify";
import { isLocalRequest } from "./serverUtils.js";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const devChatDir = path.join(projectRoot, "static", "dev-chat");

const mimeTypes: Record<string, string> = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

export function registerDevStaticRoutes(server: FastifyInstance): void {
  server.get("/", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send("Development UI is only available from localhost.");
    }

    const html = await readFile(path.join(devChatDir, "index.html"), "utf8");
    return reply.type("text/html; charset=utf-8").send(html);
  });

  server.get("/dev/chat/*", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send("Development assets are only available from localhost.");
    }

    const relativePath = (request.params as Record<string, string>)["*"] ?? "";
    const filePath = path.resolve(devChatDir, relativePath);

    if (isPathOutsideDirectory(filePath, devChatDir)) {
      return reply.status(403).send("Forbidden");
    }

    const fileStat = await stat(filePath).catch(() => undefined);
    if (!fileStat?.isFile()) {
      return reply.status(404).send("Not found");
    }

    const ext = path.extname(filePath);
    const contentType = mimeTypes[ext];
    if (!contentType) {
      return reply.status(415).send("Unsupported media type");
    }

    const content = await readFile(filePath);
    return reply.type(`${contentType}; charset=utf-8`).send(content);
  });
}

function isPathOutsideDirectory(filePath: string, directoryPath: string): boolean {
  // Normalize Windows paths to WSL mount points (e.g. "D:/foo" -> "/mnt/d/foo")
  // so the relative-path check works correctly in WSL environments.
  // Node.js path.isAbsolute() doesn't recognize Windows drive letters on Linux/WSL,
  // which could allow path traversal to bypass this check.
  const normalized = filePath.replace(/^[a-zA-Z]:[/\\]/, '/mnt/');
  const relativePath = path.relative(directoryPath, normalized);
  return relativePath.startsWith("..") || path.isAbsolute(relativePath);
}
