import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { FastifyInstance } from "fastify";
import type { AgentStreamEvent, AuthorizationService, FeedbackStore, MessageGateway, RoleConfigStore } from "../core/contracts.js";
import type { SseProgressBroker } from "./sseProgressBroker.js";
import { isLocalRequest, normalizeOptionalText } from "./serverUtils.js";
import { writeSse } from "./sseUtils.js";

type DevChatBody = {
  readonly userId?: string;
  readonly text?: string;
};

type DevRoleBody = {
  readonly userId?: string;
  readonly role?: string;
};

type DevEventsQuery = {
  readonly userId?: string;
};

type DevFeedbackBody = {
  readonly status?: string;
  readonly userId?: string;
};

type DevFeedbackQuery = {
  readonly userId?: string;
  readonly limit?: string;
  readonly offset?: string;
  readonly status?: string;
};

export type DevRoutesOptions = {
  readonly messageHandler: MessageGateway;
  readonly roleConfigStore?: RoleConfigStore | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly authorization?: AuthorizationService | undefined;
  readonly progressBroker?: SseProgressBroker | undefined;
};

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

export function registerDevRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  // Serve dev-chat page
  server.get("/", async (_request, reply) => {
    const html = await readFile(path.join(devChatDir, "index.html"), "utf8");
    return reply.type("text/html; charset=utf-8").send(html);
  });

  // Serve dev-chat static assets: /dev/chat/style.css, /dev/chat/app.js, etc.
  server.get("/dev/chat/*", async (request, reply) => {
    const relativePath = (request.params as Record<string, string>)["*"] ?? "";
    const filePath = path.resolve(devChatDir, relativePath);

    // Prevent path traversal
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

  server.get("/dev/roles", async () => {
    if (options.roleConfigStore === undefined) {
      return { roles: [] };
    }
    const configs = await options.roleConfigStore.getAll();
    return { roles: configs.map((c) => ({ name: c.name, capabilities: c.capabilities ?? [] })) };
  });

  server.get<{ Querystring: DevEventsQuery }>("/dev/events", async (request, reply) => {
    const userId = normalizeOptionalText(request.query.userId) ?? "browser-user";
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no"
    });
    reply.raw.write("\n");

    const unsubscribe = options.progressBroker?.subscribe(userId, reply.raw);
    request.raw.on("close", () => {
      unsubscribe?.();
    });

    return reply;
  });

  server.post<{ Body: DevChatBody }>("/dev/chat", async (request, reply) => {
    const text = request.body.text?.trim();
    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";

    if (text === undefined || text.length === 0) {
      return reply.status(400).send({ error: "Message text is required." });
    }

    // SSE streaming response
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no"
    });
    reply.raw.write("\n");

    const streamCallback = (event: AgentStreamEvent): void => {
      writeSse(reply.raw, "agent", event);
    };

    try {
      const response = await options.messageHandler.handle({
        id: `dev-${Date.now()}`,
        channel: "dev-browser",
        user: { id: userId },
        text,
        receivedAt: new Date(),
        stream: streamCallback,
      });

      writeSse(reply.raw, "agent", {
        type: "completed",
        sessionId: response.sessionId,
        text: response.text,
      });
    } catch (error) {
      writeSse(reply.raw, "agent", {
        type: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      reply.raw.end();
    }

    return reply;
  });

  server.post<{ Body: DevRoleBody }>("/dev/role", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send({ error: "Role management is only available from localhost." });
    }

    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";
    const role = request.body.role;

    if (typeof role !== "string" || role.trim().length === 0) {
      return reply.status(400).send({ error: "Role must be a non-empty string." });
    }

    // Validate role exists in config store if available
    if (options.roleConfigStore !== undefined) {
      const config = await options.roleConfigStore.getByName(role);
      if (config === undefined) {
        return reply.status(400).send({ error: `Unknown role: ${role}` });
      }
    }

    if (options.authorization?.setRole === undefined) {
      return reply.status(501).send({ error: "Role management is not supported by this authorization service." });
    }
    options.authorization.setRole(userId, role);

    const currentRole = await options.authorization.roleFor({ id: userId });
    return reply.send({
      userId,
      role: currentRole
    });
  });

  server.get<{ Querystring: DevFeedbackQuery }>("/dev/feedback", async (request, reply) => {
    if (options.feedbackStore === undefined || options.authorization === undefined) {
      return reply.status(501).send({ error: "Feedback management is not configured." });
    }
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send({ error: "Feedback management is only available from localhost." });
    }

    const userId = normalizeOptionalText(request.query.userId) ?? "browser-user";
    const hasView = await options.authorization.hasCapability({ id: userId }, "feedback_view");
    if (!hasView) {
      return reply.status(403).send({ error: "Insufficient permissions to view feedback." });
    }

    const entries = await options.feedbackStore.getAll(feedbackQueryFrom(request.query));
    return { feedback: entries };
  });

  server.patch<{ Params: { id: string }; Body: DevFeedbackBody }>("/dev/feedback/:id", async (request, reply) => {
    if (options.feedbackStore === undefined || options.authorization === undefined) {
      return reply.status(501).send({ error: "Feedback management is not configured." });
    }
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send({ error: "Feedback management is only available from localhost." });
    }

    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";
    const hasManage = await options.authorization.hasCapability({ id: userId }, "feedback_manage");
    if (!hasManage) {
      return reply.status(403).send({ error: "Insufficient permissions to manage feedback." });
    }

    const id = Number.parseInt(request.params.id, 10);
    if (Number.isNaN(id)) {
      return reply.status(400).send({ error: "Invalid feedback ID." });
    }

    const status = request.body.status;
    if (status !== "reviewed" && status !== "resolved") {
      return reply.status(400).send({ error: "Status must be 'reviewed' or 'resolved'." });
    }

    const updated = await options.feedbackStore.updateStatus(id, status);
    if (!updated) {
      return reply.status(404).send({ error: "Feedback entry not found." });
    }

    return { id, status };
  });
}

function parsePositiveInteger(value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function parseNonNegativeInteger(value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

function parseFeedbackStatus(value: string | undefined): "pending" | "reviewed" | "resolved" | undefined {
  return value === "pending" || value === "reviewed" || value === "resolved" ? value : undefined;
}

function feedbackQueryFrom(query: DevFeedbackQuery) {
  const limit = parsePositiveInteger(query.limit);
  const offset = parseNonNegativeInteger(query.offset);
  const status = parseFeedbackStatus(query.status);
  return {
    ...(limit !== undefined ? { limit } : {}),
    ...(offset !== undefined ? { offset } : {}),
    ...(status !== undefined ? { status } : {}),
  };
}

function isPathOutsideDirectory(filePath: string, directoryPath: string): boolean {
  const relativePath = path.relative(directoryPath, filePath);
  return relativePath.startsWith("..") || path.isAbsolute(relativePath);
}

