import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Fastify, { type FastifyInstance } from "fastify";
import type { AgentStreamEvent, AuthorizationService, FeedbackStore, MessageHandler, RoleConfigStore } from "../core/ports.js";
import type { DevRoleStore } from "../security/devRoleStore.js";
import { verifyWechatSignature } from "../wechat/signature.js";
import { buildWechatTextReply, parseWechatMessage } from "../wechat/xml.js";
import type { DevProgressBroker } from "./progressBroker.js";
import { writeSse } from "./sseUtils.js";

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

export type CreateServerOptions = {
  readonly messageHandler: MessageHandler;
  readonly wechatToken: string;
  readonly devRoleStore?: DevRoleStore;
  readonly roleConfigStore?: RoleConfigStore;
  readonly feedbackStore?: FeedbackStore;
  readonly authorization?: AuthorizationService;
  readonly progressBroker?: DevProgressBroker;
  readonly logger?: boolean;
};

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

type WechatVerifyQuery = {
  readonly msg_signature?: string;
  readonly signature?: string;
  readonly timestamp?: string;
  readonly nonce?: string;
  readonly echostr?: string;
};

export function createServer(options: CreateServerOptions): FastifyInstance {
  const server = Fastify({ logger: options.logger ?? false });

  server.addContentTypeParser(
    ["application/xml", "text/xml", "text/plain"],
    { parseAs: "string" },
    (_request, payload, done) => {
      done(null, payload);
    }
  );

  server.get("/health", () => ({
    ok: true,
    service: "pets-agent"
  }));

  server.get("/", async (_request, reply) => {
    const html = await readFile(path.join(devChatDir, "index.html"), "utf8");
    return reply.type("text/html; charset=utf-8").send(html);
  });

  // Serve dev-chat static assets: /dev/chat/style.css, /dev/chat/app.js, etc.
  server.get("/dev/chat/*", async (request, reply) => {
    const relativePath = (request.params as Record<string, string>)["*"] ?? "";
    const filePath = path.join(devChatDir, relativePath);

    // Prevent path traversal
    if (!filePath.startsWith(devChatDir)) {
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

    options.devRoleStore?.setRole(userId, role);

    return reply.send({
      userId,
      role: options.devRoleStore?.getRole(userId) ?? role
    });
  });

type DevFeedbackBody = {
  readonly status?: string;
  readonly userId?: string;
};

type DevFeedbackQuery = {
  readonly userId?: string;
};

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

    const entries = await options.feedbackStore.getAll();
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

  server.get<{ Querystring: WechatVerifyQuery }>("/wechat/callback", async (request, reply) => {
    const signature = request.query.signature ?? request.query.msg_signature;
    const { timestamp, nonce, echostr } = request.query;

    if (signature === undefined || timestamp === undefined || nonce === undefined || echostr === undefined) {
      return reply.status(400).send("missing wechat verification query");
    }

    const verified = verifyWechatSignature({
      token: options.wechatToken,
      timestamp,
      nonce,
      signature
    });

    if (!verified) {
      return reply.status(401).send("invalid signature");
    }

    return reply.type("text/plain").send(echostr);
  });

  server.post<{ Body: string }>("/wechat/callback", async (request, reply) => {
    const rawBody = typeof request.body === "string" ? request.body : String(request.body);
    const wechatMessage = parseWechatMessage(rawBody);

    if (!("content" in wechatMessage)) {
      return reply
        .type("application/xml")
        .send(
          buildWechatTextReply(
            wechatMessage.fromUserName,
            wechatMessage.toUserName,
            `Only text messages are supported for now. Received: ${wechatMessage.msgType}.`
          )
        );
    }

    const response = await options.messageHandler.handle({
      id: wechatMessage.msgId,
      channel: "wechat-work",
      user: {
        id: wechatMessage.fromUserName
      },
      text: wechatMessage.content,
      receivedAt: new Date(Number.parseInt(wechatMessage.createTime, 10) * 1000)
    });

    return reply
      .type("application/xml")
      .send(buildWechatTextReply(wechatMessage.fromUserName, wechatMessage.toUserName, response.text));
  });

  return server;
}

function normalizeOptionalText(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized === "" ? undefined : normalized;
}

function isLocalRequest(ip: string): boolean {
  return ip === "127.0.0.1"
    || ip === "::1"
    || ip === "::ffff:127.0.0.1";
}
