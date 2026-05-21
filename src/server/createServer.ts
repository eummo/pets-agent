import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Fastify, { type FastifyInstance } from "fastify";
import type { AgentStreamEvent, MessageHandler, UserRole } from "../core/ports.js";
import type { DevRoleStore } from "../security/devRoleStore.js";
import { verifyWechatSignature } from "../wechat/signature.js";
import { buildWechatTextReply, parseWechatMessage } from "../wechat/xml.js";
import type { DevProgressBroker } from "./progressBroker.js";
import { writeSse } from "./sseUtils.js";

const devChatDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "dev-chat");

export type CreateServerOptions = {
  readonly messageHandler: MessageHandler;
  readonly wechatToken: string;
  readonly devRoleStore?: DevRoleStore;
  readonly progressBroker?: DevProgressBroker;
  readonly logger?: boolean;
};

type DevChatBody = {
  readonly userId?: string;
  readonly text?: string;
};

type DevRoleBody = {
  readonly userId?: string;
  readonly role?: UserRole;
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

  server.get("/dev/chat/style.css", async (_request, reply) => {
    const css = await readFile(path.join(devChatDir, "style.css"), "utf8");
    return reply.type("text/css; charset=utf-8").send(css);
  });

  server.get("/dev/chat/app.js", async (_request, reply) => {
    const js = await readFile(path.join(devChatDir, "app.js"), "utf8");
    return reply.type("text/javascript; charset=utf-8").send(js);
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

    if (role !== "viewer" && role !== "reviewer" && role !== "developer") {
      return reply.status(400).send({ error: "Role must be reviewer or developer." });
    }

    options.devRoleStore?.setRole(userId, role);

    return reply.send({
      userId,
      role: options.devRoleStore?.getRole(userId) ?? role
    });
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
