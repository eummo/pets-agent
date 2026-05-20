import Fastify, { type FastifyInstance } from "fastify";
import type { MessageHandler, UserRole } from "../core/ports.js";
import type { DevRoleStore } from "../security/devRoleStore.js";
import { verifyWechatSignature } from "../wechat/signature.js";
import { buildWechatTextReply, parseWechatMessage } from "../wechat/xml.js";
import type { DevProgressBroker } from "./progressBroker.js";

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

  server.get("/", (_request, reply) => reply.type("text/html; charset=utf-8").send(renderDevChatPage()));

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

    const response = await options.messageHandler.handle({
      id: `dev-${Date.now()}`,
      channel: "dev-browser",
      user: {
        id: userId
      },
      text,
      receivedAt: new Date()
    });

    return reply.send({
      text: response.text
    });
  });

  server.post<{ Body: DevRoleBody }>("/dev/role", async (request, reply) => {
    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";
    const role = request.body.role;

    if (role !== "viewer" && role !== "developer") {
      return reply.status(400).send({ error: "Role must be viewer or developer." });
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

function renderDevChatPage(): string {
  return String.raw`<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Pets Agent Dev Chat</title>
    <style>
      :root {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f6f7f9;
        color: #18202f;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr auto;
      }
      header {
        padding: 18px 24px;
        border-bottom: 1px solid #d9dee7;
        background: #ffffff;
      }
      h1 {
        margin: 0;
        font-size: 18px;
        font-weight: 650;
      }
      .subtitle {
        margin-top: 4px;
        color: #5c6678;
        font-size: 13px;
      }
      main {
        width: min(920px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 24px 0;
      }
      #messages {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .message {
        border: 1px solid #dce2eb;
        background: #ffffff;
        border-radius: 8px;
        padding: 12px 14px;
        white-space: pre-wrap;
        line-height: 1.5;
      }
      .message.user {
        border-color: #b9c7dc;
        background: #eef4ff;
      }
      .message.progress {
        border-color: #cbd7c0;
        background: #f3faef;
      }
      .message.error {
        border-color: #e3b3b3;
        background: #fff0f0;
      }
      .meta {
        display: block;
        margin-bottom: 6px;
        color: #627086;
        font-size: 12px;
        font-weight: 650;
      }
      form {
        position: sticky;
        bottom: 0;
        display: grid;
        grid-template-columns: 160px 140px 1fr auto;
        gap: 10px;
        padding: 14px 24px;
        border-top: 1px solid #d9dee7;
        background: #ffffff;
      }
      input, textarea, button, select { font: inherit; }
      input, textarea, select {
        border: 1px solid #cfd6e3;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
        color: #18202f;
      }
      textarea {
        min-height: 44px;
        max-height: 160px;
        resize: vertical;
      }
      button {
        border: 0;
        border-radius: 8px;
        padding: 0 18px;
        background: #2257a5;
        color: #ffffff;
        font-weight: 650;
        cursor: pointer;
      }
      button:disabled {
        background: #8b98aa;
        cursor: wait;
      }
      @media (max-width: 720px) {
        form {
          grid-template-columns: 1fr;
          padding: 12px 16px;
        }
        button {
          min-height: 42px;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>Pets Agent Dev Chat</h1>
      <div class="subtitle">本地浏览器测试入口，支持切换 viewer/developer 角色并实时查看代码变更进度。</div>
    </header>
    <main>
      <section id="messages" aria-live="polite">
        <div class="message">
          <span class="meta">system</span>
          服务已就绪。普通用户只能查询知识库；开发者可以进入代码变更流程测试。
        </div>
      </section>
    </main>
    <form id="chat-form">
      <input id="user-id" value="browser-user" aria-label="User ID" />
      <select id="role-select" aria-label="Role">
        <option value="viewer" selected>viewer</option>
        <option value="developer">developer</option>
      </select>
      <textarea id="message-input" placeholder="输入测试消息..." aria-label="Message"></textarea>
      <button id="send-button" type="submit">发送</button>
    </form>
    <script>
      const form = document.querySelector("#chat-form");
      const messages = document.querySelector("#messages");
      const userId = document.querySelector("#user-id");
      const roleSelect = document.querySelector("#role-select");
      const input = document.querySelector("#message-input");
      const button = document.querySelector("#send-button");
      let events;

      function addMessage(role, text) {
        const item = document.createElement("div");
        item.className = role === "user" ? "message user" : role === "progress" ? "message progress" : role === "error" ? "message error" : "message";
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = role;
        item.append(meta, document.createTextNode(text));
        messages.append(item);
        item.scrollIntoView({ block: "end", behavior: "smooth" });
      }

      function connectEvents() {
        if (events) events.close();
        events = new EventSource("/dev/events?userId=" + encodeURIComponent(userId.value));
        events.addEventListener("progress", (event) => {
          const payload = JSON.parse(event.data);
          const details = payload.data ? "\n" + JSON.stringify(payload.data, null, 2) : "";
          addMessage("progress", "[" + payload.stage + "] " + payload.message + details);
        });
        events.onerror = () => {
          addMessage("progress", "[events.disconnected] 实时进度通道已断开，浏览器会自动重连。");
        };
      }

      async function setRole() {
        const response = await fetch("/dev/role", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ userId: userId.value, role: roleSelect.value })
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Failed to set role");
        return body;
      }

      roleSelect.addEventListener("change", async () => {
        try {
          const body = await setRole();
          addMessage("system", "角色已切换为 " + body.role);
        } catch (error) {
          addMessage("error", error instanceof Error ? error.message : String(error));
        }
      });

      userId.addEventListener("change", connectEvents);

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = input.value.trim();
        if (!text) return;
        input.value = "";
        button.disabled = true;
        addMessage("user", text);
        try {
          await setRole();
          const response = await fetch("/dev/chat", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ userId: userId.value, text })
          });
          const body = await response.json();
          if (!response.ok) throw new Error(body.error || "Request failed");
          addMessage("agent", body.text);
        } catch (error) {
          addMessage("error", error instanceof Error ? error.message : String(error));
        } finally {
          button.disabled = false;
          input.focus();
        }
      });

      connectEvents();
    </script>
  </body>
</html>`;
}
