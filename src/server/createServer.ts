import type { ServerResponse } from "node:http";
import Fastify, { type FastifyInstance } from "fastify";
import type { AgentStreamEvent, MessageHandler, UserRole } from "../core/ports.js";
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

function writeSse(response: ServerResponse, event: string, data: unknown): void {
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}

function renderDevChatPage(): string {
  return String.raw`<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Pets Agent</title>
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
      h1 { margin: 0; font-size: 18px; font-weight: 650; }
      .subtitle { margin-top: 4px; color: #5c6678; font-size: 13px; }
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
        line-height: 1.5;
      }
      .message.user {
        border-color: #b9c7dc;
        background: #eef4ff;
        white-space: pre-wrap;
      }
      .message.agent {
        white-space: pre-wrap;
      }
      .message.error {
        border-color: #e3b3b3;
        background: #fff0f0;
        white-space: pre-wrap;
      }
      .message.system {
        border-color: #dce2eb;
        background: #f9fafb;
        white-space: pre-wrap;
      }
      .meta {
        display: block;
        margin-bottom: 6px;
        color: #627086;
        font-size: 12px;
        font-weight: 650;
      }
      .tool-call {
        border: 1px solid #d9dee7;
        background: #f9fafb;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 13px;
      }
      .tool-call-header {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        cursor: pointer;
        user-select: none;
      }
      .tool-call-header:hover { background: #eef0f4; border-radius: 6px 6px 0 0; }
      .tool-call-icon { color: #627086; }
      .tool-call-name { font-weight: 600; color: #3d4f6f; }
      .tool-call-summary { color: #5c6678; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .tool-call-toggle { color: #8b98aa; font-size: 11px; }
      .tool-call-body { padding: 8px 10px; border-top: 1px solid #dce2eb; display: none; }
      .tool-call-body.open { display: block; }
      .tool-call-body pre { margin: 0; white-space: pre-wrap; word-break: break-all; font-size: 12px; color: #3d4f6f; }
      .thinking-indicator {
        display: inline-block;
        color: #8b98aa;
        font-style: italic;
        font-size: 13px;
      }
      .thinking-indicator::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
      }
      @keyframes dots {
        0%   { content: ''; }
        25%  { content: '.'; }
        50%  { content: '..'; }
        75%  { content: '...'; }
      }
      form {
        position: sticky;
        bottom: 0;
        display: grid;
        grid-template-columns: 160px 160px 1fr auto;
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
      textarea { min-height: 44px; max-height: 160px; resize: vertical; }
      button {
        border: 0;
        border-radius: 8px;
        padding: 0 18px;
        background: #2257a5;
        color: #ffffff;
        font-weight: 650;
        cursor: pointer;
      }
      button:disabled { background: #8b98aa; cursor: wait; }
      @media (max-width: 720px) {
        form { grid-template-columns: 1fr; padding: 12px 16px; }
        button { min-height: 42px; }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>Pets Agent</h1>
      <div class="subtitle">基于 Claude Agent SDK 的角色化助手。文档助手查询知识库，开发助手修改代码并验证。</div>
    </header>
    <main>
      <section id="messages" aria-live="polite">
        <div class="message system">
          <span class="meta">system</span>
          服务已就绪。选择角色开始对话：文档助手只能查看知识库；开发助手可以读取、修改代码并运行验证。
        </div>
      </section>
    </main>
    <form id="chat-form">
      <input id="user-id" value="browser-user" aria-label="User ID" />
      <select id="role-select" aria-label="Role">
        <option value="reviewer" selected>文档助手 (reviewer)</option>
        <option value="developer">开发助手 (developer)</option>
      </select>
      <textarea id="message-input" placeholder="输入消息..." aria-label="Message"></textarea>
      <button id="send-button" type="submit">发送</button>
    </form>
    <script>
      const form = document.querySelector("#chat-form");
      const messages = document.querySelector("#messages");
      const userIdEl = document.querySelector("#user-id");
      const roleSelect = document.querySelector("#role-select");
      const input = document.querySelector("#message-input");
      const button = document.querySelector("#send-button");

      // ── SSE progress channel (existing) ──
      let eventsSource;
      function connectEvents() {
        if (eventsSource) eventsSource.close();
        eventsSource = new EventSource("/dev/events?userId=" + encodeURIComponent(userIdEl.value));
        eventsSource.addEventListener("progress", (event) => {
          const payload = JSON.parse(event.data);
          if (payload.stage?.startsWith("agent.")) return; // handled by chat SSE
          const details = payload.data ? "\n" + JSON.stringify(payload.data, null, 2) : "";
          addSystemMessage("[" + payload.stage + "] " + payload.message + details);
        });
        eventsSource.onerror = () => {};
      }
      userIdEl.addEventListener("change", connectEvents);
      connectEvents();

      // ── Helpers ──
      function addMessage(role, text) {
        const el = document.createElement("div");
        el.className = "message " + role;
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = role;
        el.append(meta, document.createTextNode(text));
        messages.append(el);
        el.scrollIntoView({ block: "end", behavior: "smooth" });
        return el;
      }

      function addSystemMessage(text) {
        const el = document.createElement("div");
        el.className = "message system";
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = "system";
        el.append(meta, document.createTextNode(text));
        messages.append(el);
        el.scrollIntoView({ block: "end", behavior: "smooth" });
      }

      function createAgentMessage() {
        const el = document.createElement("div");
        el.className = "message agent";
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = "agent";
        const content = document.createElement("div");
        content.className = "agent-content";
        el.append(meta, content);
        messages.append(el);
        return { el, content };
      }

      function createToolCallCard(toolName, input, toolUseId) {
        const card = document.createElement("div");
        card.className = "tool-call";
        card.dataset.toolUseId = toolUseId;

        const header = document.createElement("div");
        header.className = "tool-call-header";
        const icon = document.createElement("span");
        icon.className = "tool-call-icon";
        icon.textContent = "\u{1F527}";
        const name = document.createElement("span");
        name.className = "tool-call-name";
        name.textContent = toolName;
        const summary = document.createElement("span");
        summary.className = "tool-call-summary";
        summary.textContent = " " + JSON.stringify(input).slice(0, 80);
        const toggle = document.createElement("span");
        toggle.className = "tool-call-toggle";
        toggle.textContent = "\u25B6";
        header.append(icon, name, summary, toggle);

        const body = document.createElement("div");
        body.className = "tool-call-body";
        const pre = document.createElement("pre");
        pre.textContent = "Input: " + JSON.stringify(input, null, 2);
        body.append(pre);

        header.addEventListener("click", () => {
          body.classList.toggle("open");
          toggle.textContent = body.classList.contains("open") ? "\u25BC" : "\u25B6";
        });

        card.append(header, body);
        return card;
      }

      function updateToolCallCard(toolUseId, result, isError) {
        const card = document.querySelector('[data-tool-use-id="' + toolUseId + '"]');
        if (!card) return;
        const body = card.querySelector(".tool-call-body");
        if (!body) return;
        const pre = document.createElement("pre");
        pre.textContent = (isError ? "Error: " : "Result: ") + (result || "").slice(0, 2000);
        body.append(pre);
        body.classList.add("open");
        const toggle = card.querySelector(".tool-call-toggle");
        if (toggle) toggle.textContent = "\u25BC";
      }

      // ── Role ──
      async function setRole() {
        const response = await fetch("/dev/role", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ userId: userIdEl.value, role: roleSelect.value })
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Failed to set role");
        return body;
      }

      roleSelect.addEventListener("change", async () => {
        try {
          const body = await setRole();
          addSystemMessage("角色已切换为 " + body.role);
        } catch (error) {
          addMessage("error", error instanceof Error ? error.message : String(error));
        }
      });

      // ── Chat with SSE streaming ──
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
            body: JSON.stringify({ userId: userIdEl.value, text })
          });

          if (!response.ok) {
            const body = await response.json();
            throw new Error(body.error || "Request failed");
          }

          const { el: agentEl, content: agentContent } = createAgentMessage();
          let textContent = "";
          let thinkingEl = null;

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const event = JSON.parse(line.slice(6));
                  switch (event.type) {
                    case "text_delta":
                      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                      textContent += event.text;
                      // render markdown-like: just set text
                      agentContent.textContent = textContent;
                      agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                      break;
                    case "tool_use_start":
                      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                      const card = createToolCallCard(event.toolName, event.input, event.toolUseId);
                      agentContent.append(card);
                      agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                      break;
                    case "tool_use_result":
                      updateToolCallCard(event.toolUseId, event.result, event.isError);
                      agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                      break;
                    case "thinking":
                      if (!thinkingEl) {
                        thinkingEl = document.createElement("span");
                        thinkingEl.className = "thinking-indicator";
                        thinkingEl.textContent = "thinking";
                        agentContent.append(thinkingEl);
                      }
                      agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                      break;
                    case "completed":
                      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                      if (event.text && event.text !== textContent) {
                        textContent = event.text;
                        agentContent.textContent = textContent;
                      }
                      break;
                    case "error":
                      addMessage("error", event.message);
                      break;
                  }
                } catch (parseError) {
                  // skip malformed SSE data
                }
              }
            }
          }
        } catch (error) {
          addMessage("error", error instanceof Error ? error.message : String(error));
        } finally {
          button.disabled = false;
          input.focus();
        }
      });
    </script>
  </body>
</html>`;
}
