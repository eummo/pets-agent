import { MessageType } from "@wecom/aibot-node-sdk";
import type { TextMessage, WsFrame } from "@wecom/aibot-node-sdk";
import { describe, expect, it, vi } from "vitest";
import { stripBotMention } from "./wechatSmartBotAdapter.js";
import { WechatSmartBotAdapter } from "./wechatSmartBotAdapter.js";
import { SessionLock } from "./sessionLock.js";
import type { ConversationLogger, MessageGateway, OutboundMessage } from "../core/index.js";

describe("stripBotMention", () => {
  it("strips @bot mention prefix from group chat messages", () => {
    expect(stripBotMention("@RobotA hello world")).toBe("hello world");
  });

  it("strips @mention with underscore in bot name", () => {
    expect(stripBotMention("@My_Bot some question")).toBe("some question");
  });

  it("does not strip @mention in the middle of a message", () => {
    expect(stripBotMention("hello @bot world")).toBe("hello @bot world");
  });

  it("returns original text when no @mention prefix", () => {
    expect(stripBotMention("just a question")).toBe("just a question");
  });

  it("handles empty string", () => {
    expect(stripBotMention("")).toBe("");
  });

  it("handles @mention only with trailing space", () => {
    expect(stripBotMention("@Bot ")).toBe("");
  });
});

describe("stream accumulation logic", () => {
  it("accumulates text_delta events and ignores other event types", () => {
    let accumulated = "";
    const streamCallback = (event: { type: string; text?: string }) => {
      if (event.type === "text_delta" && event.text) {
        accumulated += event.text;
      }
    };

    streamCallback({ type: "text_delta", text: "Hello " });
    streamCallback({ type: "text_delta", text: "World" });
    streamCallback({ type: "thinking", text: "pondering" });
    streamCallback({ type: "text_delta", text: "!" });

    expect(accumulated).toBe("Hello World!");
  });

  it("uses response.text when no text_delta events arrived", () => {
    const accumulated = "";
    const responseText = "fallback response";
    const finalContent = accumulated.length > 0 ? accumulated : responseText;
    expect(finalContent).toBe("fallback response");
  });

  it("uses accumulated text when text_delta events arrived", () => {
    const accumulated = "streamed content";
    const responseText = "fallback response";
    const finalContent = accumulated.length > 0 ? accumulated : responseText;
    expect(finalContent).toBe("streamed content");
  });
});

describe("SessionLock", () => {
  it("allows a single acquire and release", async () => {
    const lock = new SessionLock();
    const release = await lock.acquire("user-a");
    expect(lock.activeLockCount()).toBe(1);
    release();
    // Allow microtask queue to flush
    await Promise.resolve();
    expect(lock.activeLockCount()).toBe(0);
  });

  it("serializes concurrent acquires on the same key", async () => {
    const lock = new SessionLock();
    const order: string[] = [];

    const op1 = (async () => {
      const release = await lock.acquire("user-a");
      order.push("a1-start");
      await new Promise((r) => setTimeout(r, 50));
      order.push("a1-end");
      release();
    })();

    const op2 = (async () => {
      const release = await lock.acquire("user-a");
      order.push("a2-start");
      order.push("a2-end");
      release();
    })();

    await Promise.all([op1, op2]);

    // op2 must start after op1 ends
    expect(order).toEqual(["a1-start", "a1-end", "a2-start", "a2-end"]);
  });

  it("allows concurrent acquires on different keys", async () => {
    const lock = new SessionLock();
    const order: string[] = [];

    const op1 = (async () => {
      const release = await lock.acquire("user-a");
      order.push("a-start");
      await new Promise((r) => setTimeout(r, 50));
      order.push("a-end");
      release();
    })();

    const op2 = (async () => {
      const release = await lock.acquire("user-b");
      order.push("b-start");
      order.push("b-end");
      release();
    })();

    await Promise.all([op1, op2]);

    // Different keys should run concurrently, so b starts before a ends
    expect(order.indexOf("b-start")).toBeLessThan(order.indexOf("a-end"));
  });

  it("handles release idempotently", async () => {
    const lock = new SessionLock();
    const release = await lock.acquire("user-a");
    release();
    release(); // second call should be a no-op
    await Promise.resolve();
    expect(lock.activeLockCount()).toBe(0);
  });

  it("tracks inflight count per key", async () => {
    const lock = new SessionLock();

    expect(lock.inflightFor("user-a")).toBe(0);

    const r1 = await lock.acquire("user-a");
    expect(lock.inflightFor("user-a")).toBe(1);

    // Second acquire is queued but still counted as inflight
    const p2 = lock.acquire("user-a");
    expect(lock.inflightFor("user-a")).toBe(2);

    r1();
    const r2 = await p2;
    expect(lock.inflightFor("user-a")).toBe(1);

    r2();
    await Promise.resolve();
    expect(lock.inflightFor("user-a")).toBe(0);
  });
});

describe("WechatSmartBotAdapter replies", () => {
  it("continues processing and sends a fallback message when stream replies fail", async () => {
    const conversationEvents: Record<string, unknown>[] = [];
    const systemEvents: Record<string, unknown>[] = [];
    let handled = false;

    const messageHandler: MessageGateway = {
      handle(): Promise<OutboundMessage> {
        handled = true;
        return Promise.resolve({ text: "最终答案" });
      },
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      conversationLogger: collectingLogger(conversationEvents),
      eventLogger: collectingLogger(systemEvents),
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.reject(new Error("stream busy"))),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "sent" } } as WsFrame)),
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(groupTextFrame("msg-1", "user-1", "group-1", "@Bot 你好"));

    expect(handled).toBe(true);
    expect(fakeClient.replyStream).toHaveBeenCalled();
    expect(fakeClient.sendMessage).toHaveBeenCalledWith("group-1", {
      msgtype: "markdown",
      markdown: { content: "最终答案" },
    });
    expect(conversationEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ messageId: "msg-1", chatId: "group-1", output: "processing" }),
        expect.objectContaining({ messageId: "msg-1", chatId: "group-1", output: "最终答案" }),
      ]),
    );
    expect(systemEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "wechat.reply_stream_failed", phase: "initial", messageId: "msg-1", chatId: "group-1" }),
        expect.objectContaining({ type: "wechat.fallback_message_sent", phase: "final", messageId: "msg-1", chatId: "group-1" }),
      ]),
    );
  });

  it("passes chatType group for group chat messages", async () => {
    let capturedChatType: string | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedChatType = message.chatType;
        return Promise.resolve({ text: "ok" });
      },
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(groupTextFrame("msg-g", "user-1", "group-1", "hello"));

    expect(capturedChatType).toBe("group");
  });

  it("passes chatType single for single chat messages", async () => {
    let capturedChatType: string | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedChatType = message.chatType;
        return Promise.resolve({ text: "ok" });
      },
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(singleTextFrame("msg-s", "user-2", "hello"));

    expect(capturedChatType).toBe("single");
  });
});

function collectingLogger(events: Record<string, unknown>[]): ConversationLogger {
  return {
    write(event: Record<string, unknown>): Promise<void> {
      events.push(event);
      return Promise.resolve();
    },
  };
}

function groupTextFrame(
  messageId: string,
  userId: string,
  chatId: string,
  content: string,
): WsFrame<TextMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chatid: chatId,
      chattype: "group",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.Text,
      text: { content },
    },
  };
}

function singleTextFrame(
  messageId: string,
  userId: string,
  content: string,
): WsFrame<TextMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chattype: "single",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.Text,
      text: { content },
    },
  };
}
