import { MessageType } from "@wecom/aibot-node-sdk";
import type {
  FileMessage,
  ImageMessage,
  MixedMessage,
  TextMessage,
  WsFrame
} from "@wecom/aibot-node-sdk";
import { describe, expect, it, vi } from "vitest";
import { mkdtemp, readFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import {
  formatProactiveWechatNotificationContent,
  sanitizeOutgoingWechatContent,
  stripBotMention
} from "./wechatSmartBotAdapter.js";
import { WechatSmartBotAdapter } from "./wechatSmartBotAdapter.js";
import { SessionLock } from "./sessionLock.js";
import type {
  ConversationLogger,
  InboundMessage,
  MessageGateway,
  OutboundMessage
} from "../core/index.js";

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

describe("sanitizeOutgoingWechatContent", () => {
  it("strips smart bot mention markup that renders as literal text", () => {
    expect(
      sanitizeOutgoingWechatContent("<@wohR_KCgAAMvvV2XiALPE4KfNY-jz2kA> 我一直在的呀！")
    ).toBe("我一直在的呀！");
  });

  it("strips multiple mention tokens while preserving normal content", () => {
    expect(sanitizeOutgoingWechatContent("请 <@user-1> 和 <@user_2> 看一下")).toBe("请 和 看一下");
  });
});

describe("formatProactiveWechatNotificationContent", () => {
  it("converts notification @mentions to smart bot mention markup", () => {
    expect(formatProactiveWechatNotificationContent("@zhangsan please review")).toBe(
      "<@zhangsan> please review"
    );
    expect(formatProactiveWechatNotificationContent("cc @li.si and @user-2")).toBe(
      "cc <@li.si> and <@user-2>"
    );
  });

  it("preserves existing mention markup and email addresses", () => {
    expect(
      formatProactiveWechatNotificationContent("notify <@zhangsan> via ops@example.com")
    ).toBe("notify <@zhangsan> via ops@example.com");
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
  it("returns the current group cron delivery channel without invoking the gateway", async () => {
    const systemEvents: Record<string, unknown>[] = [];
    const handle = vi.fn(() => Promise.resolve({ text: "should not run" }));
    const messageHandler: MessageGateway = {
      handle
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      eventLogger: collectingLogger(systemEvents)
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "sent" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(
      groupTextFrame("msg-channel", "user-1", "group-1", "@Bot 获取当前群聊投递渠道")
    );

    expect(handle).not.toHaveBeenCalled();
    expect(fakeClient.replyStream).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(String),
      expect.stringContaining("wecom:chat:group-1"),
      true
    );
    expect(fakeClient.sendMessage).not.toHaveBeenCalled();
    expect(systemEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "wechat.channel_info_requested",
          messageId: "msg-channel",
          chatId: "group-1"
        }),
        expect.objectContaining({
          type: "wechat.channel_info_sent",
          messageId: "msg-channel",
          chatId: "group-1",
          replyTarget: "group-1"
        })
      ])
    );
  });

  it("sends proactive bot notifications with working @mention markup", async () => {
    const messageHandler: MessageGateway = {
      handle(): Promise<OutboundMessage> {
        return Promise.resolve({ text: "ok" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler
    });

    const fakeClient = {
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "sent" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
    };
    privateAdapter.wsClient = fakeClient;

    await adapter.sendProactiveMessage("group-1", "Build failed, @zhangsan please check.");

    expect(fakeClient.sendMessage).toHaveBeenCalledWith("group-1", {
      msgtype: "markdown",
      markdown: { content: "Build failed, <@zhangsan> please check." }
    });
  });

  it("continues processing and sends a fallback message when stream replies fail", async () => {
    const conversationEvents: Record<string, unknown>[] = [];
    const systemEvents: Record<string, unknown>[] = [];
    let handled = false;

    const messageHandler: MessageGateway = {
      handle(): Promise<OutboundMessage> {
        handled = true;
        return Promise.resolve({ text: "最终答案" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      eventLogger: collectingLogger(systemEvents)
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.reject(new Error("stream busy"))),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "sent" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(
      groupTextFrame("msg-1", "user-1", "group-1", "@Bot 你好")
    );

    expect(handled).toBe(true);
    expect(fakeClient.replyStream).toHaveBeenCalled();
    expect(fakeClient.sendMessage).toHaveBeenCalledWith("group-1", {
      msgtype: "markdown",
      markdown: { content: "最终答案" }
    });
    expect(conversationEvents).toEqual([]);
    expect(systemEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "wechat.message_processing",
          messageId: "msg-1",
          chatId: "group-1"
        }),
        expect.objectContaining({
          type: "wechat.reply_stream_failed",
          phase: "initial",
          messageId: "msg-1",
          chatId: "group-1"
        }),
        expect.objectContaining({
          type: "wechat.fallback_message_sent",
          phase: "final",
          messageId: "msg-1",
          chatId: "group-1"
        }),
        expect.objectContaining({
          type: "wechat.message_completed",
          messageId: "msg-1",
          chatId: "group-1"
        })
      ])
    );
  });

  it("passes chatType group for group chat messages", async () => {
    let capturedChatType: string | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedChatType = message.chatType;
        return Promise.resolve({ text: "ok" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame))
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
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler
    });

    const fakeClient = {
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleTextMessage(frame: WsFrame<TextMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleTextMessage(singleTextFrame("msg-s", "user-2", "hello"));

    expect(capturedChatType).toBe("single");
  });

  it("downloads image messages and passes image attachments to the gateway", async () => {
    const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-wechat-uploads-"));
    const imageBytes = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    let capturedMessage: InboundMessage | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedMessage = message;
        return Promise.resolve({ text: "ok" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      uploadRootPath
    });

    const fakeClient = {
      downloadFile: vi.fn(() => Promise.resolve({ buffer: imageBytes })),
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleImageMessage(frame: WsFrame<ImageMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleImageMessage(
      imageFrame("msg-img", "user-1", "image-url", "aes-key")
    );

    expect(fakeClient.downloadFile).toHaveBeenCalledWith("image-url", "aes-key");
    expect(capturedMessage?.text).toBe("请描述这张图片。");
    const attachment = capturedMessage?.attachments?.[0];
    expect(attachment).toMatchObject({
      type: "image",
      name: "wechat-image-1.png",
      mimeType: "image/png",
      sizeBytes: imageBytes.length
    });
    expect(attachment?.storagePath.startsWith(uploadRootPath)).toBe(true);
    await expect(readFile(attachment?.storagePath ?? "")).resolves.toEqual(imageBytes);
  });

  it("downloads file messages and passes document attachments to the gateway", async () => {
    const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-wechat-uploads-"));
    const documentBytes = Buffer.from("# Notes\nEnterprise WeChat attachment.", "utf8");
    let capturedMessage: InboundMessage | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedMessage = message;
        return Promise.resolve({ text: "ok" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      uploadRootPath
    });

    const fakeClient = {
      downloadFile: vi.fn(() => Promise.resolve({ buffer: documentBytes, filename: "notes.md" })),
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleFileMessage(frame: WsFrame<FileMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleFileMessage(fileFrame("msg-file", "user-1", "file-url", "file-key"));

    expect(fakeClient.downloadFile).toHaveBeenCalledWith("file-url", "file-key");
    expect(capturedMessage?.text).toBe("请根据上传的文档回答。");
    const attachment = capturedMessage?.attachments?.[0];
    expect(attachment).toMatchObject({
      type: "document",
      name: "notes.md",
      mimeType: "text/markdown",
      sizeBytes: documentBytes.length
    });
    expect(attachment?.storagePath.startsWith(uploadRootPath)).toBe(true);
    await expect(readFile(attachment?.storagePath ?? "", "utf8")).resolves.toBe(
      documentBytes.toString("utf8")
    );
  });

  it("combines mixed text and image items into one inbound message", async () => {
    const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-wechat-uploads-"));
    const imageBytes = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    let capturedMessage: InboundMessage | undefined;
    const messageHandler: MessageGateway = {
      handle(message): Promise<OutboundMessage> {
        capturedMessage = message;
        return Promise.resolve({ text: "ok" });
      }
    };

    const adapter = new WechatSmartBotAdapter({
      botId: "bot-id",
      secret: "secret",
      messageHandler,
      uploadRootPath
    });

    const fakeClient = {
      downloadFile: vi.fn(() => Promise.resolve({ buffer: imageBytes, filename: "diagram.png" })),
      replyStream: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame)),
      sendMessage: vi.fn(() => Promise.resolve({ headers: { req_id: "ok" } } as WsFrame))
    };

    const privateAdapter = adapter as unknown as {
      wsClient: typeof fakeClient;
      handleMixedMessage(frame: WsFrame<MixedMessage>): Promise<void>;
    };
    privateAdapter.wsClient = fakeClient;

    await privateAdapter.handleMixedMessage(
      mixedFrame("msg-mixed", "user-1", "@Bot 请看这张图", "mixed-image-url", "mixed-key")
    );

    expect(capturedMessage?.text).toBe("请看这张图");
    expect(capturedMessage?.attachments?.[0]).toMatchObject({
      type: "image",
      name: "diagram.png",
      mimeType: "image/png"
    });
  });
});

function collectingLogger(events: Record<string, unknown>[]): ConversationLogger {
  return {
    write(event: Record<string, unknown>): Promise<void> {
      events.push(event);
      return Promise.resolve();
    }
  };
}

function groupTextFrame(
  messageId: string,
  userId: string,
  chatId: string,
  content: string
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
      text: { content }
    }
  };
}

function singleTextFrame(messageId: string, userId: string, content: string): WsFrame<TextMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chattype: "single",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.Text,
      text: { content }
    }
  };
}

function imageFrame(
  messageId: string,
  userId: string,
  url: string,
  aeskey: string
): WsFrame<ImageMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chattype: "single",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.Image,
      image: { url, aeskey }
    }
  };
}

function fileFrame(
  messageId: string,
  userId: string,
  url: string,
  aeskey: string
): WsFrame<FileMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chattype: "single",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.File,
      file: { url, aeskey }
    }
  };
}

function mixedFrame(
  messageId: string,
  userId: string,
  text: string,
  imageUrl: string,
  imageAeskey: string
): WsFrame<MixedMessage> {
  return {
    headers: { req_id: `req-${messageId}` },
    body: {
      msgid: messageId,
      aibotid: "bot-id",
      chattype: "single",
      from: { userid: userId },
      create_time: 1_779_786_805,
      msgtype: MessageType.Mixed,
      mixed: {
        msg_item: [
          { msgtype: "text", text: { content: text } },
          { msgtype: "image", image: { url: imageUrl, aeskey: imageAeskey } }
        ]
      }
    }
  };
}
