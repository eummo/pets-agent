/**
 * WeChat Smart Bot adapter using @wecom/aibot-node-sdk (WebSocket long connection).
 *
 * This adapter replaces the previous HTTP callback (XML self-built app) approach.
 * It connects to WeChat's WebSocket server, receives messages, bridges them
 * to the MessageGateway, and streams responses back.
 *
 * Architecture: channel adapter -> core contracts.
 */
import { WSClient, generateReqId } from "@wecom/aibot-node-sdk";
import { formatUnknownError } from "../core/unknownRecord.js";
import type { WsFrame, TextMessage, EventMessage } from "@wecom/aibot-node-sdk";
import type { ConversationLogger, InboundMessage, MessageGateway } from "../core/index.js";
import type { AgentStreamPublisher } from "../agent/index.js";
import { SessionLock } from "./sessionLock.js";

export type WechatAdapterConfig = {
  readonly botId: string;
  readonly secret: string;
  readonly messageHandler: MessageGateway;
  readonly conversationLogger?: ConversationLogger;
  readonly eventLogger?: ConversationLogger;
  /**
   * Custom WebSocket URL (for private deployment).
   * Defaults to wss://openws.work.weixin.qq.com
   */
  readonly wsUrl?: string;
  /**
   * Reconnect base interval in ms. Default: 1000
   */
  readonly reconnectInterval?: number;
  /**
   * Max reconnect attempts. -1 for infinite. Default: 10
   */
  readonly maxReconnectAttempts?: number;
  /**
   * Max inflight messages per user session before rejecting with "please wait".
   * Default: 3 (matches WeChat platform limit).
   */
  readonly maxInflightPerSession?: number;
};

export class WechatSmartBotAdapter {
  private readonly wsClient: WSClient;
  private readonly sessionLock = new SessionLock();

  public constructor(private readonly config: WechatAdapterConfig) {
    this.wsClient = new WSClient({
      botId: config.botId,
      secret: config.secret,
      ...(config.wsUrl !== undefined ? { wsUrl: config.wsUrl } : {}),
      ...(config.reconnectInterval !== undefined
        ? { reconnectInterval: config.reconnectInterval }
        : {}),
      ...(config.maxReconnectAttempts !== undefined
        ? { maxReconnectAttempts: config.maxReconnectAttempts }
        : {})
    });

    this.registerHandlers();
  }

  public connect(): this {
    this.wsClient.connect();
    return this;
  }

  public disconnect(): void {
    this.wsClient.disconnect();
  }

  public get isConnected(): boolean {
    return this.wsClient.isConnected;
  }

  /**
   * Send a message to a user or group chat via the smart bot SDK.
   * Used by cron delivery to proactively push results.
   */
  public async sendProactiveMessage(targetId: string, content: string): Promise<void> {
    await this.wsClient.sendMessage(targetId, {
      msgtype: "markdown",
      markdown: { content },
    });
  }

  private registerHandlers(): void {
    this.wsClient.on("authenticated", () => {
      void this.logEvent("wechat.authenticated", {});
    });

    this.wsClient.on("disconnected", (reason: string) => {
      void this.logEvent("wechat.disconnected", { reason });
    });

    this.wsClient.on("reconnecting", (attempt: number) => {
      void this.logEvent("wechat.reconnecting", { attempt });
    });

    this.wsClient.on("error", (error: Error) => {
      void this.logEvent("wechat.error", { message: error.message });
    });

    // Handle text messages
    this.wsClient.on("message.text", (frame: WsFrame<TextMessage>) => {
      void this.handleTextMessage(frame);
    });

    // Handle enter chat event - send welcome message
    this.wsClient.on("event.enter_chat", (frame: WsFrame<EventMessage>) => {
      this.handleEnterChat(frame);
    });
  }

  private async handleTextMessage(frame: WsFrame<TextMessage>): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const userId = body.from.userid;
    const content = body.text.content;

    // Strip @bot mention prefix from content if present in group chat
    const cleanContent = stripBotMention(content);

    const chatId = body.chattype === "group" ? body.chatid : undefined;
    const replyTarget = chatId ?? userId;
    const streamId = generateReqId("stream");
    const streamState: { inbound: InboundMessage | undefined } = { inbound: undefined };

    // Accumulate text deltas for intermediate stream updates.
    let accumulated = "";
    const streamCallback: AgentStreamPublisher = (event) => {
      if (event.type === "text_delta" && streamState.inbound !== undefined) {
        accumulated += event.text;
        void this.replyStreamBestEffort(
          frame,
          streamId,
          accumulated,
          false,
          streamState.inbound,
          "delta"
        );
      }
    };

    const inbound: InboundMessage = {
      id: body.msgid,
      channel: "wechat-work",
      user: { id: userId },
      text: cleanContent,
      receivedAt: body.create_time !== undefined ? new Date(body.create_time * 1000) : new Date(),
      stream: streamCallback,
      ...(chatId !== undefined ? { chatId } : {}),
      chatType: body.chattype
    };
    streamState.inbound = inbound;

    await this.logConversation(inbound, "processing");

    // Send "thinking" feedback without letting transport backpressure drop the message.
    await this.replyStreamBestEffort(frame, streamId, "思考中...", false, inbound, "initial");

    // Serialize per-user messages: if a previous message from the same
    // user/session is still being processed, wait for it to complete first.
    // This prevents Claude SDK session collision and history interleaving.
    const lockKey = `wechat-work:${userId}${chatId !== undefined ? `:${chatId}` : ""}`;

    // Reject immediately if too many inflight messages for this session
    const maxInflight = this.config.maxInflightPerSession ?? 3;
    if (this.sessionLock.inflightFor(lockKey) >= maxInflight) {
      await this.sendReply(
        frame,
        streamId,
        replyTarget,
        inbound,
        "请稍等，您还有消息正在处理中，请等我回复后再发送新消息。",
        "rejection"
      );
      await this.logConversation(inbound, "rejected: too many inflight messages");
      return;
    }

    const release = await this.sessionLock.acquire(lockKey);

    try {
      const response = await this.config.messageHandler.handle(inbound);
      // Send the final response with finish=true, replacing any intermediate content
      const finalContent = accumulated.length > 0 ? accumulated : response.text;
      await this.sendReply(frame, streamId, replyTarget, inbound, finalContent, "final");
      await this.logConversation(inbound, response.text);
    } catch (error) {
      await this.logConversation(inbound, `WeChat adapter error: ${formatUnknownError(error)}`);
      // Try to send an error reply
      await this.sendReply(
        frame,
        streamId,
        replyTarget,
        inbound,
        "抱歉，处理消息时出错，请稍后重试。",
        "error"
      );
    } finally {
      release();
    }
  }

  private handleEnterChat(frame: WsFrame<EventMessage>): void {
    void this.wsClient.replyWelcome(frame, {
      msgtype: "text",
      text: { content: "您好！我是知识库助手，可以帮您查询知识库内容。请问有什么可以帮您的？" }
    });
  }

  private async logConversation(message: InboundMessage, output: string): Promise<void> {
    await this.config.conversationLogger?.write({
      type: "conversation.turn",
      channel: message.channel,
      messageId: message.id,
      userId: message.user.id,
      chatId: message.chatId,
      input: message.text,
      output
    });
  }

  private async logEvent(type: string, data: Record<string, unknown>): Promise<void> {
    await this.config.eventLogger?.write({
      type,
      ...data
    });
  }

  private async sendReply(
    frame: WsFrame<TextMessage>,
    streamId: string,
    replyTarget: string,
    message: InboundMessage,
    content: string,
    phase: "rejection" | "final" | "error"
  ): Promise<void> {
    const streamed = await this.replyStreamBestEffort(
      frame,
      streamId,
      content,
      true,
      message,
      phase
    );
    if (streamed) {
      return;
    }
    await this.sendMessageFallback(replyTarget, message, content, phase);
  }

  private async replyStreamBestEffort(
    frame: WsFrame<TextMessage>,
    streamId: string,
    content: string,
    finish: boolean,
    message: InboundMessage,
    phase: "initial" | "delta" | "rejection" | "final" | "error"
  ): Promise<boolean> {
    try {
      await this.wsClient.replyStream(frame, streamId, content, finish);
      return true;
    } catch (error) {
      await this.logMessageEvent("wechat.reply_stream_failed", message, {
        phase,
        error: formatUnknownError(error)
      });
      return false;
    }
  }

  private async sendMessageFallback(
    replyTarget: string,
    message: InboundMessage,
    content: string,
    phase: "rejection" | "final" | "error"
  ): Promise<void> {
    try {
      await this.wsClient.sendMessage(replyTarget, {
        msgtype: "markdown",
        markdown: { content }
      });
      await this.logMessageEvent("wechat.fallback_message_sent", message, { phase, replyTarget });
    } catch (error) {
      await this.logMessageEvent("wechat.fallback_message_failed", message, {
        phase,
        replyTarget,
        error: formatUnknownError(error)
      });
    }
  }

  private async logMessageEvent(
    type: string,
    message: InboundMessage,
    data: Record<string, unknown>
  ): Promise<void> {
    await this.logEvent(type, {
      channel: message.channel,
      messageId: message.id,
      userId: message.user.id,
      chatId: message.chatId,
      ...data
    });
  }
}

/**
 * Strip @bot mention prefix from group chat messages.
 * WeChat group messages start with "@BotName " when the bot is mentioned.
 */
export function stripBotMention(content: string): string {
  // Match @ mention at the start of the message, followed by a space
  const mentionPattern = /^@\S+\s*/;
  return content.replace(mentionPattern, "").trim();
}
