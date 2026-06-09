/**
 * WeChat Smart Bot adapter using the WeCom SDK wrapper (WebSocket long connection).
 *
 * This adapter replaces the previous HTTP callback (XML self-built app) approach.
 * It connects to WeChat's WebSocket server, receives messages, bridges them
 * to the MessageGateway, and streams responses back.
 *
 * Architecture: channel adapter -> core contracts.
 */
import path from "node:path";
import { formatUnknownError } from "../core/unknownRecord.js";
import type {
  ConversationLogger,
  InboundAttachment,
  InboundMessage,
  MessageGateway
} from "../core/index.js";
import type { AgentStreamPublisher } from "../agent/index.js";
import { SessionLock } from "./sessionLock.js";
import { saveWechatAttachments, type WechatDownloadedAttachment } from "./wechatAttachmentStore.js";
import {
  WecomAibotSdkClient,
  createWechatStreamId,
  type WechatChatMessage,
  type WechatEventMessage,
  type WechatFileMessage,
  type WechatFrame,
  type WechatImageMessage,
  type WechatMediaContent,
  type WechatMixedMessage,
  type WechatSdkClient,
  type WechatTextMessage
} from "./wecomSdkClient.js";

export type WechatAdapterConfig = {
  readonly botId: string;
  readonly secret: string;
  readonly messageHandler: MessageGateway;
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
  readonly uploadRootPath?: string;
  readonly sdkClient?: WechatSdkClient;
  /**
   * Reject inbound messages while the WSS connection is unavailable.
   * Default: false, so already-received frames can still be answered through
   * the HTTP fallback channel when streaming is unavailable.
   */
  readonly rejectWhenConnectionUnavailable?: boolean;
  /**
   * Max inflight messages per user session before rejecting with "please wait".
   * Default: 3 (matches WeChat platform limit).
   */
  readonly maxInflightPerSession?: number;
};

export type WechatSessionMetrics = {
  readonly connected: boolean;
  readonly activeLockCount: number;
  /**
   * Messages waiting for or holding a session lock.
   */
  readonly inflightMessageCount: number;
  readonly trackedSessionCount: number;
  readonly streamFailureCount: number;
  readonly connectionUnavailableRejectionCount: number;
};

type ChatFrame = WechatFrame<WechatChatMessage>;
type DownloadableMediaContent = WechatMediaContent;

const DEFAULT_WECHAT_UPLOAD_ROOT_PATH = path.resolve(".harness", "wechat-uploads");
const WECHAT_CONNECTION_UNAVAILABLE_REPLY =
  "企业微信长连接正在重连，当前消息未进入处理队列，请稍后重试。";

export class WechatSmartBotAdapter {
  private readonly wsClient: WechatSdkClient;
  private readonly sessionLock = new SessionLock();
  private streamFailureCount = 0;
  private connectionUnavailableRejectionCount = 0;
  private readonly streamFailureCountsByPhase = new Map<string, number>();

  public constructor(private readonly config: WechatAdapterConfig) {
    this.wsClient =
      config.sdkClient ??
      new WecomAibotSdkClient({
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

  public getSessionMetrics(): WechatSessionMetrics {
    return {
      connected: this.isConnected,
      activeLockCount: this.sessionLock.activeLockCount(),
      inflightMessageCount: this.sessionLock.totalQueuedOrHeldCount(),
      trackedSessionCount: this.sessionLock.trackedKeyCount(),
      streamFailureCount: this.streamFailureCount,
      connectionUnavailableRejectionCount: this.connectionUnavailableRejectionCount
    };
  }

  /**
   * Send a message to a user or group chat via the smart bot SDK.
   * Used by cron delivery to proactively push results.
   */
  public async sendProactiveMessage(targetId: string, content: string): Promise<void> {
    await this.wsClient.sendMessage(targetId, {
      msgtype: "markdown",
      markdown: { content: formatProactiveWechatNotificationContent(content) }
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
    this.wsClient.on("message.text", (frame: WechatFrame<WechatTextMessage>) => {
      void this.handleTextMessage(frame);
    });

    this.wsClient.on("message.image", (frame: WechatFrame<WechatImageMessage>) => {
      void this.handleImageMessage(frame);
    });

    this.wsClient.on("message.file", (frame: WechatFrame<WechatFileMessage>) => {
      void this.handleFileMessage(frame);
    });

    this.wsClient.on("message.mixed", (frame: WechatFrame<WechatMixedMessage>) => {
      void this.handleMixedMessage(frame);
    });

    // Handle enter chat event - send welcome message
    this.wsClient.on("event.enter_chat", (frame: WechatFrame<WechatEventMessage>) => {
      this.handleEnterChat(frame);
    });
  }

  private async handleTextMessage(frame: WechatFrame<WechatTextMessage>): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const text = stripBotMention(body.text.content);
    if (await this.rejectWhenConnectionUnavailable(frame, text)) {
      return;
    }

    await this.handleChatMessage(frame, text, []);
  }

  private async handleImageMessage(frame: WechatFrame<WechatImageMessage>): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const text = "请描述这张图片。";
    if (await this.rejectWhenConnectionUnavailable(frame, text)) {
      return;
    }

    await this.handleChatMessage(frame, text, [
      await this.downloadWechatAttachment(body.image, "image")
    ]);
  }

  private async handleFileMessage(frame: WechatFrame<WechatFileMessage>): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const text = "请根据上传的文档回答。";
    if (await this.rejectWhenConnectionUnavailable(frame, text)) {
      return;
    }

    await this.handleChatMessage(frame, text, [
      await this.downloadWechatAttachment(body.file, "file")
    ]);
  }

  private async handleMixedMessage(frame: WechatFrame<WechatMixedMessage>): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const textParts: string[] = [];
    const imageContents: WechatMediaContent[] = [];
    for (const item of body.mixed.msg_item) {
      if (item.msgtype === "text") {
        textParts.push(item.text.content);
      }
      if (item.msgtype === "image") {
        imageContents.push(item.image);
      }
    }

    const text = stripBotMention(textParts.join("\n").trim());
    const normalizedText = text.length > 0 ? text : "请描述这张图片。";
    if (await this.rejectWhenConnectionUnavailable(frame, normalizedText)) {
      return;
    }

    const downloads: WechatDownloadedAttachment[] = [];
    for (const image of imageContents) {
      downloads.push(await this.downloadWechatAttachment(image, "image"));
    }

    await this.handleChatMessage(frame, text.length > 0 ? text : "请描述这张图片。", downloads);
  }

  private async rejectWhenConnectionUnavailable(frame: ChatFrame, text: string): Promise<boolean> {
    const body = frame.body;
    const rejectWhenUnavailable = this.config.rejectWhenConnectionUnavailable ?? false;
    if (body === undefined || this.isConnected || !rejectWhenUnavailable) {
      return false;
    }

    const inbound = buildWechatInboundMessage(body, text, []);
    const replyTarget =
      body.chattype === "group" && body.chatid !== undefined ? body.chatid : body.from.userid;
    this.connectionUnavailableRejectionCount += 1;
    const eventData = {
      reason: "wss disconnected",
      connected: false,
      connectionUnavailableRejectionCount: this.connectionUnavailableRejectionCount
    };

    await this.logMessageEvent("wechat.message_rejected", inbound, eventData);
    await this.logMessageEvent(
      "wechat.connection_unavailable_message_rejected",
      inbound,
      eventData
    );
    await this.sendReply(
      frame,
      createWechatStreamId(),
      replyTarget,
      inbound,
      WECHAT_CONNECTION_UNAVAILABLE_REPLY,
      "rejection"
    );
    return true;
  }

  private async handleChatMessage(
    frame: ChatFrame,
    text: string,
    downloads: readonly WechatDownloadedAttachment[]
  ): Promise<void> {
    const body = frame.body;
    if (body === undefined) return;

    const userId = body.from.userid;

    const chatId = body.chattype === "group" ? body.chatid : undefined;
    const replyTarget = chatId ?? userId;
    const streamId = createWechatStreamId();
    const streamState: { inbound: InboundMessage | undefined } = { inbound: undefined };
    const channelInfoReply = buildWechatDeliveryChannelInfo(body);
    if (channelInfoReply !== undefined && downloads.length === 0) {
      const inbound = buildWechatInboundMessage(body, text, []);
      await this.logMessageEvent("wechat.channel_info_requested", inbound, {});
      await this.sendReply(frame, streamId, replyTarget, inbound, channelInfoReply, "final");
      await this.logMessageEvent("wechat.channel_info_sent", inbound, { replyTarget });
      return;
    }

    let savedAttachments: readonly InboundAttachment[];
    try {
      savedAttachments =
        downloads.length > 0
          ? await saveWechatAttachments({
              uploadRootPath: this.config.uploadRootPath ?? DEFAULT_WECHAT_UPLOAD_ROOT_PATH,
              messageId: body.msgid,
              attachments: downloads
            })
          : [];
    } catch (error) {
      const errorMessage = `附件下载或校验失败：${formatUnknownError(error)}`;
      const inbound = buildWechatInboundMessage(body, text, []);
      await this.logMessageEvent("wechat.message_failed", inbound, {
        phase: "attachment",
        error: formatUnknownError(error)
      });
      await this.sendReply(frame, streamId, replyTarget, inbound, errorMessage, "error");
      return;
    }

    // Accumulate text deltas for intermediate stream updates.
    let accumulated = "";
    const streamCallback: AgentStreamPublisher = (event) => {
      if (event.type === "text_delta" && streamState.inbound !== undefined) {
        accumulated += event.text;
        void this.replyStreamBestEffort(
          frame,
          streamId,
          sanitizeOutgoingWechatContent(accumulated),
          false,
          streamState.inbound,
          "delta"
        );
      }
    };

    const inbound = buildWechatInboundMessage(body, text, savedAttachments, streamCallback);
    streamState.inbound = inbound;

    await this.logMessageEvent("wechat.message_processing", inbound, {
      attachmentCount: inbound.attachments?.length ?? 0
    });

    // Send "thinking" feedback without letting transport backpressure drop the message.
    await this.replyStreamBestEffort(frame, streamId, "思考中...", false, inbound, "initial");

    // Serialize per-user messages: if a previous message from the same
    // user/session is still being processed, wait for it to complete first.
    // This prevents Claude SDK session collision and history interleaving.
    const lockKey = `wechat-work:${userId}${chatId !== undefined ? `:${chatId}` : ""}`;

    // Reject immediately if too many inflight messages for this session
    const maxInflight = this.config.maxInflightPerSession ?? 3;
    if (this.sessionLock.queuedOrHeldFor(lockKey) >= maxInflight) {
      await this.sendReply(
        frame,
        streamId,
        replyTarget,
        inbound,
        "请稍等，您还有消息正在处理中，请等我回复后再发送新消息。",
        "rejection"
      );
      await this.logMessageEvent("wechat.message_rejected", inbound, {
        reason: "too many inflight messages"
      });
      return;
    }

    const release = await this.sessionLock.acquire(lockKey);

    try {
      const response = await this.config.messageHandler.handle(inbound);
      // Send the final response with finish=true, replacing any intermediate content
      const finalContent = sanitizeOutgoingWechatContent(
        accumulated.length > 0 ? accumulated : response.text
      );
      await this.sendReply(frame, streamId, replyTarget, inbound, finalContent, "final");
      await this.logMessageEvent("wechat.message_completed", inbound, {
        attachmentCount: inbound.attachments?.length ?? 0
      });
    } catch (error) {
      await this.logMessageEvent("wechat.message_failed", inbound, {
        phase: "gateway",
        error: formatUnknownError(error)
      });
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

  private handleEnterChat(frame: WechatFrame<WechatEventMessage>): void {
    void this.wsClient.replyWelcome(frame, {
      msgtype: "text",
      text: { content: "您好！我是知识库助手，可以帮您查询知识库内容。请问有什么可以帮您的？" }
    });
  }

  private async logEvent(type: string, data: Record<string, unknown>): Promise<void> {
    await this.config.eventLogger?.write({
      type,
      ...data
    });
  }

  private async sendReply(
    frame: ChatFrame,
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
    frame: ChatFrame,
    streamId: string,
    content: string,
    finish: boolean,
    message: InboundMessage,
    phase: "initial" | "delta" | "rejection" | "final" | "error"
  ): Promise<boolean> {
    try {
      await this.wsClient.replyStream(
        frame,
        streamId,
        sanitizeOutgoingWechatContent(content),
        finish
      );
      return true;
    } catch (error) {
      const failureCounts = this.recordStreamFailure(phase);
      const failureEvent = {
        phase,
        error: formatUnknownError(error),
        streamFailureCount: failureCounts.total,
        phaseFailureCount: failureCounts.phase
      };
      await this.logMessageEvent("wechat.reply_stream_failed", message, failureEvent);
      await this.logMessageEvent("wechat.stream.failure", message, failureEvent);
      return false;
    }
  }

  private recordStreamFailure(phase: string): { readonly total: number; readonly phase: number } {
    this.streamFailureCount += 1;
    const phaseFailureCount = (this.streamFailureCountsByPhase.get(phase) ?? 0) + 1;
    this.streamFailureCountsByPhase.set(phase, phaseFailureCount);
    return { total: this.streamFailureCount, phase: phaseFailureCount };
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
        markdown: { content: sanitizeOutgoingWechatContent(content) }
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

  private async downloadWechatAttachment(
    content: DownloadableMediaContent,
    kind: "image" | "file"
  ): Promise<WechatDownloadedAttachment> {
    const result = await this.wsClient.downloadFile(content.url, content.aeskey);
    return {
      kind,
      ...(result.filename !== undefined ? { name: result.filename } : {}),
      content: result.buffer
    };
  }
}

function buildWechatInboundMessage(
  body: WechatChatMessage,
  text: string,
  attachments: readonly InboundAttachment[],
  stream?: AgentStreamPublisher
): InboundMessage {
  const chatId = body.chattype === "group" ? body.chatid : undefined;
  return {
    id: body.msgid,
    channel: "wechat-work",
    user: { id: body.from.userid },
    text,
    ...(attachments.length > 0 ? { attachments } : {}),
    receivedAt: body.create_time !== undefined ? new Date(body.create_time * 1000) : new Date(),
    ...(stream !== undefined ? { stream } : {}),
    ...(chatId !== undefined ? { chatId } : {}),
    chatType: body.chattype
  };
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

/**
 * Strip mention markup that Enterprise WeChat smart bot replies render as
 * literal text instead of a notification.
 */
export function sanitizeOutgoingWechatContent(content: string): string {
  return content.replace(/<@[A-Za-z0-9_.-]+>\s*/g, "").trimStart();
}

/**
 * Convert human-authored notification mentions into Enterprise WeChat smart bot
 * mention markup. This is only for proactive notifications, not model replies.
 */
export function formatProactiveWechatNotificationContent(content: string): string {
  return content.replace(/(^|[^A-Za-z0-9_<@.-])@([A-Za-z0-9][A-Za-z0-9_.-]*|all)\b/g, "$1<@$2>");
}

export function buildWechatDeliveryChannelInfo(body: WechatChatMessage): string | undefined {
  if (!isDeliveryChannelInfoRequest(getChatMessageText(body))) {
    return undefined;
  }

  if (body.chattype === "group" && body.chatid !== undefined) {
    const channel = `wecom:chat:${body.chatid}`;
    return [
      "当前群聊的定时任务投递渠道：",
      "",
      `\`${channel}\``,
      "",
      "在定时任务的 `delivery.channels` 中填写：",
      "",
      "```text",
      channel,
      "```"
    ].join("\n");
  }

  const channel = `wecom:user:${body.from.userid}`;
  return [
    "当前单聊的定时任务投递渠道：",
    "",
    `\`${channel}\``,
    "",
    "如果要推送到群聊，请在目标群里发送 `/channel` 获取群聊渠道。"
  ].join("\n");
}

function getChatMessageText(body: WechatChatMessage): string {
  if (isTextChatMessage(body)) {
    return body.text.content;
  }
  if (isMixedChatMessage(body)) {
    return body.mixed.msg_item
      .flatMap((item) => (item.msgtype === "text" ? [item.text.content] : []))
      .join("\n");
  }
  return "";
}

function isTextChatMessage(body: WechatChatMessage): body is WechatTextMessage {
  return Object.prototype.hasOwnProperty.call(body, "text");
}

function isMixedChatMessage(body: WechatChatMessage): body is WechatMixedMessage {
  return Object.prototype.hasOwnProperty.call(body, "mixed");
}

function isDeliveryChannelInfoRequest(text: string): boolean {
  const normalized = stripBotMention(text).trim().toLowerCase();
  if (["/channel", "/channels", "/cron-channel", "channel", "channels"].includes(normalized)) {
    return true;
  }

  const asksForChannel = normalized.includes("channel") || normalized.includes("渠道");
  const asksForDelivery =
    normalized.includes("定时") ||
    normalized.includes("投递") ||
    normalized.includes("消息") ||
    normalized.includes("推送") ||
    normalized.includes("配置");
  const asksForCurrent =
    normalized.includes("当前") ||
    normalized.includes("本群") ||
    normalized.includes("群聊") ||
    normalized.includes("获取") ||
    normalized.includes("查看") ||
    normalized.includes("配置");
  return asksForChannel && asksForDelivery && asksForCurrent;
}
