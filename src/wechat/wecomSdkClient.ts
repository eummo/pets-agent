import { WSClient, generateReqId } from "@wecom/aibot-node-sdk";
import type {
  EventMessage,
  FileMessage,
  ImageMessage,
  MixedMessage,
  TextMessage,
  WsFrame,
  WSClientOptions
} from "@wecom/aibot-node-sdk";

/**
 * Thin boundary around `@wecom/aibot-node-sdk`.
 *
 * The npm package matches the Enterprise WeChat smart-bot WebSocket capability, but its official
 * ownership is not fully proven by public WeCom developer docs. Keep all package-specific imports,
 * casts, and event names in this file so the channel adapter and core orchestration depend only on
 * project-owned WechatSdkClient contracts. Replacing the SDK should not require changes outside this
 * wrapper and the composition root wiring.
 */
export type WechatSdkClientConfig = {
  readonly botId: string;
  readonly secret: string;
  readonly wsUrl?: string;
  readonly reconnectInterval?: number;
  readonly maxReconnectAttempts?: number;
};

export type WechatFrameHeaders = {
  readonly headers: {
    readonly req_id: string;
    readonly [key: string]: unknown;
  };
};

export type WechatFrame<T> = WechatFrameHeaders & {
  readonly cmd?: string;
  readonly body?: T;
  readonly errcode?: number;
  readonly errmsg?: string;
};

export type WechatSender = {
  readonly userid: string;
};

export type WechatBaseMessage = {
  readonly msgid: string;
  readonly aibotid: string;
  readonly chatid?: string;
  readonly chattype: "single" | "group";
  readonly from: WechatSender;
  readonly create_time?: number;
  readonly response_url?: string;
  readonly msgtype: string;
};

export type WechatTextContent = {
  readonly content: string;
};

export type WechatMediaContent = {
  readonly url: string;
  readonly aeskey?: string;
};

export type WechatMixedItem =
  | {
      readonly msgtype: "text";
      readonly text: WechatTextContent;
    }
  | {
      readonly msgtype: "image";
      readonly image: WechatMediaContent;
    };

export type WechatTextMessage = WechatBaseMessage & {
  readonly msgtype: "text";
  readonly text: WechatTextContent;
};

export type WechatImageMessage = WechatBaseMessage & {
  readonly msgtype: "image";
  readonly image: WechatMediaContent;
};

export type WechatFileMessage = WechatBaseMessage & {
  readonly msgtype: "file";
  readonly file: WechatMediaContent;
};

export type WechatMixedMessage = WechatBaseMessage & {
  readonly msgtype: "mixed";
  readonly mixed: {
    readonly msg_item: readonly WechatMixedItem[];
  };
};

export type WechatEventMessage = WechatBaseMessage & {
  readonly msgtype: string;
};

export type WechatChatMessage =
  | WechatTextMessage
  | WechatImageMessage
  | WechatFileMessage
  | WechatMixedMessage;

export type WechatMarkdownMessageBody = {
  readonly msgtype: "markdown";
  readonly markdown: {
    readonly content: string;
  };
};

export type WechatWelcomeTextBody = {
  readonly msgtype: "text";
  readonly text: {
    readonly content: string;
  };
};

export type WechatDownloadedFile = {
  readonly buffer: Buffer;
  readonly filename?: string;
};

export type WechatSdkEventMap = {
  readonly authenticated: () => void;
  readonly disconnected: (reason: string) => void;
  readonly reconnecting: (attempt: number) => void;
  readonly error: (error: Error) => void;
  readonly "message.text": (frame: WechatFrame<WechatTextMessage>) => void;
  readonly "message.image": (frame: WechatFrame<WechatImageMessage>) => void;
  readonly "message.file": (frame: WechatFrame<WechatFileMessage>) => void;
  readonly "message.mixed": (frame: WechatFrame<WechatMixedMessage>) => void;
  readonly "event.enter_chat": (frame: WechatFrame<WechatEventMessage>) => void;
};

export type WechatSdkClient = {
  readonly isConnected: boolean;
  connect(): void;
  disconnect(): void;
  on<EventName extends keyof WechatSdkEventMap>(
    eventName: EventName,
    handler: WechatSdkEventMap[EventName]
  ): void;
  replyStream(
    frame: WechatFrameHeaders,
    streamId: string,
    content: string,
    finish: boolean
  ): Promise<WechatFrame<unknown>>;
  replyWelcome(
    frame: WechatFrameHeaders,
    body: WechatWelcomeTextBody
  ): Promise<WechatFrame<unknown>>;
  sendMessage(targetId: string, body: WechatMarkdownMessageBody): Promise<WechatFrame<unknown>>;
  downloadFile(url: string, aesKey?: string): Promise<WechatDownloadedFile>;
};

export class WecomAibotSdkClient implements WechatSdkClient {
  private readonly client: WSClient;

  public constructor(config: WechatSdkClientConfig) {
    this.client = new WSClient(buildSdkOptions(config));
  }

  public get isConnected(): boolean {
    return this.client.isConnected;
  }

  public connect(): void {
    this.client.connect();
  }

  public disconnect(): void {
    this.client.disconnect();
  }

  public on<EventName extends keyof WechatSdkEventMap>(
    eventName: EventName,
    handler: WechatSdkEventMap[EventName]
  ): void {
    switch (eventName) {
      case "authenticated":
        this.client.on("authenticated", handler as () => void);
        return;
      case "disconnected":
        this.client.on("disconnected", handler as (reason: string) => void);
        return;
      case "reconnecting":
        this.client.on("reconnecting", handler as (attempt: number) => void);
        return;
      case "error":
        this.client.on("error", handler as (error: Error) => void);
        return;
      case "message.text":
        this.client.on("message.text", handler as (data: WsFrame<TextMessage>) => void);
        return;
      case "message.image":
        this.client.on("message.image", handler as (data: WsFrame<ImageMessage>) => void);
        return;
      case "message.file":
        this.client.on("message.file", handler as (data: WsFrame<FileMessage>) => void);
        return;
      case "message.mixed":
        this.client.on("message.mixed", handler as (data: WsFrame<MixedMessage>) => void);
        return;
      case "event.enter_chat":
        this.client.on("event.enter_chat", handler as (data: WsFrame<EventMessage>) => void);
        return;
      default:
        assertNever(eventName);
    }
  }

  public async replyStream(
    frame: WechatFrameHeaders,
    streamId: string,
    content: string,
    finish: boolean
  ): Promise<WechatFrame<unknown>> {
    return this.client.replyStream(frame, streamId, content, finish);
  }

  public async replyWelcome(
    frame: WechatFrameHeaders,
    body: WechatWelcomeTextBody
  ): Promise<WechatFrame<unknown>> {
    return this.client.replyWelcome(frame, body);
  }

  public async sendMessage(
    targetId: string,
    body: WechatMarkdownMessageBody
  ): Promise<WechatFrame<unknown>> {
    return this.client.sendMessage(targetId, body);
  }

  public async downloadFile(url: string, aesKey?: string): Promise<WechatDownloadedFile> {
    return this.client.downloadFile(url, aesKey);
  }
}

export function createWechatStreamId(): string {
  return generateReqId("stream");
}

function buildSdkOptions(config: WechatSdkClientConfig): WSClientOptions {
  return {
    botId: config.botId,
    secret: config.secret,
    ...(config.wsUrl !== undefined ? { wsUrl: config.wsUrl } : {}),
    ...(config.reconnectInterval !== undefined
      ? { reconnectInterval: config.reconnectInterval }
      : {}),
    ...(config.maxReconnectAttempts !== undefined
      ? { maxReconnectAttempts: config.maxReconnectAttempts }
      : {})
  };
}

function assertNever(value: never): never {
  throw new Error(`Unsupported WeCom SDK event: ${String(value)}`);
}
