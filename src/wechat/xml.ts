import { XMLParser } from "fast-xml-parser";

const parser = new XMLParser({
  ignoreAttributes: false,
  parseTagValue: false,
  trimValues: true
});

export type WechatTextMessage = {
  readonly toUserName: string;
  readonly fromUserName: string;
  readonly createTime: string;
  readonly content: string;
  readonly msgId: string;
};

export type WechatUnsupportedMessage = {
  readonly toUserName: string;
  readonly fromUserName: string;
  readonly createTime: string;
  readonly msgType: string;
  readonly msgId?: string;
};

export type WechatMessage = WechatTextMessage | WechatUnsupportedMessage;

type ParsedWechatXml = {
  readonly xml?: {
    readonly ToUserName?: string;
    readonly FromUserName?: string;
    readonly CreateTime?: string;
    readonly MsgType?: string;
    readonly Content?: string;
    readonly MsgId?: string;
  };
};

export function parseWechatTextMessage(xml: string): WechatTextMessage {
  const parsed = parser.parse(xml) as ParsedWechatXml;
  const message = parsed.xml;

  if (message?.MsgType !== "text") {
    throw new Error(`Unsupported WeChat message type: ${message?.MsgType ?? "unknown"}`);
  }

  if (
    message.ToUserName === undefined ||
    message.FromUserName === undefined ||
    message.CreateTime === undefined ||
    message.Content === undefined ||
    message.MsgId === undefined
  ) {
    throw new Error("Invalid WeChat text message payload.");
  }

  return {
    toUserName: message.ToUserName,
    fromUserName: message.FromUserName,
    createTime: message.CreateTime,
    content: message.Content,
    msgId: message.MsgId
  };
}

export function parseWechatMessage(xml: string): WechatMessage {
  const parsed = parser.parse(xml) as ParsedWechatXml;
  const message = parsed.xml;

  if (
    message?.ToUserName === undefined ||
    message.FromUserName === undefined ||
    message.CreateTime === undefined ||
    message.MsgType === undefined
  ) {
    throw new Error("Invalid WeChat message payload.");
  }

  if (message.MsgType === "text") {
    return parseWechatTextMessage(xml);
  }

  const unsupportedMessage: WechatUnsupportedMessage = {
    toUserName: message.ToUserName,
    fromUserName: message.FromUserName,
    createTime: message.CreateTime,
    msgType: message.MsgType
  };

  if (message.MsgId !== undefined) {
    return { ...unsupportedMessage, msgId: message.MsgId };
  }

  return unsupportedMessage;
}

export function buildWechatTextReply(toUserName: string, fromUserName: string, content: string): string {
  return [
    "<xml>",
    `<ToUserName><![CDATA[${escapeCdata(toUserName)}]]></ToUserName>`,
    `<FromUserName><![CDATA[${escapeCdata(fromUserName)}]]></FromUserName>`,
    `<CreateTime>${Math.floor(Date.now() / 1000)}</CreateTime>`,
    "<MsgType><![CDATA[text]]></MsgType>",
    `<Content><![CDATA[${escapeCdata(content)}]]></Content>`,
    "</xml>"
  ].join("");
}

function escapeCdata(value: string): string {
  return value.replaceAll("]]>", "]]]]><![CDATA[>");
}
