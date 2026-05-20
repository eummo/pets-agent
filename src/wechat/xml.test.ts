import { describe, expect, it } from "vitest";
import {
  buildWechatTextReply,
  parseWechatMessage,
  parseWechatTextMessage
} from "./xml.js";

const sampleTextXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
  <CreateTime>1609459200</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[hello world]]></Content>
  <MsgId>1234567890</MsgId>
</xml>`.trim();

describe("parseWechatTextMessage", () => {
  it("parses a valid text message", () => {
    const message = parseWechatTextMessage(sampleTextXml);

    expect(message).toEqual({
      toUserName: "server",
      fromUserName: "user-1",
      createTime: "1609459200",
      content: "hello world",
      msgId: "1234567890"
    });
  });

  it("throws for non-text message type", () => {
    const imageXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
  <CreateTime>1609459200</CreateTime>
  <MsgType><![CDATA[image]]></MsgType>
  <MsgId>1234567890</MsgId>
</xml>`.trim();

    expect(() => parseWechatTextMessage(imageXml)).toThrow(
      "Unsupported WeChat message type: image"
    );
  });

  it("throws for unknown message type when MsgType is missing", () => {
    const noTypeXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
</xml>`.trim();

    expect(() => parseWechatTextMessage(noTypeXml)).toThrow(
      "Unsupported WeChat message type: unknown"
    );
  });

  it("throws when required fields are missing from a text message", () => {
    const incompleteXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
  <CreateTime>1609459200</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
</xml>`.trim();

    expect(() => parseWechatTextMessage(incompleteXml)).toThrow(
      "Invalid WeChat text message payload."
    );
  });
});

describe("parseWechatMessage", () => {
  it("returns a WechatTextMessage for text messages", () => {
    const message = parseWechatMessage(sampleTextXml);

    expect(message).toEqual({
      toUserName: "server",
      fromUserName: "user-1",
      createTime: "1609459200",
      content: "hello world",
      msgId: "1234567890"
    });
  });

  it("returns a WechatUnsupportedMessage for image messages", () => {
    const imageXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
  <CreateTime>1609459200</CreateTime>
  <MsgType><![CDATA[image]]></MsgType>
  <MsgId>9999</MsgId>
</xml>`.trim();

    const message = parseWechatMessage(imageXml);

    expect(message).toEqual({
      toUserName: "server",
      fromUserName: "user-1",
      createTime: "1609459200",
      msgType: "image",
      msgId: "9999"
    });
  });

  it("returns a WechatUnsupportedMessage without msgId when absent", () => {
    const eventXml = `
<xml>
  <ToUserName><![CDATA[server]]></ToUserName>
  <FromUserName><![CDATA[user-1]]></FromUserName>
  <CreateTime>1609459200</CreateTime>
  <MsgType><![CDATA[event]]></MsgType>
</xml>`.trim();

    const message = parseWechatMessage(eventXml);

    expect(message).toEqual({
      toUserName: "server",
      fromUserName: "user-1",
      createTime: "1609459200",
      msgType: "event"
    });
  });

  it("throws when required fields are missing", () => {
    const emptyXml = "<xml></xml>";

    expect(() => parseWechatMessage(emptyXml)).toThrow(
      "Invalid WeChat message payload."
    );
  });
});

describe("buildWechatTextReply", () => {
  it("builds an XML reply with CDATA-wrapped fields", () => {
    const reply = buildWechatTextReply("user-1", "server", "hello");

    expect(reply).toContain("<ToUserName><![CDATA[user-1]]></ToUserName>");
    expect(reply).toContain("<FromUserName><![CDATA[server]]></FromUserName>");
    expect(reply).toContain("<MsgType><![CDATA[text]]></MsgType>");
    expect(reply).toContain("<Content><![CDATA[hello]]></Content>");
    expect(reply).toContain("<CreateTime>");
    expect(reply).toMatch(/^<xml>.*<\/xml>$/);
  });

  it("escapes CDATA closing sequences in content", () => {
    const reply = buildWechatTextReply("user-1", "server", "hello]]>world");

    expect(reply).toContain("<Content><![CDATA[hello]]]]><![CDATA[>world]]></Content>");
  });

  it("escapes CDATA closing sequences in user names", () => {
    const reply = buildWechatTextReply("user]]>1", "server", "hello");

    expect(reply).toContain("<ToUserName><![CDATA[user]]]]><![CDATA[>1]]></ToUserName>");
  });
});
