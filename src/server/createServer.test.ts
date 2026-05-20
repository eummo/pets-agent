import { describe, expect, it } from "vitest";
import type { MessageHandler } from "../core/ports.js";
import { createDevRoleStore } from "../security/devRoleStore.js";
import { createWechatSignature } from "../wechat/signature.js";
import { createServer } from "./createServer.js";

describe("createServer", () => {
  it("serves health checks", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({ method: "GET", url: "/health" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ ok: true, service: "pets-agent" });
  });

  it("serves the development chat page", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({ method: "GET", url: "/" });

    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toContain("text/html");
    expect(response.body).toContain("Pets Agent Dev Chat");
  });

  it("routes browser chat messages to the message handler", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "hello"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ text: "received: hello" });
  });

  it("sets development roles from the browser", async () => {
    const roleStore = createDevRoleStore();
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token",
      devRoleStore: roleStore
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/role",
      payload: {
        userId: "browser-user",
        role: "developer"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ userId: "browser-user", role: "developer" });
    expect(roleStore.getRole("browser-user")).toBe("developer");
  });

  it("verifies WeChat callback requests", async () => {
    const signature = createWechatSignature("token", "123", "nonce");
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({
      method: "GET",
      url: `/wechat/callback?signature=${signature}&timestamp=123&nonce=nonce&echostr=ok`
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toBe("ok");
  });

  it("routes WeChat text messages to the message handler", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({
      method: "POST",
      url: "/wechat/callback",
      headers: {
        "content-type": "application/xml"
      },
      payload: [
        "<xml>",
        "<ToUserName><![CDATA[agent]]></ToUserName>",
        "<FromUserName><![CDATA[user-1]]></FromUserName>",
        "<CreateTime>1700000000</CreateTime>",
        "<MsgType><![CDATA[text]]></MsgType>",
        "<Content><![CDATA[hello]]></Content>",
        "<MsgId>42</MsgId>",
        "</xml>"
      ].join("")
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toContain("<![CDATA[received: hello]]>");
  });

  it("returns a friendly reply for unsupported WeChat message types", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      wechatToken: "token"
    });

    const response = await server.inject({
      method: "POST",
      url: "/wechat/callback",
      headers: {
        "content-type": "application/xml"
      },
      payload: [
        "<xml>",
        "<ToUserName><![CDATA[agent]]></ToUserName>",
        "<FromUserName><![CDATA[user-1]]></FromUserName>",
        "<CreateTime>1700000000</CreateTime>",
        "<MsgType><![CDATA[image]]></MsgType>",
        "<MsgId>43</MsgId>",
        "</xml>"
      ].join("")
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toContain("<![CDATA[Only text messages are supported for now. Received: image.]]>");
  });
});

const echoHandler: MessageHandler = {
  handle(message) {
    return Promise.resolve({ text: `received: ${message.text}` });
  }
};
