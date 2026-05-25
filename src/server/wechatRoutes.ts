import type { FastifyInstance } from "fastify";
import type { MessageHandler } from "../core/ports.js";
import { verifyWechatSignature } from "../wechat/signature.js";
import { buildWechatTextReply, parseWechatMessage } from "../wechat/xml.js";

type WechatVerifyQuery = {
  readonly msg_signature?: string;
  readonly signature?: string;
  readonly timestamp?: string;
  readonly nonce?: string;
  readonly echostr?: string;
};

export type WechatRoutesOptions = {
  readonly messageHandler: MessageHandler;
  readonly wechatToken: string;
};

export function registerWechatRoutes(server: FastifyInstance, options: WechatRoutesOptions): void {
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
}
