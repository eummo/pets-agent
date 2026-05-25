import Fastify, { type FastifyInstance } from "fastify";
import type { AuthorizationService, FeedbackStore, MessageHandler, RoleConfigStore } from "../core/ports.js";
import type { DevRoleStore } from "../security/devRoleStore.js";
import { registerDevRoutes, type DevRoutesOptions } from "./devRoutes.js";
import type { DevProgressBroker } from "./progressBroker.js";
import { registerWechatRoutes, type WechatRoutesOptions } from "./wechatRoutes.js";

export type CreateServerOptions = {
  readonly messageHandler: MessageHandler;
  readonly wechatToken: string;
  readonly devRoleStore?: DevRoleStore | undefined;
  readonly roleConfigStore?: RoleConfigStore | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly authorization?: AuthorizationService | undefined;
  readonly progressBroker?: DevProgressBroker | undefined;
  readonly logger?: boolean;
  readonly enableDevRoutes?: boolean;
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

  if (options.enableDevRoutes !== false) {
    const devOptions: DevRoutesOptions = {
      messageHandler: options.messageHandler,
      devRoleStore: options.devRoleStore,
      roleConfigStore: options.roleConfigStore,
      feedbackStore: options.feedbackStore,
      authorization: options.authorization,
      progressBroker: options.progressBroker,
    };
    registerDevRoutes(server, devOptions);
  }

  const wechatOptions: WechatRoutesOptions = {
    messageHandler: options.messageHandler,
    wechatToken: options.wechatToken,
  };
  registerWechatRoutes(server, wechatOptions);

  return server;
}
