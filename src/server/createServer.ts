import Fastify, { type FastifyInstance } from "fastify";
import type { AuthorizationService, FeedbackStore, MessageHandler, RoleConfigStore } from "../core/ports.js";
import { registerDevRoutes, type DevRoutesOptions } from "./devRoutes.js";
import type { SseProgressBroker } from "./sseProgressBroker.js";

export type CreateServerOptions = {
  readonly messageHandler: MessageHandler;
  readonly roleConfigStore?: RoleConfigStore | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly authorization?: AuthorizationService | undefined;
  readonly progressBroker?: SseProgressBroker | undefined;
  readonly logger?: boolean;
  readonly enableDevRoutes?: boolean;
};

export function createServer(options: CreateServerOptions): FastifyInstance {
  const enableDevRoutes = options.enableDevRoutes !== false;
  const server = Fastify({ logger: options.logger ?? false });

  server.get("/health", () => ({
    ok: true,
    service: "pets-agent"
  }));

  if (enableDevRoutes) {
    const devOptions: DevRoutesOptions = {
      messageHandler: options.messageHandler,
      roleConfigStore: options.roleConfigStore,
      feedbackStore: options.feedbackStore,
      authorization: options.authorization,
      progressBroker: options.progressBroker,
    };
    registerDevRoutes(server, devOptions);
  }

  return server;
}
