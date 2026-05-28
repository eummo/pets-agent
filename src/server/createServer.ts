import Fastify, { type FastifyInstance } from "fastify";
import { registerDevRoutes, type DevRoutesOptions } from "./devRoutes.js";

export type CreateServerOptions = DevRoutesOptions & {
  readonly logger?: boolean;
  readonly enableDevRoutes?: boolean;
};

export function createServer(options: CreateServerOptions): FastifyInstance {
  const enableDevRoutes = options.enableDevRoutes === true;
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

