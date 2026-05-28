import type { FastifyInstance } from "fastify";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { isLocalRequest, normalizeOptionalText } from "./serverUtils.js";
import { setupSseResponse } from "./sseUtils.js";

type DevEventsQuery = {
  readonly userId?: string;
};

export function registerDevEventRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  server.get<{ Querystring: DevEventsQuery }>("/dev/events", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send({ error: "Development events are only available from localhost." });
    }

    const userId = normalizeOptionalText(request.query.userId) ?? "browser-user";
    setupSseResponse(reply);

    const unsubscribe = options.progressBroker?.subscribe(userId, reply.raw);
    request.raw.on("close", () => {
      unsubscribe?.();
    });

    return reply;
  });
}
