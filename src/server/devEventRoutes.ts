import type { FastifyInstance } from "fastify";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { normalizeOptionalText } from "./serverUtils.js";

type DevEventsQuery = {
  readonly userId?: string;
};

export function registerDevEventRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  server.get<{ Querystring: DevEventsQuery }>("/dev/events", async (request, reply) => {
    const userId = normalizeOptionalText(request.query.userId) ?? "browser-user";
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no"
    });
    reply.raw.write("\n");

    const unsubscribe = options.progressBroker?.subscribe(userId, reply.raw);
    request.raw.on("close", () => {
      unsubscribe?.();
    });

    return reply;
  });
}
