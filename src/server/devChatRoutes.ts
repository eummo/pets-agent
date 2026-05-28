import type { FastifyInstance } from "fastify";
import type { AgentStreamEvent } from "../agent/index.js";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { isLocalRequest, normalizeOptionalText } from "./serverUtils.js";
import { setupSseResponse, writeSse } from "./sseUtils.js";

type DevChatBody = {
  readonly userId?: string;
  readonly text?: string;
};

export function registerDevChatRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  server.post<{ Body: DevChatBody }>("/dev/chat", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply
        .status(403)
        .send({ error: "Development chat is only available from localhost." });
    }

    const text = request.body.text?.trim();
    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";

    if (text === undefined || text.length === 0) {
      return reply.status(400).send({ error: "Message text is required." });
    }

    setupSseResponse(reply);

    const streamCallback = (event: AgentStreamEvent): void => {
      writeSse(reply.raw, "agent", event);
    };

    try {
      const response = await options.messageHandler.handle({
        id: `dev-${Date.now()}`,
        channel: "dev-browser",
        user: { id: userId },
        text,
        receivedAt: new Date(),
        stream: streamCallback
      });

      writeSse(reply.raw, "agent", {
        type: "completed",
        sessionId: response.sessionId,
        text: response.text
      });
    } catch (error) {
      writeSse(reply.raw, "agent", {
        type: "error",
        message: error instanceof Error ? error.message : String(error)
      });
    } finally {
      reply.raw.end();
    }

    return reply;
  });
}
