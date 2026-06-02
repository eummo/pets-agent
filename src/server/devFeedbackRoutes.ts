import type { FastifyInstance } from "fastify";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { isLocalRequest, normalizeOptionalText } from "./serverUtils.js";

type DevFeedbackBody = {
  readonly status?: string;
  readonly userId?: string;
};

type DevFeedbackQuery = {
  readonly userId?: string;
  readonly limit?: string;
  readonly offset?: string;
  readonly status?: string;
};

export function registerDevFeedbackRoutes(
  server: FastifyInstance,
  options: DevRoutesOptions
): void {
  server.get<{ Querystring: DevFeedbackQuery }>("/dev/feedback", async (request, reply) => {
    if (options.feedbackStore === undefined || options.authorization === undefined) {
      return reply.status(501).send({ error: "Feedback management is not configured." });
    }
    if (!isLocalRequest(request.ip)) {
      return reply
        .status(403)
        .send({ error: "Feedback management is only available from localhost." });
    }

    const userId = normalizeOptionalText(request.query.userId) ?? "browser-user";
    const hasView = await options.authorization.hasCapability({ id: userId }, "feedback_view");
    if (!hasView) {
      return reply.status(403).send({ error: "Insufficient permissions to view feedback." });
    }

    const entries = await options.feedbackStore.getAll(feedbackQueryFrom(request.query));
    return { feedback: entries };
  });

  server.patch<{ Params: { id: string }; Body: DevFeedbackBody }>(
    "/dev/feedback/:id",
    async (request, reply) => {
      if (options.feedbackStore === undefined || options.authorization === undefined) {
        return reply.status(501).send({ error: "Feedback management is not configured." });
      }
      if (!isLocalRequest(request.ip)) {
        return reply
          .status(403)
          .send({ error: "Feedback management is only available from localhost." });
      }

      const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";
      const hasManage = await options.authorization.hasCapability(
        { id: userId },
        "feedback_manage"
      );
      if (!hasManage) {
        return reply.status(403).send({ error: "Insufficient permissions to manage feedback." });
      }

      const id = parsePositiveInteger(request.params.id);
      if (id === undefined) {
        return reply.status(400).send({ error: "Invalid feedback ID." });
      }

      const status = request.body.status;
      if (status !== "reviewed" && status !== "resolved") {
        return reply.status(400).send({ error: "Status must be 'reviewed' or 'resolved'." });
      }

      const updated = await options.feedbackStore.updateStatus(id, status);
      if (!updated) {
        return reply.status(404).send({ error: "Feedback entry not found." });
      }

      return { id, status };
    }
  );
}

function parsePositiveInteger(value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function parseNonNegativeInteger(value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

function parseFeedbackStatus(
  value: string | undefined
): "pending" | "reviewed" | "resolved" | undefined {
  return value === "pending" || value === "reviewed" || value === "resolved" ? value : undefined;
}

function feedbackQueryFrom(query: DevFeedbackQuery) {
  const limit = parsePositiveInteger(query.limit);
  const offset = parseNonNegativeInteger(query.offset);
  const status = parseFeedbackStatus(query.status);
  return {
    ...(limit !== undefined ? { limit } : {}),
    ...(offset !== undefined ? { offset } : {}),
    ...(status !== undefined ? { status } : {})
  };
}
