import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { isLocalRequest, normalizeOptionalText } from "../server/serverUtils.js";
import type { AuthorizationService } from "../auth/index.js";
import type { CronJob, CronJobResult, CronJobStore } from "./cronTypes.js";
import type { CronScheduler } from "./cronTypes.js";
import type { CronParseService } from "./cronParseService.js";
import { cronScheduleSchema, deliveryTargetSchema } from "./cronTypes.js";

export type CronRoutesOptions = {
  readonly jobStore: CronJobStore;
  readonly scheduler: CronScheduler;
  readonly authorization?: AuthorizationService;
  readonly cronParseService?: CronParseService;
};

const createJobBodySchema = z.object({
  name: z.string().min(1),
  schedule: cronScheduleSchema,
  prompt: z.string().min(1),
  workspacePath: z.string().min(1),
  role: z.string().min(1).optional(),
  enabled: z.boolean().default(true),
  delivery: deliveryTargetSchema,
  timeoutMs: z.number().int().positive().optional(),
  silentOnEmpty: z.boolean().optional()
});

const updateJobBodySchema = z.object({
  name: z.string().min(1).optional(),
  schedule: cronScheduleSchema.optional(),
  prompt: z.string().min(1).optional(),
  workspacePath: z.string().min(1).optional(),
  role: z.string().min(1).optional(),
  enabled: z.boolean().optional(),
  delivery: deliveryTargetSchema.optional(),
  timeoutMs: z.number().int().positive().optional(),
  silentOnEmpty: z.boolean().optional()
});

const parseBodySchema = z.object({
  description: z.string().min(1),
  userId: z.string().optional()
});

type CronQuery = {
  readonly userId?: string;
};

type CronJobWithRunState = CronJob & {
  readonly nextRunAt?: string;
  readonly lastStatus?: CronJobResult["status"];
  readonly lastError?: string;
  readonly lastResult?: CronJobResult;
};

export function registerCronRoutes(server: FastifyInstance, options: CronRoutesOptions): void {
  const { jobStore, scheduler } = options;

  // List all jobs
  server.get<{ Querystring: CronQuery }>("/cron/jobs", async (request, reply) => {
    const authResult = await requireCronManage(options, request.ip, request.query.userId);
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    const jobs = await jobStore.getAll();
    const results = await Promise.all(
      jobs.map(async (job) => serializeCronJobWithRunState(jobStore, job))
    );
    return reply.send(results);
  });

  // Get a single job
  server.get<{ Params: { id: string }; Querystring: CronQuery }>(
    "/cron/jobs/:id",
    async (request, reply) => {
      const authResult = await requireCronManage(options, request.ip, request.query.userId);
      if (!authResult.authorized) {
        return reply.code(authResult.statusCode).send({ error: authResult.error });
      }
      const { id } = request.params;
      const job = await jobStore.getById(id);
      if (job === undefined) {
        return reply.code(404).send({ error: "Job not found" });
      }
      return reply.send(await serializeCronJobWithRunState(jobStore, job));
    }
  );

  // Create a new job
  server.post("/cron/jobs", async (request, reply) => {
    const body = request.body as Record<string, unknown>;
    const authResult = await requireCronManage(
      options,
      request.ip,
      body["userId"] as string | undefined
    );
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    const parsed = createJobBodySchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({
        error: "Invalid request body",
        details: parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`)
      });
    }
    const job = await jobStore.create(parsed.data);
    return reply.code(201).send(job);
  });

  // Update a job
  server.patch<{ Params: { id: string } }>("/cron/jobs/:id", async (request, reply) => {
    const body = request.body as Record<string, unknown>;
    const authResult = await requireCronManage(
      options,
      request.ip,
      body["userId"] as string | undefined
    );
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    const { id } = request.params;
    const parsed = updateJobBodySchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({
        error: "Invalid request body",
        details: parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`)
      });
    }
    const updated = await jobStore.update(
      id,
      stripUndefined(parsed.data) as Partial<
        Omit<import("./cronTypes.js").CronJob, "id" | "createdAt">
      >
    );
    if (updated === undefined) {
      return reply.code(404).send({ error: "Job not found" });
    }
    return reply.send(updated);
  });

  // Delete a job
  server.delete<{ Params: { id: string }; Body?: { userId?: string } }>(
    "/cron/jobs/:id",
    async (request, reply) => {
      const body = request.body as Record<string, unknown> | undefined;
      const authResult = await requireCronManage(
        options,
        request.ip,
        body?.["userId"] as string | undefined
      );
      if (!authResult.authorized) {
        return reply.code(authResult.statusCode).send({ error: authResult.error });
      }
      const { id } = request.params;
      const deleted = await jobStore.delete(id);
      if (!deleted) {
        return reply.code(404).send({ error: "Job not found" });
      }
      return reply.code(204).send();
    }
  );

  // Trigger a job immediately
  server.post<{ Params: { id: string } }>("/cron/jobs/:id/trigger", async (request, reply) => {
    const body = request.body as Record<string, unknown> | undefined;
    const authResult = await requireCronManage(
      options,
      request.ip,
      body?.["userId"] as string | undefined
    );
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    const { id } = request.params;
    try {
      const result = await scheduler.triggerNow(id);
      return await reply.send(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return reply.code(404).send({ error: message });
    }
  });

  // Scheduler status
  server.get<{ Querystring: CronQuery }>("/cron/status", async (request, reply) => {
    const authResult = await requireCronManage(options, request.ip, request.query.userId);
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    const jobs = await jobStore.getAll();
    return reply.send({
      running: scheduler.isRunning,
      ...(scheduler.isLeader !== undefined ? { leader: scheduler.isLeader } : {}),
      totalJobs: jobs.length,
      enabledJobs: jobs.filter((j) => j.enabled).length
    });
  });

  // Parse natural language into cron job config
  server.post("/cron/parse", async (request, reply) => {
    const body = request.body as Record<string, unknown>;
    const authResult = await requireCronManage(
      options,
      request.ip,
      body["userId"] as string | undefined
    );
    if (!authResult.authorized) {
      return reply.code(authResult.statusCode).send({ error: authResult.error });
    }
    if (options.cronParseService === undefined) {
      return reply.code(501).send({ error: "Cron parse service is not configured." });
    }
    const parsed = parseBodySchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({
        error: "Invalid request body",
        details: parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`)
      });
    }
    try {
      const result = await options.cronParseService.parse(parsed.data.description);
      return await reply.send(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return reply.code(422).send({ error: "Failed to parse description", details: message });
    }
  });
}

async function serializeCronJobWithRunState(
  jobStore: CronJobStore,
  job: CronJob
): Promise<CronJobWithRunState> {
  const nextRunAt = await jobStore.getNextRunAt(job.id);
  const lastResult = await jobStore.getLastResult(job.id);
  return {
    ...job,
    ...(nextRunAt !== undefined ? { nextRunAt } : {}),
    ...(lastResult !== undefined
      ? {
          lastStatus: lastResult.status,
          ...(lastResult.error !== undefined ? { lastError: lastResult.error } : {}),
          lastResult
        }
      : {})
  };
}

// ── Auth Helper ──────────────────────────────────────────────────────────────

type AuthCheckResult =
  | { readonly authorized: true }
  | { readonly authorized: false; readonly statusCode: number; readonly error: string };

async function requireCronManage(
  options: CronRoutesOptions,
  requestIp: string,
  userId?: string
): Promise<AuthCheckResult> {
  if (!isLocalRequest(requestIp)) {
    return {
      authorized: false,
      statusCode: 403,
      error: "Cron management is only available from localhost."
    };
  }

  if (options.authorization === undefined) {
    return { authorized: false, statusCode: 501, error: "Cron management is not configured." };
  }

  const resolvedUserId = normalizeOptionalText(userId) ?? "browser-user";
  const hasPermission = await options.authorization.hasCapability(
    { id: resolvedUserId },
    "cron_manage"
  );

  if (!hasPermission) {
    return {
      authorized: false,
      statusCode: 403,
      error: "Insufficient permissions for cron management."
    };
  }

  return { authorized: true };
}

function stripUndefined<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== undefined) {
      result[key] = value;
    }
  }
  return result as Partial<T>;
}
