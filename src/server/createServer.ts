import Fastify, { type FastifyInstance } from "fastify";
import { registerDevRoutes, type DevRoutesOptions } from "./devRoutes.js";

export type HealthCheckStatus = "ok" | "warn" | "fail";

export type HealthCheckResult = {
  readonly status: HealthCheckStatus;
  readonly message?: string;
};

export type HealthCheck = {
  readonly name: string;
  check(): HealthCheckResult | Promise<HealthCheckResult>;
};

type HealthzResponse = {
  readonly ok: true;
  readonly service: "pets-agent";
};

type ReadyzResponse = {
  readonly ok: boolean;
  readonly service: "pets-agent";
  readonly status: "ok" | "degraded" | "not_ready";
  readonly checks: Record<string, HealthCheckResult>;
};

export type CreateServerOptions = DevRoutesOptions & {
  readonly logger?: boolean;
  readonly enableDevRoutes?: boolean;
  readonly readinessChecks?: readonly HealthCheck[];
};

export function createServer(options: CreateServerOptions): FastifyInstance {
  const enableDevRoutes = options.enableDevRoutes === true;
  const server = Fastify({ logger: options.logger ?? false });

  server.get("/health", () => ({
    ok: true,
    service: "pets-agent"
  }));

  server.get(
    "/healthz",
    (): HealthzResponse => ({
      ok: true,
      service: "pets-agent"
    })
  );

  server.get("/readyz", async (_request, reply): Promise<ReadyzResponse> => {
    const checks = await runReadinessChecks(options.readinessChecks ?? []);
    const status = summarizeReadiness(checks);
    const ok = status !== "not_ready";
    if (!ok) {
      reply.status(503);
    }
    return {
      ok,
      service: "pets-agent",
      status,
      checks
    };
  });

  if (enableDevRoutes) {
    const devOptions: DevRoutesOptions = {
      messageHandler: options.messageHandler,
      roleConfigStore: options.roleConfigStore,
      feedbackStore: options.feedbackStore,
      authorization: options.authorization,
      progressBroker: options.progressBroker,
      uploadRootPath: options.uploadRootPath
    };
    registerDevRoutes(server, devOptions);
  }

  return server;
}

async function runReadinessChecks(
  checks: readonly HealthCheck[]
): Promise<Record<string, HealthCheckResult>> {
  const results: Record<string, HealthCheckResult> = {};
  for (const check of checks) {
    try {
      results[check.name] = await check.check();
    } catch (error) {
      results[check.name] = {
        status: "fail",
        message: error instanceof Error ? error.message : String(error)
      };
    }
  }
  return results;
}

function summarizeReadiness(checks: Record<string, HealthCheckResult>): ReadyzResponse["status"] {
  const statuses = Object.values(checks).map((check) => check.status);
  if (statuses.includes("fail")) {
    return "not_ready";
  }
  if (statuses.includes("warn")) {
    return "degraded";
  }
  return "ok";
}
