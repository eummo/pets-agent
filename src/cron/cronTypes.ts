import { z } from "zod";

// ── Schedule Types ──────────────────────────────────────────────────────────

export const cronExpressionSchema = z.object({
  type: z.literal("cron"),
  expression: z.string().min(1),
});

export const intervalScheduleSchema = z.object({
  type: z.literal("interval"),
  milliseconds: z.number().int().positive(),
});

export const onceScheduleSchema = z.object({
  type: z.literal("once"),
  runAt: z.string().min(1), // ISO 8601
});

export const cronScheduleSchema = z.discriminatedUnion("type", [
  cronExpressionSchema,
  intervalScheduleSchema,
  onceScheduleSchema,
]);

export type CronSchedule = z.infer<typeof cronScheduleSchema>;

// ── Delivery Target ─────────────────────────────────────────────────────────

export const deliveryTargetSchema = z.object({
  channels: z.array(z.string().min(1)),
  template: z.string().min(1).optional(),
});

export type DeliveryTarget = z.infer<typeof deliveryTargetSchema>;

// ── Job Definition ──────────────────────────────────────────────────────────

export const cronJobSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  schedule: cronScheduleSchema,
  prompt: z.string().min(1),
  workspacePath: z.string().min(1),
  role: z.string().min(1).optional(),
  enabled: z.boolean(),
  delivery: deliveryTargetSchema,
  timeoutMs: z.number().int().positive().optional(),
  silentOnEmpty: z.boolean().optional(),
  createdAt: z.string().min(1),
  updatedAt: z.string().min(1),
});

export type CronJob = z.infer<typeof cronJobSchema>;

// ── Job Execution Result ───────────────────────────────────────────────────

export const cronJobResultSchema = z.object({
  jobId: z.string().min(1),
  startedAt: z.string().min(1),
  finishedAt: z.string().min(1),
  status: z.enum(["success", "error", "timeout", "skipped"]),
  output: z.string(),
  error: z.string().optional(),
});

export type CronJobResult = z.infer<typeof cronJobResultSchema>;

// ── Run State ───────────────────────────────────────────────────────────────

export const cronRunStateSchema = z.object({
  nextRunAt: z.string().min(1).optional(),
  lastResult: cronJobResultSchema.optional(),
});

export type CronRunState = z.infer<typeof cronRunStateSchema>;

// ── Persistence File Schema ─────────────────────────────────────────────────

export const cronJobStoreFileSchema = z.object({
  jobs: z.record(z.string(), cronJobSchema),
  runState: z.record(z.string(), cronRunStateSchema),
});

export type CronJobStoreFile = z.infer<typeof cronJobStoreFileSchema>;

// ── Delivery Payload ────────────────────────────────────────────────────────

export type DeliveryPayload = {
  readonly jobName: string;
  readonly output: string;
  readonly error?: string;
  readonly template?: string;
};

// ── Contracts (interfaces, not Zod) ─────────────────────────────────────────

export type CronJobStore = {
  getAll(): Promise<readonly CronJob[]>;
  getById(id: string): Promise<CronJob | undefined>;
  create(job: Omit<CronJob, "id" | "createdAt" | "updatedAt">): Promise<CronJob>;
  update(
    id: string,
    patch: Partial<Omit<CronJob, "id" | "createdAt">>
  ): Promise<CronJob | undefined>;
  delete(id: string): Promise<boolean>;

  getNextRunAt(id: string): Promise<string | undefined>;
  setNextRunAt(id: string, nextRunAt: string): Promise<void>;
  getLastResult(id: string): Promise<CronJobResult | undefined>;
  setLastResult(id: string, result: CronJobResult): Promise<void>;
};

export type DeliveryChannel = {
  readonly prefix: string;
  deliver(target: string, payload: DeliveryPayload): Promise<void>;
};

export type CronScheduler = {
  start(): void;
  stop(): void;
  triggerNow(jobId: string): Promise<CronJobResult>;
  readonly isRunning: boolean;
};
