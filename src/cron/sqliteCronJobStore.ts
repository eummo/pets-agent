import type Database from "better-sqlite3";
import { toLocalIsoString } from "../logging/jsonlLogger.js";
import {
  cronJobResultSchema,
  cronScheduleSchema,
  deliveryTargetSchema,
  type CronJob,
  type CronJobResult,
  type CronJobStore
} from "./cronTypes.js";

type CronJobRow = {
  readonly id: string;
  readonly name: string;
  readonly schedule_json: string;
  readonly prompt: string;
  readonly workspace_path: string;
  readonly role: string | null;
  readonly enabled: number;
  readonly delivery_json: string;
  readonly timeout_ms: number | null;
  readonly silent_on_empty: number | null;
  readonly created_at: string;
  readonly updated_at: string;
};

type CronRunStateRow = {
  readonly next_run_at: string | null;
  readonly last_result_json: string | null;
};

export class SqliteCronJobStore implements CronJobStore {
  public constructor(private readonly db: Database.Database) {}

  public getAll(): Promise<readonly CronJob[]> {
    const rows = this.db
      .prepare(
        `
      SELECT id, name, schedule_json, prompt, workspace_path, role, enabled, delivery_json,
             timeout_ms, silent_on_empty, created_at, updated_at
      FROM cron_jobs
      ORDER BY rowid ASC
    `
      )
      .all() as CronJobRow[];
    return Promise.resolve(rows.map(rowToCronJob));
  }

  public getById(id: string): Promise<CronJob | undefined> {
    const row = this.db
      .prepare(
        `
      SELECT id, name, schedule_json, prompt, workspace_path, role, enabled, delivery_json,
             timeout_ms, silent_on_empty, created_at, updated_at
      FROM cron_jobs
      WHERE id = ?
    `
      )
      .get(id) as CronJobRow | undefined;
    return Promise.resolve(row === undefined ? undefined : rowToCronJob(row));
  }

  public create(job: Omit<CronJob, "id" | "createdAt" | "updatedAt">): Promise<CronJob> {
    const id = generateJobId(job.name);
    const now = toLocalIsoString(new Date());
    const created: CronJob = { ...job, id, createdAt: now, updatedAt: now };
    const transaction = this.db.transaction(() => {
      this.db
        .prepare(
          `
        INSERT INTO cron_jobs (
          id, name, schedule_json, prompt, workspace_path, role, enabled, delivery_json,
          timeout_ms, silent_on_empty, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `
        )
        .run(
          created.id,
          created.name,
          JSON.stringify(created.schedule),
          created.prompt,
          created.workspacePath,
          created.role ?? null,
          created.enabled ? 1 : 0,
          JSON.stringify(created.delivery),
          created.timeoutMs ?? null,
          created.silentOnEmpty === undefined ? null : created.silentOnEmpty ? 1 : 0,
          created.createdAt,
          created.updatedAt
        );
      this.db.prepare("INSERT INTO cron_run_state (job_id) VALUES (?)").run(created.id);
    });
    transaction();
    return Promise.resolve(created);
  }

  public update(
    id: string,
    patch: Partial<Omit<CronJob, "id" | "createdAt">>
  ): Promise<CronJob | undefined> {
    const existing = this.getRowById(id);
    if (existing === undefined) {
      return Promise.resolve(undefined);
    }

    const current = rowToCronJob(existing);
    const updated: CronJob = {
      ...current,
      ...patch,
      id,
      createdAt: current.createdAt,
      updatedAt: toLocalIsoString(new Date())
    };
    this.db
      .prepare(
        `
      UPDATE cron_jobs
      SET name = ?,
          schedule_json = ?,
          prompt = ?,
          workspace_path = ?,
          role = ?,
          enabled = ?,
          delivery_json = ?,
          timeout_ms = ?,
          silent_on_empty = ?,
          updated_at = ?
      WHERE id = ?
    `
      )
      .run(
        updated.name,
        JSON.stringify(updated.schedule),
        updated.prompt,
        updated.workspacePath,
        updated.role ?? null,
        updated.enabled ? 1 : 0,
        JSON.stringify(updated.delivery),
        updated.timeoutMs ?? null,
        updated.silentOnEmpty === undefined ? null : updated.silentOnEmpty ? 1 : 0,
        updated.updatedAt,
        id
      );
    return Promise.resolve(updated);
  }

  public delete(id: string): Promise<boolean> {
    const result = this.db.prepare("DELETE FROM cron_jobs WHERE id = ?").run(id);
    return Promise.resolve(result.changes > 0);
  }

  public getNextRunAt(id: string): Promise<string | undefined> {
    const row = this.getRunStateRow(id);
    return Promise.resolve(row?.next_run_at ?? undefined);
  }

  public setNextRunAt(id: string, nextRunAt: string): Promise<void> {
    this.upsertRunState(id, { nextRunAt });
    return Promise.resolve();
  }

  public getLastResult(id: string): Promise<CronJobResult | undefined> {
    const row = this.getRunStateRow(id);
    if (row?.last_result_json === null || row?.last_result_json === undefined) {
      return Promise.resolve(undefined);
    }
    return Promise.resolve(cronJobResultSchema.parse(JSON.parse(row.last_result_json)));
  }

  public setLastResult(id: string, result: CronJobResult): Promise<void> {
    this.upsertRunState(id, { lastResult: result });
    return Promise.resolve();
  }

  private getRowById(id: string): CronJobRow | undefined {
    return this.db
      .prepare(
        `
      SELECT id, name, schedule_json, prompt, workspace_path, role, enabled, delivery_json,
             timeout_ms, silent_on_empty, created_at, updated_at
      FROM cron_jobs
      WHERE id = ?
    `
      )
      .get(id) as CronJobRow | undefined;
  }

  private getRunStateRow(id: string): CronRunStateRow | undefined {
    return this.db
      .prepare("SELECT next_run_at, last_result_json FROM cron_run_state WHERE job_id = ?")
      .get(id) as CronRunStateRow | undefined;
  }

  private upsertRunState(
    id: string,
    patch: { readonly nextRunAt?: string; readonly lastResult?: CronJobResult }
  ): void {
    const existing = this.getRunStateRow(id);
    this.db
      .prepare(
        `
      INSERT INTO cron_run_state (job_id, next_run_at, last_result_json, updated_at)
      VALUES (?, ?, ?, datetime('now', 'localtime'))
      ON CONFLICT(job_id) DO UPDATE SET
        next_run_at = excluded.next_run_at,
        last_result_json = excluded.last_result_json,
        updated_at = datetime('now', 'localtime')
    `
      )
      .run(
        id,
        patch.nextRunAt ?? existing?.next_run_at ?? null,
        patch.lastResult !== undefined
          ? JSON.stringify(patch.lastResult)
          : (existing?.last_result_json ?? null)
      );
  }
}

function rowToCronJob(row: CronJobRow): CronJob {
  return {
    id: row.id,
    name: row.name,
    schedule: cronScheduleSchema.parse(JSON.parse(row.schedule_json)),
    prompt: row.prompt,
    workspacePath: row.workspace_path,
    ...(row.role !== null ? { role: row.role } : {}),
    enabled: row.enabled === 1,
    delivery: deliveryTargetSchema.parse(JSON.parse(row.delivery_json)),
    ...(row.timeout_ms !== null ? { timeoutMs: row.timeout_ms } : {}),
    ...(row.silent_on_empty !== null ? { silentOnEmpty: row.silent_on_empty === 1 } : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function generateJobId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const suffix = Math.random().toString(36).slice(2, 8);
  return slug.length > 0 ? `${slug}-${suffix}` : `job-${suffix}`;
}
