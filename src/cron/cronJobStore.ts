import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { toLocalIsoString } from "../logging/jsonlLogger.js";
import { FileMutex, isFileNotFound } from "../persistence/fileStoreUtils.js";
import {
  cronJobStoreFileSchema,
  type CronJob,
  type CronJobResult,
  type CronJobStore,
  type CronJobStoreFile,
} from "./cronTypes.js";

export class FileCronJobStore implements CronJobStore {
  private readonly filePath: string;
  private readonly mutex = new FileMutex();

  public constructor(filePathInput: string) {
    this.filePath = path.resolve(filePathInput);
  }

  public async getAll(): Promise<readonly CronJob[]> {
    const store = await this.readStore();
    return Object.values(store.jobs);
  }

  public async getById(id: string): Promise<CronJob | undefined> {
    const store = await this.readStore();
    return store.jobs[id];
  }

  public async create(
    job: Omit<CronJob, "id" | "createdAt" | "updatedAt">
  ): Promise<CronJob> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const store = await this.readStore();
      const id = generateJobId(job.name);
      if (store.jobs[id] !== undefined) {
        throw new Error(`Cron job already exists: ${id}`);
      }
      const now = toLocalIsoString(new Date());
      const newJob: CronJob = { ...job, id, createdAt: now, updatedAt: now };
      store.jobs[id] = newJob;
      store.runState[id] = {};
      await this.writeStore(store);
      return newJob;
    } finally {
      release();
    }
  }

  public async update(
    id: string,
    patch: Partial<Omit<CronJob, "id" | "createdAt">>
  ): Promise<CronJob | undefined> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const store = await this.readStore();
      const existing = store.jobs[id];
      if (existing === undefined) return undefined;
      const now = toLocalIsoString(new Date());
      const updated: CronJob = { ...existing, ...patch, id, updatedAt: now };
      store.jobs[id] = updated;
      await this.writeStore(store);
      return updated;
    } finally {
      release();
    }
  }

  public async delete(id: string): Promise<boolean> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const store = await this.readStore();
      if (store.jobs[id] === undefined) return false;
      const { [id]: _removed, ...remainingJobs } = store.jobs;
      void _removed;
      const { [id]: _removed2, ...remainingStates } = store.runState;
      void _removed2;
      await this.writeStore({ jobs: remainingJobs, runState: remainingStates });
      return true;
    } finally {
      release();
    }
  }

  public async getNextRunAt(id: string): Promise<string | undefined> {
    const store = await this.readStore();
    return store.runState[id]?.nextRunAt;
  }

  public async setNextRunAt(id: string, nextRunAt: string): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const store = await this.readStore();
      store.runState[id] ??= {};
      store.runState[id].nextRunAt = nextRunAt;
      await this.writeStore(store);
    } finally {
      release();
    }
  }

  public async getLastResult(id: string): Promise<CronJobResult | undefined> {
    const store = await this.readStore();
    return store.runState[id]?.lastResult;
  }

  public async setLastResult(id: string, result: CronJobResult): Promise<void> {
    const release = await this.mutex.acquire(this.filePath);
    try {
      const store = await this.readStore();
      store.runState[id] ??= {};
      store.runState[id].lastResult = result;
      await this.writeStore(store);
    } finally {
      release();
    }
  }

  private async readStore(): Promise<CronJobStoreFile> {
    try {
      const raw = await readFile(this.filePath, "utf8");
      const parsed = cronJobStoreFileSchema.parse(JSON.parse(raw));
      return {
        jobs: { ...parsed.jobs },
        runState: { ...parsed.runState },
      };
    } catch (error) {
      if (isFileNotFound(error)) {
        return { jobs: {}, runState: {} };
      }
      throw error;
    }
  }

  private async writeStore(store: CronJobStoreFile): Promise<void> {
    await mkdir(path.dirname(this.filePath), { recursive: true });
    const tempPath = `${this.filePath}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(tempPath, `${JSON.stringify(store, null, 2)}\n`, "utf8");
    await rename(tempPath, this.filePath);
  }
}

function generateJobId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const suffix = Math.random().toString(36).slice(2, 8);
  return slug.length > 0 ? `${slug}-${suffix}` : `job-${suffix}`;
}
