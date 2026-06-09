import { mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { isFileNotFound } from "../persistence/fileStoreUtils.js";

export type CronLeaderLease = {
  readonly leasePath: string;
  readonly ownerId: string;
  acquire(): Promise<boolean>;
  renew(): Promise<boolean>;
  release(): Promise<void>;
};

export type FileCronLeaderLeaseOptions = {
  readonly leasePath: string;
  readonly ttlMs: number;
  readonly ownerId?: string;
  readonly nowMs?: () => number;
};

type LeaseMetadata = {
  readonly ownerId: string;
  readonly pid: number;
  readonly host: string;
  readonly acquiredAt: string;
  readonly renewedAt: string;
};

export class FileCronLeaderLease implements CronLeaderLease {
  public readonly leasePath: string;
  public readonly ownerId: string;
  private readonly ttlMs: number;
  private readonly nowMs: () => number;

  public constructor(options: FileCronLeaderLeaseOptions) {
    this.leasePath = path.resolve(options.leasePath);
    this.ttlMs = options.ttlMs;
    this.ownerId =
      options.ownerId ??
      `${os.hostname()}-${process.pid}-${Math.random().toString(36).slice(2, 8)}`;
    this.nowMs = options.nowMs ?? (() => Date.now());
  }

  public async acquire(): Promise<boolean> {
    if (await this.tryCreateLease()) {
      return true;
    }

    if (!(await this.isExistingLeaseStale())) {
      return false;
    }

    await this.removeLeaseDirectory();
    return this.tryCreateLease();
  }

  public async renew(): Promise<boolean> {
    const metadata = await this.readMetadata();
    if (metadata?.ownerId !== this.ownerId) {
      return false;
    }

    await this.writeMetadata(metadata.acquiredAt);
    return true;
  }

  public async release(): Promise<void> {
    const metadata = await this.readMetadata();
    if (metadata?.ownerId !== this.ownerId) {
      return;
    }
    await this.removeLeaseDirectory();
  }

  private async tryCreateLease(): Promise<boolean> {
    try {
      await mkdir(this.leasePath, { recursive: false });
      await this.writeMetadata(new Date(this.nowMs()).toISOString());
      return true;
    } catch (error) {
      if (isAlreadyExists(error)) {
        return false;
      }
      throw error;
    }
  }

  private async isExistingLeaseStale(): Promise<boolean> {
    const metadata = await this.readMetadata();
    const renewedAtMs = metadata !== undefined ? Date.parse(metadata.renewedAt) : undefined;

    if (renewedAtMs !== undefined && !Number.isNaN(renewedAtMs)) {
      return this.nowMs() - renewedAtMs > this.ttlMs;
    }

    try {
      const stats = await stat(this.leasePath);
      return this.nowMs() - stats.mtimeMs > this.ttlMs;
    } catch (error) {
      if (isFileNotFound(error)) {
        return true;
      }
      throw error;
    }
  }

  private async readMetadata(): Promise<LeaseMetadata | undefined> {
    try {
      const raw = await readFile(this.metadataPath(), "utf8");
      const parsed: unknown = JSON.parse(raw);
      if (!isLeaseMetadata(parsed)) {
        return undefined;
      }
      return parsed;
    } catch (error) {
      if (isFileNotFound(error) || error instanceof SyntaxError) {
        return undefined;
      }
      throw error;
    }
  }

  private async writeMetadata(acquiredAt: string): Promise<void> {
    const now = new Date(this.nowMs()).toISOString();
    const metadata: LeaseMetadata = {
      ownerId: this.ownerId,
      pid: process.pid,
      host: os.hostname(),
      acquiredAt,
      renewedAt: now
    };
    await writeFile(this.metadataPath(), `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
  }

  private async removeLeaseDirectory(): Promise<void> {
    await this.assertLeaseDirectoryRemovable();
    await rm(this.leasePath, { recursive: true, force: true });
  }

  private metadataPath(): string {
    return path.join(this.leasePath, "owner.json");
  }

  private async assertLeaseDirectoryRemovable(): Promise<void> {
    let entries: string[];
    try {
      entries = await readdir(this.leasePath);
    } catch (error) {
      if (isFileNotFound(error)) {
        return;
      }
      throw error;
    }

    const unexpectedEntries = entries.filter((entry) => entry !== "owner.json");
    if (unexpectedEntries.length > 0) {
      throw new Error(
        `Refusing to remove cron leader lease directory with unexpected entries: ${this.leasePath}`
      );
    }
  }
}

function isLeaseMetadata(value: unknown): value is LeaseMetadata {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate["ownerId"] === "string" &&
    typeof candidate["pid"] === "number" &&
    typeof candidate["host"] === "string" &&
    typeof candidate["acquiredAt"] === "string" &&
    typeof candidate["renewedAt"] === "string"
  );
}

function isAlreadyExists(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "EEXIST";
}
