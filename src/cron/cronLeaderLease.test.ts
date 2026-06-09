import { mkdtemp, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { FileCronLeaderLease } from "./cronLeaderLease.js";

describe("FileCronLeaderLease", () => {
  it("allows one owner to acquire the lease at a time", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-lease-"));
    const leasePath = path.join(root, "leader");
    const first = new FileCronLeaderLease({
      leasePath,
      ttlMs: 60_000,
      ownerId: "first"
    });
    const second = new FileCronLeaderLease({
      leasePath,
      ttlMs: 60_000,
      ownerId: "second"
    });

    await expect(first.acquire()).resolves.toBe(true);
    await expect(second.acquire()).resolves.toBe(false);

    await first.release();
    await expect(second.acquire()).resolves.toBe(true);
  });

  it("reclaims a stale lease", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-lease-"));
    const leasePath = path.join(root, "leader");
    let now = Date.parse("2026-06-08T00:00:00.000Z");

    const first = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "first",
      nowMs: () => now
    });
    const second = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "second",
      nowMs: () => now
    });

    await expect(first.acquire()).resolves.toBe(true);
    now += 1_001;
    await expect(second.acquire()).resolves.toBe(true);
    await expect(first.renew()).resolves.toBe(false);
  });

  it("renew keeps the current owner from expiring", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-lease-"));
    const leasePath = path.join(root, "leader");
    let now = Date.parse("2026-06-08T00:00:00.000Z");

    const first = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "first",
      nowMs: () => now
    });
    const second = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "second",
      nowMs: () => now
    });

    await expect(first.acquire()).resolves.toBe(true);
    now += 800;
    await expect(first.renew()).resolves.toBe(true);
    now += 800;
    await expect(second.acquire()).resolves.toBe(false);
  });

  it("refuses to reclaim a stale lease directory that contains unexpected files", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-lease-"));
    const leasePath = path.join(root, "leader");
    let now = Date.parse("2026-06-08T00:00:00.000Z");

    const first = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "first",
      nowMs: () => now
    });
    await expect(first.acquire()).resolves.toBe(true);
    await writeFile(path.join(leasePath, "business-state.json"), "{}", "utf8");

    const lease = new FileCronLeaderLease({
      leasePath,
      ttlMs: 1_000,
      ownerId: "second",
      nowMs: () => now
    });

    now += 1_001;
    await expect(lease.acquire()).rejects.toThrow(
      "Refusing to remove cron leader lease directory with unexpected entries"
    );
    await expect(readdir(leasePath)).resolves.toEqual(
      expect.arrayContaining(["owner.json", "business-state.json"])
    );
  });
});
