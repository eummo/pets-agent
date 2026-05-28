/**
 * Per-key mutex that serializes async operations sharing the same key.
 *
 * Extends the core AsyncMutex with inflight count tracking for monitoring
 * concurrent operations per session key.
 */
import { AsyncMutex } from "../core/asyncMutex.js";

export class SessionLock {
  private readonly mutex = new AsyncMutex();
  private readonly inflightCounts = new Map<string, number>();

  public async acquire(key: string): Promise<() => void> {
    this.inflightCounts.set(key, (this.inflightCounts.get(key) ?? 0) + 1);

    const release = await this.mutex.acquire(key);
    return () => {
      release();
      const count = (this.inflightCounts.get(key) ?? 1) - 1;
      if (count <= 0) {
        this.inflightCounts.delete(key);
      } else {
        this.inflightCounts.set(key, count);
      }
    };
  }

  public inflightFor(key: string): number {
    return this.inflightCounts.get(key) ?? 0;
  }

  public activeLockCount(): number {
    return this.mutex.activeLockCount();
  }
}
