/**
 * Per-key mutex that serializes async operations sharing the same key.
 *
 * Uses the same Promise-chain pattern as FileMutex: each acquire() chains
 * onto the previous lock for that key, so callers are processed in FIFO order.
 * Different keys are independent and never block each other.
 */
type Release = () => void;

export class SessionLock {
  private readonly locks = new Map<string, Promise<void>>();
  private readonly inflightCounts = new Map<string, number>();

  public async acquire(key: string): Promise<Release> {
    this.inflightCounts.set(key, (this.inflightCounts.get(key) ?? 0) + 1);

    const previous = this.locks.get(key);
    let resolve!: Release;
    const next = new Promise<void>((r) => { resolve = r; });
    const queued = previous ? previous.then(() => next) : next;
    this.locks.set(key, queued);
    if (previous !== undefined) {
      await previous;
    }
    let released = false;
    return () => {
      if (released) {
        return;
      }
      released = true;
      resolve();
      const count = (this.inflightCounts.get(key) ?? 1) - 1;
      if (count <= 0) {
        this.inflightCounts.delete(key);
      } else {
        this.inflightCounts.set(key, count);
      }
      void queued.finally(() => {
        if (this.locks.get(key) === queued) {
          this.locks.delete(key);
        }
      });
    };
  }

  public inflightFor(key: string): number {
    return this.inflightCounts.get(key) ?? 0;
  }

  public activeLockCount(): number {
    return this.locks.size;
  }
}
