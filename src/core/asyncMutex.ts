/**
 * Per-key mutex that serializes async operations sharing the same key.
 *
 * Each acquire() chains onto the previous lock for that key, so callers
 * are processed in FIFO order. Different keys are independent and never
 * block each other.
 */
type Release = () => void;

export class AsyncMutex {
  private readonly locks = new Map<string, Promise<void>>();

  public async acquire(key: string): Promise<Release> {
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
      void queued.finally(() => {
        if (this.locks.get(key) === queued) {
          this.locks.delete(key);
        }
      });
    };
  }

  public activeLockCount(): number {
    return this.locks.size;
  }
}
