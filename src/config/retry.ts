const DEFAULT_RETRIES = 1;
const DEFAULT_DELAY_MS = 500;
const DEFAULT_BACKOFF_MULTIPLIER = 2;
const DEFAULT_JITTER_MS = 100;

export type RetryOptions = {
  readonly retries?: number;
  readonly delayMs?: number;
  readonly backoffMultiplier?: number;
  readonly jitterMs?: number;
  readonly shouldRetry?: (error: unknown, attempt: number) => boolean;
  readonly onRetry?: (event: { readonly attempt: number; readonly delayMs: number; readonly error: unknown }) => void;
};

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const retries = options.retries ?? DEFAULT_RETRIES;
  const delayMs = options.delayMs ?? DEFAULT_DELAY_MS;
  const backoffMultiplier = options.backoffMultiplier ?? DEFAULT_BACKOFF_MULTIPLIER;
  const jitterMs = options.jitterMs ?? DEFAULT_JITTER_MS;

  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt >= retries || !(options.shouldRetry?.(error, attempt) ?? true)) {
        throw error;
      }
      const delay = retryDelay(attempt, delayMs, backoffMultiplier, jitterMs);
      options.onRetry?.({ attempt: attempt + 1, delayMs: delay, error });
      await sleep(delay);
    }
  }
  throw lastError;
}

function retryDelay(attempt: number, delayMs: number, backoffMultiplier: number, jitterMs: number): number {
  const exponentialDelay = delayMs * (backoffMultiplier ** attempt);
  const jitter = jitterMs > 0 ? Math.floor(Math.random() * jitterMs) : 0;
  return Math.max(0, Math.trunc(exponentialDelay + jitter));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
