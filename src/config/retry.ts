const DEFAULT_RETRIES = 1;
const DEFAULT_DELAY_MS = 500;

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: { readonly retries?: number; readonly delayMs?: number } = {},
): Promise<T> {
  const retries = options.retries ?? DEFAULT_RETRIES;
  const delayMs = options.delayMs ?? DEFAULT_DELAY_MS;

  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }
  throw lastError;
}
