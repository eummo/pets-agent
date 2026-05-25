import { describe, expect, it } from "vitest";
import { withRetry } from "./retry.js";

describe("withRetry", () => {
  it("retries failing operations with retry callbacks", async () => {
    let attempts = 0;
    const retryEvents: { readonly attempt: number; readonly delayMs: number }[] = [];

    const result = await withRetry(
      () => {
        attempts += 1;
        if (attempts < 3) {
          return Promise.reject(new Error("temporary"));
        }
        return Promise.resolve("ok");
      },
      {
        retries: 2,
        delayMs: 0,
        jitterMs: 0,
        onRetry(event) {
          retryEvents.push({ attempt: event.attempt, delayMs: event.delayMs });
        },
      }
    );

    expect(result).toBe("ok");
    expect(attempts).toBe(3);
    expect(retryEvents).toEqual([
      { attempt: 1, delayMs: 0 },
      { attempt: 2, delayMs: 0 },
    ]);
  });

  it("does not retry errors rejected by shouldRetry", async () => {
    let attempts = 0;

    await expect(
      withRetry(
        () => {
          attempts += 1;
          return Promise.reject(new Error("configuration failed"));
        },
        {
          retries: 3,
          delayMs: 0,
          jitterMs: 0,
          shouldRetry: () => false,
        }
      )
    ).rejects.toThrow("configuration failed");

    expect(attempts).toBe(1);
  });
});
