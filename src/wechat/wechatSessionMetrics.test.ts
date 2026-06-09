import { afterEach, describe, expect, it, vi } from "vitest";
import type { ConversationLogger } from "../core/index.js";
import {
  startWechatSessionMetricsLogger,
  writeWechatSessionMetrics
} from "./wechatSessionMetrics.js";
import type { WechatSessionMetricsSource } from "./wechatSessionMetrics.js";

describe("writeWechatSessionMetrics", () => {
  it("writes a structured metrics event", async () => {
    const events: Record<string, unknown>[] = [];
    const source = makeMetricsSource();

    await writeWechatSessionMetrics(source, collectingLogger(events));

    expect(events).toEqual([
      {
        type: "wechat.session_metrics",
        connected: true,
        activeLockCount: 1,
        inflightMessageCount: 2,
        trackedSessionCount: 1,
        streamFailureCount: 3,
        connectionUnavailableRejectionCount: 4
      }
    ]);
  });
});

describe("startWechatSessionMetricsLogger", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("writes immediately and then on the configured interval", async () => {
    vi.useFakeTimers();
    const events: Record<string, unknown>[] = [];
    const source = makeMetricsSource();

    const stop = startWechatSessionMetricsLogger({
      source,
      logger: collectingLogger(events),
      intervalMs: 1_000
    });
    await Promise.resolve();

    expect(events).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(2_000);
    expect(events).toHaveLength(3);

    stop();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(events).toHaveLength(3);
  });
});

function makeMetricsSource(): WechatSessionMetricsSource {
  return {
    getSessionMetrics() {
      return {
        connected: true,
        activeLockCount: 1,
        inflightMessageCount: 2,
        trackedSessionCount: 1,
        streamFailureCount: 3,
        connectionUnavailableRejectionCount: 4
      };
    }
  };
}

function collectingLogger(events: Record<string, unknown>[]): ConversationLogger {
  return {
    write(event: Record<string, unknown>): Promise<void> {
      events.push(event);
      return Promise.resolve();
    }
  };
}
