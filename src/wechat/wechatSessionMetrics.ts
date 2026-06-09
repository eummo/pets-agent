import type { ConversationLogger } from "../core/index.js";
import type { WechatSessionMetrics } from "./wechatSmartBotAdapter.js";

export type WechatSessionMetricsSource = {
  getSessionMetrics(): WechatSessionMetrics;
};

export type WechatSessionMetricsLoggerOptions = {
  readonly source: WechatSessionMetricsSource;
  readonly logger: ConversationLogger;
  readonly intervalMs?: number;
};

const DEFAULT_WECHAT_SESSION_METRICS_INTERVAL_MS = 60_000;

export function startWechatSessionMetricsLogger(
  options: WechatSessionMetricsLoggerOptions
): () => void {
  const intervalMs = options.intervalMs ?? DEFAULT_WECHAT_SESSION_METRICS_INTERVAL_MS;
  const logMetrics = (): void => {
    void writeWechatSessionMetrics(options.source, options.logger);
  };

  logMetrics();
  const interval = setInterval(logMetrics, intervalMs);
  return () => {
    clearInterval(interval);
  };
}

export async function writeWechatSessionMetrics(
  source: WechatSessionMetricsSource,
  logger: ConversationLogger
): Promise<void> {
  await logger.write({
    type: "wechat.session_metrics",
    ...source.getSessionMetrics()
  });
}
