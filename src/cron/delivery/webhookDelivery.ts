import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";

/**
 * Delivers cron job results by POSTing JSON to a user-provided URL.
 *
 * Target format: "webhook:<url>"
 * E.g. "webhook:https://example.com/hook"
 *
 * POST body: { jobName, output, error?, timestamp }
 * Timeout: 10 seconds. No retry for v1.
 */
export class WebhookDeliveryChannel implements DeliveryChannel {
  public readonly prefix = "webhook";

  public async deliver(target: string, payload: DeliveryPayload): Promise<void> {
    const url = target.slice(this.prefix.length + 1);
    if (url.length === 0) {
      throw new Error(`Invalid webhook delivery target: "${target}". Expected "webhook:<url>"`);
    }

    const body = {
      jobName: payload.jobName,
      output: payload.output,
      ...(payload.error !== undefined ? { error: payload.error } : {}),
      timestamp: new Date().toISOString(),
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`Webhook delivery failed: HTTP ${response.status} ${response.statusText}`);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
}
