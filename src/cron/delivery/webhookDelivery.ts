import { lookup } from "node:dns/promises";
import { isIP, isIPv4 } from "node:net";
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
    const rawUrl = target.slice(this.prefix.length + 1);
    if (rawUrl.length === 0) {
      throw new Error(`Invalid webhook delivery target: "${target}". Expected "webhook:<url>"`);
    }
    const url = await validateWebhookUrl(rawUrl);

    const body = {
      jobName: payload.jobName,
      output: payload.output,
      ...(payload.error !== undefined ? { error: payload.error } : {}),
      timestamp: new Date().toISOString()
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal
      });
      if (!response.ok) {
        throw new Error(`Webhook delivery failed: HTTP ${response.status} ${response.statusText}`);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
}

async function validateWebhookUrl(rawUrl: string): Promise<string> {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error("Invalid webhook URL.");
  }

  if (parsed.protocol !== "https:") {
    throw new Error("Webhook URL must use https.");
  }

  await rejectPrivateWebhookHost(parsed.hostname);
  return parsed.toString();
}

async function rejectPrivateWebhookHost(hostname: string): Promise<void> {
  const normalized = hostname.toLowerCase();
  if (isLocalHostname(normalized)) {
    throw new Error("Webhook URL must not target local or private network hosts.");
  }

  const directIpVersion = isIP(normalized);
  if (directIpVersion !== 0) {
    rejectPrivateAddress(normalized);
    return;
  }

  const addresses = await lookup(normalized, { all: true });
  for (const address of addresses) {
    rejectPrivateAddress(address.address);
  }
}

function isLocalHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname.endsWith(".localhost") || hostname.endsWith(".local");
}

function rejectPrivateAddress(address: string): void {
  if (isPrivateAddress(address)) {
    throw new Error("Webhook URL must not target local or private network hosts.");
  }
}

function isPrivateAddress(address: string): boolean {
  const normalized = address.toLowerCase();
  const ipv4 = normalized.startsWith("::ffff:") ? normalized.slice("::ffff:".length) : normalized;
  if (isIPv4(ipv4)) {
    const [firstOctet = 0, secondOctet = 0] = ipv4.split(".").map(Number);
    return (
      firstOctet === 0 ||
      firstOctet === 10 ||
      firstOctet === 127 ||
      (firstOctet === 100 && secondOctet >= 64 && secondOctet <= 127) ||
      (firstOctet === 169 && secondOctet === 254) ||
      (firstOctet === 172 && secondOctet >= 16 && secondOctet <= 31) ||
      (firstOctet === 192 && secondOctet === 168)
    );
  }

  return (
    normalized === "::1" ||
    normalized.startsWith("fc") ||
    normalized.startsWith("fd") ||
    normalized.startsWith("fe8") ||
    normalized.startsWith("fe9") ||
    normalized.startsWith("fea") ||
    normalized.startsWith("feb")
  );
}
