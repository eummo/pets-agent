import { afterEach, describe, expect, it, vi } from "vitest";
import { WebhookDeliveryChannel } from "./webhookDelivery.js";

const payload = {
  jobName: "Daily Report",
  output: "Done"
};

describe("WebhookDeliveryChannel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts payloads to valid public https targets", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response("", { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);

    await new WebhookDeliveryChannel().deliver("webhook:https://203.0.113.10/hook", payload);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://203.0.113.10/hook",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" }
      })
    );
  });

  it("rejects non-https targets before sending", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      new WebhookDeliveryChannel().deliver("webhook:http://example.com/hook", payload)
    ).rejects.toThrow("Webhook URL must use https.");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects localhost targets before sending", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      new WebhookDeliveryChannel().deliver("webhook:https://localhost/hook", payload)
    ).rejects.toThrow("Webhook URL must not target local or private network hosts.");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects private network targets before sending", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      new WebhookDeliveryChannel().deliver("webhook:https://192.168.1.10/hook", payload)
    ).rejects.toThrow("Webhook URL must not target local or private network hosts.");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
