import { describe, expect, it, vi, beforeEach } from "vitest";
import { WecomAppMessageDeliveryChannel } from "./wecomAppMessageDelivery.js";
import type { WecomDeliveryConfig } from "./wecomAppMessageDelivery.js";
import type { DeliveryPayload } from "../cronTypes.js";

function makeConfig(overrides: Partial<WecomDeliveryConfig> = {}): WecomDeliveryConfig {
  return {
    corpId: "test-corp-id",
    corpSecret: "test-corp-secret",
    agentId: "1000002",
    tokenCacheMs: 7_200_000,
    ...overrides,
  };
}

function makePayload(overrides: Partial<DeliveryPayload> = {}): DeliveryPayload {
  return {
    jobName: "Test Job",
    output: "Summary result",
    ...overrides,
  };
}

function mockGettoken(): Response {
  return new Response(JSON.stringify({ errcode: 0, errmsg: "ok", access_token: "test-token", expires_in: 7200 }), {
    headers: { "Content-Type": "application/json" },
  });
}

function mockSendSuccess(): Response {
  return new Response(JSON.stringify({ errcode: 0, errmsg: "ok" }), {
    headers: { "Content-Type": "application/json" },
  });
}

type WecomSendBody = { touser: string; msgtype: string; agentid: number; markdown: { content: string } };

function extractBody(call: unknown[]): WecomSendBody {
  const init = call[1] as RequestInit;
  if (typeof init.body !== "string") {
    throw new Error("Expected string body");
  }
  return JSON.parse(init.body) as WecomSendBody;
}

describe("WecomAppMessageDeliveryChannel", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
  });

  it("has correct prefix", () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    expect(channel.prefix).toBe("wecom");
  });

  it("resolves user target correctly", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce(mockSendSuccess());

    await channel.deliver("wecom:user:zhangsan", makePayload());

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const body = extractBody(fetchSpy.mock.calls[1] as unknown[]);
    expect(body.touser).toBe("zhangsan");
    expect(body.msgtype).toBe("markdown");
    expect(body.agentid).toBe(1000002);
  });

  it("resolves chat target correctly", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce(mockSendSuccess());

    await channel.deliver("wecom:chat:mygroup", makePayload());

    const body = extractBody(fetchSpy.mock.calls[1] as unknown[]);
    expect(body.touser).toBe("mygroup");
  });

  it("resolves @all target correctly", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce(mockSendSuccess());

    await channel.deliver("wecom:@all", makePayload());

    const body = extractBody(fetchSpy.mock.calls[1] as unknown[]);
    expect(body.touser).toBe("@all");
  });

  it("caches access token for subsequent calls", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());

    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce(mockSendSuccess());
    await channel.deliver("wecom:user:zhangsan", makePayload());

    fetchSpy.mockResolvedValueOnce(mockSendSuccess());
    await channel.deliver("wecom:user:lisi", makePayload());

    expect(fetchSpy).toHaveBeenCalledTimes(3);
  });

  it("throws on invalid target format", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    await expect(channel.deliver("wecom:invalid", makePayload())).rejects.toThrow(
      "Invalid WeCom delivery target"
    );
  });

  it("throws when gettoken fails", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ errcode: 40013, errmsg: "invalid corpid" }),
    });

    await expect(channel.deliver("wecom:user:zhangsan", makePayload())).rejects.toThrow(
      "WeCom gettoken failed"
    );
  });

  it("throws when message send fails", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ errcode: 40014, errmsg: "invalid access_token" }),
    });

    await expect(channel.deliver("wecom:user:zhangsan", makePayload())).rejects.toThrow(
      "WeCom message send failed"
    );
  });

  it("formats error payload correctly in markdown", async () => {
    const channel = new WecomAppMessageDeliveryChannel(makeConfig());
    fetchSpy.mockResolvedValueOnce(mockGettoken());
    fetchSpy.mockResolvedValueOnce(mockSendSuccess());

    await channel.deliver("wecom:user:zhangsan", makePayload({ error: "Connection timeout" }));

    const body = extractBody(fetchSpy.mock.calls[1] as unknown[]);
    expect(body.markdown.content).toContain("error");
    expect(body.markdown.content).toContain("Connection timeout");
  });
});
