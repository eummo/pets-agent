import { describe, expect, it, vi } from "vitest";
import { CompositeDeliveryChannel } from "./compositeDelivery.js";
import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";

function makePayload(overrides: Partial<DeliveryPayload> = {}): DeliveryPayload {
  return {
    jobName: "Test Job",
    output: "Summary result",
    ...overrides,
  };
}

type MockFn = ReturnType<typeof vi.fn>;

function createMockChannel(prefix: string, shouldFail = false): DeliveryChannel & { deliverMock: MockFn } {
  const deliverMock = shouldFail
    ? vi.fn(() => Promise.reject(new Error(`Delivery failed for ${prefix}`)))
    : vi.fn(() => Promise.resolve());

  return { prefix, deliver: deliverMock, deliverMock };
}

describe("CompositeDeliveryChannel", () => {
  it("routes to the correct channel by prefix", async () => {
    const sse = createMockChannel("sse");
    const wecom = createMockChannel("wecom");
    const composite = new CompositeDeliveryChannel([sse, wecom]);

    await composite.deliver("sse:admin", makePayload());

    expect(sse.deliverMock).toHaveBeenCalledWith("sse:admin", expect.anything());
    expect(wecom.deliverMock).not.toHaveBeenCalled();
  });

  it("routes to wecom channel", async () => {
    const sse = createMockChannel("sse");
    const wecom = createMockChannel("wecom");
    const composite = new CompositeDeliveryChannel([sse, wecom]);

    await composite.deliver("wecom:user:zhangsan", makePayload());

    expect(wecom.deliverMock).toHaveBeenCalledWith("wecom:user:zhangsan", expect.anything());
    expect(sse.deliverMock).not.toHaveBeenCalled();
  });

  it("throws when no matching channel is found", async () => {
    const sse = createMockChannel("sse");
    const composite = new CompositeDeliveryChannel([sse]);

    await expect(composite.deliver("unknown:target", makePayload())).rejects.toThrow(
      "No delivery channel found"
    );
  });

  it("deliverAll fans out to all channels", async () => {
    const sse = createMockChannel("sse");
    const wecom = createMockChannel("wecom");
    const composite = new CompositeDeliveryChannel([sse, wecom]);

    await composite.deliverAll(["sse:admin", "wecom:user:zhangsan"], makePayload());

    expect(sse.deliverMock).toHaveBeenCalledWith("sse:admin", expect.anything());
    expect(wecom.deliverMock).toHaveBeenCalledWith("wecom:user:zhangsan", expect.anything());
  });

  it("deliverAll continues on error and throws first error", async () => {
    const sse = createMockChannel("sse", true);
    const wecom = createMockChannel("wecom");
    const composite = new CompositeDeliveryChannel([sse, wecom]);

    await expect(
      composite.deliverAll(["sse:admin", "wecom:user:zhangsan"], makePayload())
    ).rejects.toThrow("Delivery failed for sse");

    expect(sse.deliverMock).toHaveBeenCalled();
    expect(wecom.deliverMock).toHaveBeenCalled();
  });

  it("deliverAll succeeds when all channels succeed", async () => {
    const sse = createMockChannel("sse");
    const wecom = createMockChannel("wecom");
    const composite = new CompositeDeliveryChannel([sse, wecom]);

    await expect(
      composite.deliverAll(["sse:admin", "wecom:user:zhangsan"], makePayload())
    ).resolves.toBeUndefined();
  });
});
