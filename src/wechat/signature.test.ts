import { describe, expect, it } from "vitest";
import { createWechatSignature, verifyWechatSignature } from "./signature.js";

describe("wechat signature", () => {
  it("verifies token, timestamp, and nonce signatures", () => {
    const timestamp = String(Math.floor(Date.now() / 1000));
    const signature = createWechatSignature("token", timestamp, "nonce");

    expect(
      verifyWechatSignature({
        token: "token",
        timestamp,
        nonce: "nonce",
        signature
      })
    ).toBe(true);
  });

  it("rejects mismatched signatures", () => {
    const timestamp = String(Math.floor(Date.now() / 1000));
    expect(
      verifyWechatSignature({
        token: "token",
        timestamp,
        nonce: "nonce",
        signature: "bad"
      })
    ).toBe(false);
  });

  it("rejects signatures with stale timestamps", () => {
    const staleTimestamp = String(Math.floor(Date.now() / 1000) - 600);
    const signature = createWechatSignature("token", staleTimestamp, "nonce");

    expect(
      verifyWechatSignature({
        token: "token",
        timestamp: staleTimestamp,
        nonce: "nonce",
        signature
      })
    ).toBe(false);
  });

  it("rejects signatures with future timestamps beyond tolerance", () => {
    const futureTimestamp = String(Math.floor(Date.now() / 1000) + 600);
    const signature = createWechatSignature("token", futureTimestamp, "nonce");

    expect(
      verifyWechatSignature({
        token: "token",
        timestamp: futureTimestamp,
        nonce: "nonce",
        signature
      })
    ).toBe(false);
  });

  it("rejects invalid timestamp strings", () => {
    const signature = createWechatSignature("token", "not-a-number", "nonce");

    expect(
      verifyWechatSignature({
        token: "token",
        timestamp: "not-a-number",
        nonce: "nonce",
        signature
      })
    ).toBe(false);
  });
});
