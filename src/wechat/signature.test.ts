import { describe, expect, it } from "vitest";
import { createWechatSignature, verifyWechatSignature } from "./signature.js";

describe("wechat signature", () => {
  it("verifies token, timestamp, and nonce signatures", () => {
    const signature = createWechatSignature("token", "123", "nonce");

    expect(
      verifyWechatSignature({
        token: "token",
        timestamp: "123",
        nonce: "nonce",
        signature
      })
    ).toBe(true);
  });

  it("rejects mismatched signatures", () => {
    expect(
      verifyWechatSignature({
        token: "token",
        timestamp: "123",
        nonce: "nonce",
        signature: "bad"
      })
    ).toBe(false);
  });
});
