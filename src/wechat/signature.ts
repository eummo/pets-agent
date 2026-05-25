import { createHash } from "node:crypto";

type VerifyWechatSignatureInput = {
  readonly token: string;
  readonly timestamp: string;
  readonly nonce: string;
  readonly signature: string;
};

const WECHAT_TIMESTAMP_TOLERANCE_MS = 5 * 60 * 1000;

export function createWechatSignature(
  token: string,
  timestamp: string,
  nonce: string,
  encryptedPayload = ""
): string {
  return createHash("sha1")
    .update([token, timestamp, nonce, encryptedPayload].sort().join(""))
    .digest("hex");
}

export function verifyWechatSignature(input: VerifyWechatSignatureInput): boolean {
  const timestampMs = Number.parseInt(input.timestamp, 10) * 1000;
  if (Number.isNaN(timestampMs)) {
    return false;
  }

  const age = Math.abs(Date.now() - timestampMs);
  if (age > WECHAT_TIMESTAMP_TOLERANCE_MS) {
    return false;
  }

  return createWechatSignature(input.token, input.timestamp, input.nonce) === input.signature;
}
