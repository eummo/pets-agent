import { createHash } from "node:crypto";

export type VerifyWechatSignatureInput = {
  readonly token: string;
  readonly timestamp: string;
  readonly nonce: string;
  readonly signature: string;
};

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
  return createWechatSignature(input.token, input.timestamp, input.nonce) === input.signature;
}
