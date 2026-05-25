export function normalizeOptionalText(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized === "" ? undefined : normalized;
}

export function isLocalRequest(ip: string): boolean {
  return ip === "127.0.0.1"
    || ip === "::1"
    || ip === "::ffff:127.0.0.1";
}
