import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export type JsonlLogger = {
  readonly filePath: string;
  write(event: Record<string, unknown>): Promise<void>;
};

export function createJsonlLogger(filePathInput: string): JsonlLogger {
  const filePath = path.resolve(filePathInput);

  return {
    filePath,
    async write(event) {
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, `${JSON.stringify(withTimestamp(redactRecord(event)))}\n`, {
        flag: "a",
        encoding: "utf8"
      });
    }
  };
}

function redactRecord(event: Record<string, unknown>): Record<string, unknown> {
  return redactSecrets(event) as Record<string, unknown>;
}

function withTimestamp(event: Record<string, unknown>): Record<string, unknown> {
  return {
    timestamp: new Date().toISOString(),
    ...event
  };
}

function redactSecrets(value: unknown): unknown {
  if (typeof value === "string") {
    return value.replaceAll(/sk-[A-Za-z0-9_-]{12,}/g, "[REDACTED_API_KEY]");
  }

  if (Array.isArray(value)) {
    return value.map((item) => redactSecrets(item));
  }

  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        /api[-_]?key|secret|authorization|access[-_]?token|refresh[-_]?token/i.test(key)
          ? "[REDACTED]"
          : redactSecrets(item)
      ])
    );
  }

  return value;
}
