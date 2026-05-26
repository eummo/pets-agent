import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export type JsonlLogger = {
  readonly filePath: string;
  write(event: Record<string, unknown>): Promise<void>;
};

export function createJsonlLogger(filePathInput: string): JsonlLogger {
  const filePath = path.resolve(filePathInput);
  const directoryReady = mkdir(path.dirname(filePath), { recursive: true });
  let writeQueue: Promise<void> = Promise.resolve();

  return {
    filePath,
    async write(event) {
      const line = `${JSON.stringify(withTimestamp(redactRecord(event)))}\n`;
      const writeOperation = async (): Promise<void> => {
        await directoryReady;
        await writeFile(filePath, line, {
          flag: "a",
          encoding: "utf8"
        });
      };
      const nextWrite = writeQueue.catch(() => undefined).then(writeOperation);
      writeQueue = nextWrite.catch(() => undefined);
      await nextWrite;
    }
  };
}

function redactRecord(event: Record<string, unknown>): Record<string, unknown> {
  return redactSecrets(event) as Record<string, unknown>;
}

export function toLocalIsoString(date: Date): string {
  const pad = (n: number, width = 2): string => String(n).padStart(width, "0");
  const offsetMin = date.getTimezoneOffset();
  const sign = offsetMin <= 0 ? "+" : "-";
  const absOffsetMin = Math.abs(offsetMin);
  const offsetHours = pad(Math.floor(absOffsetMin / 60));
  const offsetMins = pad(absOffsetMin % 60);

  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
    `.${pad(date.getMilliseconds(), 3)}${sign}${offsetHours}:${offsetMins}`
  );
}

function withTimestamp(event: Record<string, unknown>): Record<string, unknown> {
  return {
    timestamp: toLocalIsoString(new Date()),
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
