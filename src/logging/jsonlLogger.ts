import { createWriteStream, type WriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { isRecord } from "../core/unknownRecord.js";

export type JsonlLogger = {
  readonly filePath: string;
  write(event: Record<string, unknown>): Promise<void>;
  flush?(): Promise<void>;
  close?(): Promise<void>;
};

export function createJsonlLogger(filePathInput: string): JsonlLogger {
  const filePath = path.resolve(filePathInput);
  const directoryReady = mkdir(path.dirname(filePath), { recursive: true });
  let writeQueue: Promise<void> = Promise.resolve();
  let stream: WriteStream | undefined;
  let closed = false;

  return {
    filePath,
    async write(event) {
      const line = `${JSON.stringify(withTimestamp(redactRecord(event)))}\n`;
      await enqueue(async () => {
        if (closed) {
          throw new Error(`Cannot write to closed JSONL logger: ${filePath}`);
        }
        await writeLine(await getStream(), line);
      });
    },
    async flush() {
      await enqueue(() => Promise.resolve());
    },
    async close() {
      await enqueue(async () => {
        if (closed) {
          return;
        }
        closed = true;
        if (stream !== undefined) {
          await endStream(stream);
          stream = undefined;
        }
      });
    }
  };

  async function getStream(): Promise<WriteStream> {
    await directoryReady;
    stream ??= createWriteStream(filePath, { flags: "a", encoding: "utf8" });
    return stream;
  }

  async function enqueue(operation: () => Promise<void>): Promise<void> {
    const nextWrite = writeQueue.catch(() => undefined).then(operation);
    writeQueue = nextWrite.catch(() => undefined);
    await nextWrite;
  }
}

function writeLine(stream: WriteStream, line: string): Promise<void> {
  return new Promise((resolve, reject) => {
    stream.write(line, "utf8", (error) => {
      if (error !== undefined && error !== null) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function endStream(stream: WriteStream): Promise<void> {
  return new Promise((resolve, reject) => {
    stream.once("error", reject);
    stream.end(() => {
      stream.off("error", reject);
      resolve();
    });
  });
}

function redactRecord(event: Record<string, unknown>): Record<string, unknown> {
  const redacted = redactSecrets(event);
  return isRecord(redacted) ? redacted : {};
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
        /api[-_]?key|secret|authorization|access[-_]?token|refresh[-_]?token|password|cookie/i.test(
          key
        )
          ? "[REDACTED]"
          : redactSecrets(item)
      ])
    );
  }

  return value;
}
