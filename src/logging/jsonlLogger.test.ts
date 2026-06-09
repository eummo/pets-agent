import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createJsonlLogger, toLocalIsoString } from "./jsonlLogger.js";

describe("toLocalIsoString", () => {
  it("produces local time with timezone offset", () => {
    const date = new Date("2026-05-26T08:30:00.000Z");
    const result = toLocalIsoString(date);
    // Should contain timezone offset like +08:00 or -05:00
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}$/);
  });

  it("reflects local time, not UTC", () => {
    const utcNoon = new Date("2026-05-26T12:00:00.000Z");
    const result = toLocalIsoString(utcNoon);
    // Verify by parsing: the local time should equal what Date.getLocal methods return
    const expectedHour = utcNoon.getHours();
    const hourStr = result.split("T")[1]?.split(":")[0];
    const hourInResult = hourStr ? Number(hourStr) : -1;
    expect(hourInResult).toBe(expectedHour);
  });
});

describe("createJsonlLogger", () => {
  it("writes jsonl events and redacts secrets", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({
      message: "hello",
      apiKey: "sk-secretsecretsecret",
      nested: {
        accessToken: "plain-token",
        input_tokens: 123
      }
    });

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();

    expect(content).toContain('"message":"hello"');
    expect(content).toContain('"apiKey":"[REDACTED]"');
    expect(content).toContain('"accessToken":"[REDACTED]"');
    expect(content).toContain('"input_tokens":123');
    expect(content).not.toContain("sk-secretsecretsecret");
    expect(content.endsWith("\n")).toBe(true);
  });

  it("redacts sk- prefixed API keys in string values", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({
      message: "key is sk-abcdefghijklmnopqrstuv in the text"
    });

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();

    expect(content).toContain("[REDACTED_API_KEY]");
    expect(content).not.toContain("sk-abcdefghijklmnopqrstuv");
  });

  it("redacts secret, authorization, refresh-token, password, and cookie keys", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({
      secret: "my-secret",
      Authorization: "Bearer token",
      "refresh-token": "rt-value",
      password: "p@ssw0rd",
      cookie: "session=abc",
      safeField: "visible"
    });

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();

    expect(content).toContain('"secret":"[REDACTED]"');
    expect(content).toContain('"Authorization":"[REDACTED]"');
    expect(content).toContain('"refresh-token":"[REDACTED]"');
    expect(content).toContain('"password":"[REDACTED]"');
    expect(content).toContain('"cookie":"[REDACTED]"');
    expect(content).toContain('"safeField":"visible"');
  });

  it("adds a timestamp to each event", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({ message: "ts-test" });

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();
    // Local ISO format: 2026-05-26T16:30:00.123+08:00
    expect(content).toMatch(
      /"timestamp":"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}"/
    );
  });

  it("appends multiple events to the same file", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({ message: "first" });
    await logger.write({ message: "second" });

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();
    const lines = content.trim().split("\n");

    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain('"message":"first"');
    expect(lines[1]).toContain('"message":"second"');
  });

  it("serializes concurrent writes", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await Promise.all(
      Array.from({ length: 25 }, (_, index) => logger.write({ message: `event-${index}` }))
    );
    await logger.flush?.();

    const content = await readFile(logger.filePath, "utf8");
    await logger.close?.();
    const lines = content.trim().split("\n");

    expect(lines).toHaveLength(25);
    for (const line of lines) {
      expect(() => JSON.parse(line) as unknown).not.toThrow();
    }
    expect(new Set(lines.map((line) => (JSON.parse(line) as { message: string }).message))).toEqual(
      new Set(Array.from({ length: 25 }, (_, index) => `event-${index}`))
    );
  });

  it("closes idempotently and rejects writes after close", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({ message: "before-close" });
    await logger.close?.();
    await logger.close?.();

    await expect(logger.write({ message: "after-close" })).rejects.toThrow(
      "Cannot write to closed JSONL logger"
    );

    const content = await readFile(logger.filePath, "utf8");
    const lines = content.trim().split("\n");
    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain('"message":"before-close"');
  });
});
