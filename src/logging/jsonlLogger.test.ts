import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createJsonlLogger } from "./jsonlLogger.js";

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

    expect(content).toContain("[REDACTED_API_KEY]");
    expect(content).not.toContain("sk-abcdefghijklmnopqrstuv");
  });

  it("redacts secret, authorization, and refresh-token keys", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({
      secret: "my-secret",
      Authorization: "Bearer token",
      "refresh-token": "rt-value",
      safeField: "visible"
    });

    const content = await readFile(logger.filePath, "utf8");

    expect(content).toContain('"secret":"[REDACTED]"');
    expect(content).toContain('"Authorization":"[REDACTED]"');
    expect(content).toContain('"refresh-token":"[REDACTED]"');
    expect(content).toContain('"safeField":"visible"');
  });

  it("adds a timestamp to each event", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({ message: "ts-test" });

    const content = await readFile(logger.filePath, "utf8");

    expect(content).toMatch(/"timestamp":"\d{4}-\d{2}-\d{2}T/);
  });

  it("appends multiple events to the same file", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "pets-agent-log-"));
    const logger = createJsonlLogger(path.join(dir, "events.jsonl"));

    await logger.write({ message: "first" });
    await logger.write({ message: "second" });

    const content = await readFile(logger.filePath, "utf8");
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

    const content = await readFile(logger.filePath, "utf8");
    const lines = content.trim().split("\n");

    expect(lines).toHaveLength(25);
    for (const line of lines) {
      expect(() => JSON.parse(line) as unknown).not.toThrow();
    }
    expect(new Set(lines.map((line) => (JSON.parse(line) as { message: string }).message))).toEqual(
      new Set(Array.from({ length: 25 }, (_, index) => `event-${index}`))
    );
  });
});
