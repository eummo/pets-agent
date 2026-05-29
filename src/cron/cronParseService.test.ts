import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
} from "@earendil-works/pi-ai";
import { LlmCronParseService } from "./cronParseService.js";

describe("LlmCronParseService", () => {
  let registration: ReturnType<typeof registerFauxProvider>;

  beforeEach(() => {
    registration = registerFauxProvider({ tokensPerSecond: 50 });
  });

  afterEach(() => {
    registration.unregister();
  });

  function createService(responses: ReturnType<typeof fauxAssistantMessage>[]): LlmCronParseService {
    registration.setResponses(responses);
    return new LlmCronParseService(registration.getModel(), "test-key");
  }

  it("parses a cron expression from natural language", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText(JSON.stringify({
        name: "每日早报",
        schedule: { type: "cron", expression: "0 9 * * 1-5" },
        prompt: "总结最近24小时的重要变更",
        workspacePath: ".harness/knowledge-base",
        delivery: { channels: ["sse:admin"] },
        enabled: true,
      }))]),
    ]);

    const result = await service.parse("每个工作日早上9点发送早报");
    expect(result.name).toBe("每日早报");
    expect(result.schedule).toEqual({ type: "cron", expression: "0 9 * * 1-5" });
    expect(result.prompt).toBe("总结最近24小时的重要变更");
  });

  it("parses an interval schedule from natural language", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText(JSON.stringify({
        name: "每小时检查",
        schedule: { type: "interval", milliseconds: 3600000 },
        prompt: "检查系统状态",
        workspacePath: ".harness/knowledge-base",
        delivery: { channels: ["wecom:chat:test-group"] },
      }))]),
    ]);

    const result = await service.parse("每小时检查一次系统状态");
    expect(result.schedule).toEqual({ type: "interval", milliseconds: 3600000 });
  });

  it("parses a one-shot schedule from natural language", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText(JSON.stringify({
        name: "一次性任务",
        schedule: { type: "once", runAt: "2026-06-01T09:00:00.000Z" },
        prompt: "执行部署前检查",
        workspacePath: ".harness/knowledge-base",
        delivery: { channels: ["sse:admin"] },
      }))]),
    ]);

    const result = await service.parse("6月1日早上9点执行部署前检查");
    expect(result.schedule).toEqual({ type: "once", runAt: "2026-06-01T09:00:00.000Z" });
  });

  it("handles markdown-fenced JSON responses", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("```json\n" + JSON.stringify({
        name: "带围栏的任务",
        schedule: { type: "cron", expression: "0 8 * * *" },
        prompt: "每日8点任务",
        workspacePath: ".harness/knowledge-base",
        delivery: { channels: ["sse:admin"] },
      }) + "\n```")]),
    ]);

    const result = await service.parse("每天8点执行任务");
    expect(result.name).toBe("带围栏的任务");
    expect(result.schedule).toEqual({ type: "cron", expression: "0 8 * * *" });
  });

  it("throws on invalid JSON from LLM", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText("not valid json at all")]),
    ]);
    await expect(service.parse("每天做点什么")).rejects.toThrow();
  });

  it("throws on JSON that fails schema validation", async () => {
    const service = createService([
      fauxAssistantMessage([fauxText(JSON.stringify({ name: "" }))]),
    ]);
    await expect(service.parse("每天做点什么")).rejects.toThrow();
  });
});
