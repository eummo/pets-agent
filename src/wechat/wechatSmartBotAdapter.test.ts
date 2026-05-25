import { describe, expect, it } from "vitest";
import { stripBotMention } from "./wechatSmartBotAdapter.js";

describe("stripBotMention", () => {
  it("strips @bot mention prefix from group chat messages", () => {
    expect(stripBotMention("@RobotA hello world")).toBe("hello world");
  });

  it("strips @mention with underscore in bot name", () => {
    expect(stripBotMention("@My_Bot some question")).toBe("some question");
  });

  it("does not strip @mention in the middle of a message", () => {
    expect(stripBotMention("hello @bot world")).toBe("hello @bot world");
  });

  it("returns original text when no @mention prefix", () => {
    expect(stripBotMention("just a question")).toBe("just a question");
  });

  it("handles empty string", () => {
    expect(stripBotMention("")).toBe("");
  });

  it("handles @mention only with trailing space", () => {
    expect(stripBotMention("@Bot ")).toBe("");
  });
});
