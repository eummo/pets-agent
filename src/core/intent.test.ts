import { describe, expect, it } from "vitest";
import { classifyMessageIntent } from "./intent.js";

describe("classifyMessageIntent", () => {
  it("classifies read questions", () => {
    expect(classifyMessageIntent("当前架构是什么？")).toBe("read");
    expect(classifyMessageIntent("what is the current architecture?")).toBe("read");
    expect(classifyMessageIntent("how does auth work?")).toBe("read");
  });

  it("classifies refactor requests as mutate intent", () => {
    expect(classifyMessageIntent("重构订单系统")).toBe("mutate");
    expect(classifyMessageIntent("refactor order service")).toBe("mutate");
  });

  it("classifies English mutate keywords", () => {
    const keywords = ["rewrite", "implement", "modify", "change", "fix", "update", "delete", "create", "add", "remove"];
    for (const keyword of keywords) {
      expect(classifyMessageIntent(`please ${keyword} the feature`)).toBe("mutate");
    }
  });

  it("classifies Chinese mutate keywords", () => {
    const keywords = ["实现", "修改", "修复", "更新", "删除", "创建", "新增", "调整", "改造", "优化"];
    for (const keyword of keywords) {
      expect(classifyMessageIntent(`请${keyword}这个功能`)).toBe("mutate");
    }
  });

  it("classifies empty string as read", () => {
    expect(classifyMessageIntent("")).toBe("read");
  });

  it("does not false-positive on read-only questions containing mutate words", () => {
    expect(classifyMessageIntent("what does the add function do?")).toBe("mutate");
    expect(classifyMessageIntent("how is the fix applied?")).toBe("mutate");
  });

  it("classifies mixed-language text", () => {
    expect(classifyMessageIntent("update the 订单")).toBe("mutate");
    expect(classifyMessageIntent("请 change this")).toBe("mutate");
  });
});
