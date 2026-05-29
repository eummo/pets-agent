import { describe, expect, it } from "vitest";
import { buildChatContext, splitAtHeadings, truncateToBudget } from "./workspacePromptBuilder.js";
import type { AgentRequest } from "../index.js";

describe("splitAtHeadings", () => {
  it("splits content at h1/h2/h3 headings", () => {
    const content = [
      "# Title",
      "Intro text",
      "## Section A",
      "Content A",
      "### Subsection",
      "Sub content",
      "## Section B",
      "Content B"
    ].join("\n");

    const sections = splitAtHeadings(content);

    expect(sections).toHaveLength(4);
    expect(sections[0]).toContain("# Title");
    expect(sections[0]).toContain("Intro text");
    expect(sections[1]).toContain("## Section A");
    expect(sections[2]).toContain("### Subsection");
    expect(sections[3]).toContain("## Section B");
  });

  it("treats content before the first heading as a single section", () => {
    const content = "Preamble text\n# First heading\nBody text";

    const sections = splitAtHeadings(content);

    expect(sections).toHaveLength(2);
    expect(sections[0]).toBe("Preamble text\n");
    expect(sections[1]).toContain("# First heading");
  });

  it("returns the whole content as one section when there are no headings", () => {
    const content = "Just plain text\nNo headings here";

    const sections = splitAtHeadings(content);

    expect(sections).toEqual(["Just plain text\nNo headings here\n"]);
  });

  it("does not split at h4 or deeper headings", () => {
    const content = "# Title\nBody\n#### Deep heading\nMore body";

    const sections = splitAtHeadings(content);

    expect(sections).toHaveLength(1);
    expect(sections[0]).toContain("#### Deep heading");
  });
});

describe("truncateToBudget", () => {
  it("returns the full content when it fits within the budget", () => {
    const content = "Short content";

    // splitAtHeadings adds trailing \n to the single section
    expect(truncateToBudget(content, 100)).toBe("Short content\n");
  });

  it("drops later sections to stay within the character budget", () => {
    const content = "# Section A\nAAA\n# Section B\nBBB\n# Section C\nCCC";

    const result = truncateToBudget(content, 40);

    expect(result).toContain("# Section A");
    expect(result).toContain("AAA");
    expect(result).toContain("# Section B");
    expect(result).toContain("BBB");
    expect(result).not.toContain("# Section C");
  });

  it("falls back to hard truncation when the first section exceeds the budget", () => {
    const content =
      "This is a very long first section with no headings at all that goes well beyond the budget limit.";

    const result = truncateToBudget(content, 20);

    expect(result).toBe(content.slice(0, 20));
  });

  it("includes complete sections only, never a partial section", () => {
    const content = "# A\nAA\n# B\nBB";

    const result = truncateToBudget(content, 10);

    // "# A\nAA\n" = 8 chars fits, "# B\nBB\n" = 7 chars, 8+7=15 > 10
    expect(result).toBe("# A\nAA\n");
  });
});

describe("buildChatContext", () => {
  it("returns undefined when chatType is not set", () => {
    const request: AgentRequest = {
      user: { id: "user-1" },
      text: "hello",
      workspacePath: "/tmp/ws"
    };
    expect(buildChatContext(request)).toBeUndefined();
  });

  it("returns group chat mention instructions for group chatType", () => {
    const request: AgentRequest = {
      user: { id: "zhangsan" },
      text: "hello",
      workspacePath: "/tmp/ws",
      chatType: "group"
    };
    const result = buildChatContext(request);
    expect(result).toBeDefined();
    expect(result).toContain("group chat");
    expect(result).toContain("zhangsan");
    expect(result).toContain("<@userid>");
    expect(result).toContain("<@zhangsan>");
    expect(result).toContain("@mention");
  });

  it("returns single chat no-mention instruction for single chatType", () => {
    const request: AgentRequest = {
      user: { id: "lisi" },
      text: "hello",
      workspacePath: "/tmp/ws",
      chatType: "single"
    };
    const result = buildChatContext(request);
    expect(result).toBeDefined();
    expect(result).toContain("single chat");
    expect(result).toContain("Do not use @mentions");
    expect(result).not.toContain("<@");
  });

  it("includes the sender userid in group chat context", () => {
    const request: AgentRequest = {
      user: { id: "wangwu" },
      text: "hello",
      workspacePath: "/tmp/ws",
      chatType: "group",
      chatId: "group-123"
    };
    const result = buildChatContext(request);
    expect(result).toContain("wangwu");
    expect(result).toContain("<@wangwu>");
  });
});
