import { describe, expect, it } from "vitest";
import { fallbackIntentFor } from "./intentHeuristics.js";

describe("fallbackIntentFor", () => {
  it("classifies Chinese creation explanation questions as query", () => {
    expect(fallbackIntentFor("客户订单是怎么创建的")).toEqual({ type: "query" });
  });

  it("still classifies Chinese mutation requests as mutate", () => {
    expect(fallbackIntentFor("请修改订单系统")).toEqual({ type: "mutate" });
  });
});
