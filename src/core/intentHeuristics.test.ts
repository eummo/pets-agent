import { describe, expect, it } from "vitest";
import { fallbackIntentFor, reconcileIntentWithHeuristics } from "./intentHeuristics.js";

describe("fallbackIntentFor", () => {
  it("classifies Chinese creation explanation questions as query", () => {
    expect(fallbackIntentFor("客户订单是怎么创建的")).toEqual({ type: "query" });
  });

  it("still classifies Chinese mutation requests as mutate", () => {
    expect(fallbackIntentFor("请修改订单系统")).toEqual({ type: "mutate" });
  });

  it("keeps uploaded image acknowledgements as query even if a model marks them mutate", () => {
    expect(
      reconcileIntentWithHeuristics("Acknowledge the uploaded image named smoke-diagram.png.", {
        type: "mutate"
      })
    ).toEqual({ type: "query" });
  });

  it("does not override explicit uploaded document update requests", () => {
    expect(
      reconcileIntentWithHeuristics("Update the uploaded document in the knowledge base.", {
        type: "update_kb"
      })
    ).toEqual({ type: "update_kb" });
  });
});
