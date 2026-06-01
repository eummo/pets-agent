import { describe, expect, it } from "vitest";
import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService
} from "../auth/index.js";
import type { KnowledgeWorkspace } from "../workspace/index.js";
import type { ChannelUser, InboundMessage } from "./index.js";
import { RequestAuthorizationGate } from "./requestAuthorizationGate.js";

describe("RequestAuthorizationGate", () => {
  it("denies read access before detecting intent", async () => {
    let intentDetected = false;
    const gate = new RequestAuthorizationGate({
      authorization: authorizationWithCan(() => ({ allowed: false, reason: "no read" })),
      detectIntent() {
        intentDetected = true;
        return Promise.resolve({ type: "mutate" });
      }
    });

    const result = await gate.evaluate({
      message: testMessage("please change files"),
      workspace
    });

    expect(result).toMatchObject({
      status: "denied",
      deniedAt: "read",
      role: "reviewer",
      responseText: "no read"
    });
    expect(intentDetected).toBe(false);
  });

  it("authorizes intent actions through the trusted role override", async () => {
    const checkedActions: AuthorizationAction[] = [];
    const authorization: AuthorizationService = {
      roleFor() {
        return Promise.resolve("reviewer");
      },
      can() {
        return Promise.resolve({ allowed: false });
      },
      canRole(role, action) {
        checkedActions.push(action);
        return Promise.resolve({
          allowed: role === "developer" && (action === "read" || action === "mutate")
        });
      },
      hasCapability() {
        return Promise.resolve(false);
      }
    };
    const gate = new RequestAuthorizationGate({
      authorization,
      detectIntent() {
        return Promise.resolve({ type: "mutate" });
      }
    });

    const result = await gate.evaluate({
      message: { ...testMessage("please change files"), roleOverride: "developer" },
      workspace
    });

    expect(result).toMatchObject({
      status: "allowed",
      role: "developer",
      intent: { type: "mutate" },
      requiredAction: "mutate"
    });
    expect(checkedActions).toEqual(["read", "mutate"]);
  });

  it("returns a denied intent decision when the role lacks the action", async () => {
    const gate = new RequestAuthorizationGate({
      authorization: authorizationWithCan((action) =>
        action === "mutate" ? { allowed: false, reason: "no mutate" } : { allowed: true }
      ),
      detectIntent() {
        return Promise.resolve({ type: "mutate" });
      }
    });

    const result = await gate.evaluate({
      message: testMessage("please change files"),
      workspace
    });

    expect(result).toMatchObject({
      status: "denied",
      deniedAt: "intent",
      role: "reviewer",
      intent: { type: "mutate" },
      requiredAction: "mutate",
      decision: { allowed: false, reason: "no mutate" }
    });
    if (result.status !== "denied") {
      throw new Error("expected intent authorization to be denied");
    }
    expect(result.responseText.length).toBeGreaterThan(0);
  });
});

const workspace: KnowledgeWorkspace = {
  kind: "knowledge-base",
  id: "kb",
  path: "D:/kb"
};

function authorizationWithCan(
  decide: (action: AuthorizationAction) => AuthorizationDecision
): AuthorizationService {
  return {
    roleFor() {
      return Promise.resolve("reviewer");
    },
    can(_user: ChannelUser, action: AuthorizationAction) {
      return Promise.resolve(decide(action));
    },
    hasCapability() {
      return Promise.resolve(false);
    }
  };
}

function testMessage(text: string): InboundMessage {
  return {
    id: "message-1",
    channel: "test",
    user: { id: "user-1" },
    text,
    receivedAt: new Date()
  };
}
