import type { ServerResponse } from "node:http";
import { describe, expect, it } from "vitest";
import { DevProgressBroker } from "./progressBroker.js";

describe("DevProgressBroker", () => {
  it("sends a connected event when a user subscribes", () => {
    const broker = new DevProgressBroker();
    const response = createWritableResponse();

    broker.subscribe("user-1", response);

    expect(response.joinedWrites()).toContain("event: progress\n");
    expect(response.joinedWrites()).toContain('"stage":"events.connected"');
    expect(response.joinedWrites()).toContain("\n\n");
  });

  it("publishes progress events only to subscribers for the target user", async () => {
    const broker = new DevProgressBroker();
    const targetResponse = createWritableResponse();
    const otherResponse = createWritableResponse();

    broker.subscribe("user-1", targetResponse);
    broker.subscribe("user-2", otherResponse);
    await broker.publish(
      { id: "user-1" },
      {
        stage: "agent.thinking",
        message: "Thinking",
        data: { step: 1 }
      }
    );

    expect(targetResponse.joinedWrites()).toContain('"stage":"agent.thinking"');
    expect(targetResponse.joinedWrites()).toContain('"step":1');
    expect(targetResponse.joinedWrites()).toContain('"timestamp"');
    expect(otherResponse.joinedWrites()).not.toContain('"stage":"agent.thinking"');
  });

  it("stops publishing after the unsubscribe callback is called", async () => {
    const broker = new DevProgressBroker();
    const response = createWritableResponse();
    const unsubscribe = broker.subscribe("user-1", response);

    unsubscribe();
    await broker.publish({ id: "user-1" }, { stage: "agent.completed", message: "Done" });

    expect(response.joinedWrites()).not.toContain('"stage":"agent.completed"');
  });

  it("resolves publishing when there are no subscribers", async () => {
    const broker = new DevProgressBroker();

    await expect(
      broker.publish({ id: "user-1" }, { stage: "agent.completed", message: "Done" })
    ).resolves.toBeUndefined();
  });
});

type WritableResponse = ServerResponse & {
  readonly writes: string[];
  joinedWrites(): string;
};

function createWritableResponse(): WritableResponse {
  const writes: string[] = [];
  return {
    writes,
    write(chunk: string | Uint8Array): boolean {
      writes.push(typeof chunk === "string" ? chunk : chunk.toString());
      return true;
    },
    joinedWrites() {
      return writes.join("");
    }
  } as WritableResponse;
}
