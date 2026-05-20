import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  AnthropicManagedSessionAgentRuntime,
  type ManagedSessionsClient
} from "./anthropicManagedSessionAgentRuntime.js";

describe("AnthropicManagedSessionAgentRuntime", () => {
  it("creates a session, sends a grounded user message, and returns agent text", async () => {
    const workspacePath = await createWorkspace();
    const sentMessages: string[] = [];
    const client = createClient({
      sendText(text) {
        sentMessages.push(text);
      },
      events: [
        {
          id: "agent-1",
          type: "agent.message",
          processed_at: new Date().toISOString(),
          content: [{ type: "text", text: "Order service handles order workflows." }]
        },
        {
          id: "idle-1",
          type: "session.status_idle",
          processed_at: new Date().toISOString(),
          stop_reason: { type: "end_turn" }
        }
      ]
    });
    const runtime = new AnthropicManagedSessionAgentRuntime({
      baseUrl: "https://example.test",
      apiKey: "secret",
      agentId: "agent_123",
      environmentId: "env_123",
      client,
      pollIntervalMs: 0,
      maxPolls: 1
    });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "What does the order service do?",
      workspacePath
    });

    expect(response).toEqual({
      text: "Order service handles order workflows.",
      sessionId: "session-1"
    });
    expect(sentMessages[0]).toContain("Workspace context:");
    expect(sentMessages[0]).toContain("Order service owns order workflows.");
  });

  it("reuses a provided session id without creating a new session", async () => {
    let createCalls = 0;
    const client = createClient({
      createSession() {
        createCalls += 1;
      },
      events: [
        {
          id: "idle-1",
          type: "session.status_idle",
          processed_at: new Date().toISOString(),
          stop_reason: { type: "end_turn" }
        }
      ]
    });
    const runtime = new AnthropicManagedSessionAgentRuntime({
      baseUrl: "https://example.test",
      apiKey: "secret",
      agentId: "agent_123",
      environmentId: "env_123",
      client,
      pollIntervalMs: 0,
      maxPolls: 1
    });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "follow up",
      workspacePath: await createWorkspace(),
      sessionId: "session-existing"
    });

    expect(response.sessionId).toBe("session-existing");
    expect(createCalls).toBe(0);
  });

  it("archives sessions via the SDK", async () => {
    const archivedSessions: string[] = [];
    const client = createClient({
      archiveSession(sessionId) {
        archivedSessions.push(sessionId);
      },
      events: []
    });
    const runtime = new AnthropicManagedSessionAgentRuntime({
      baseUrl: "https://example.test",
      apiKey: "secret",
      agentId: "agent_123",
      environmentId: "env_123",
      client
    });

    await runtime.disposeSession("session-1");

    expect(archivedSessions).toEqual(["session-1"]);
  });
});

type ClientOptions = {
  readonly events: readonly unknown[];
  readonly sendText?: (text: string) => void;
  readonly createSession?: () => void;
  readonly archiveSession?: (sessionId: string) => void;
};

function createClient(options: ClientOptions): ManagedSessionsClient {
  return {
    beta: {
      sessions: {
        create() {
          options.createSession?.();
          return Promise.resolve({ id: "session-1" });
        },
        archive(sessionId) {
          options.archiveSession?.(sessionId);
          return Promise.resolve({});
        },
        events: {
          send(_sessionId, params) {
            const event = params.events[0];
            if (event?.type === "user.message") {
              const block = event.content[0];
              if (block?.type === "text") {
                options.sendText?.(block.text);
              }
            }
            return Promise.resolve({});
          },
          list() {
            return toAsyncIterable(options.events);
          }
        }
      }
    }
  } as ManagedSessionsClient;
}

async function createWorkspace(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "managed-session-runtime-"));
  await writeFile(path.join(root, "CLAUDE.md"), "Order service owns order workflows.", "utf8");
  return root;
}

async function* toAsyncIterable<T>(items: readonly T[]): AsyncIterable<T> {
  for (const item of items) {
    await Promise.resolve();
    yield item;
  }
}
