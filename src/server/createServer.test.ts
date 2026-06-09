import { describe, expect, it } from "vitest";
import { mkdtemp, readFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import type {
  MessageGateway,
  OutboundMessage,
  UserRole,
  ChannelUser,
  InboundMessage
} from "../core/index.js";
import type {
  AuthorizationService,
  RoleCapability,
  AuthorizationAction,
  AuthorizationDecision
} from "../auth/index.js";
import type { FeedbackEntry } from "../persistence/index.js";
import { InMemoryRoleAuthorizationService } from "../auth/inMemoryRoleAuthorizationService.js";
import { createServer, type CreateServerOptions } from "./createServer.js";

describe("createServer", () => {
  it("serves health checks", async () => {
    const server = createServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({ method: "GET", url: "/health" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ ok: true, service: "pets-agent" });
  });

  it("serves liveness checks", async () => {
    const server = createServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({ method: "GET", url: "/healthz" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ ok: true, service: "pets-agent" });
  });

  it("serves readiness checks as ok when all checks pass", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      readinessChecks: [
        {
          name: "sqlite",
          check: () => ({ status: "ok" })
        }
      ]
    });

    const response = await server.inject({ method: "GET", url: "/readyz" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({
      ok: true,
      service: "pets-agent",
      status: "ok",
      checks: {
        sqlite: { status: "ok" }
      }
    });
  });

  it("serves readiness checks as degraded when a check warns", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      readinessChecks: [
        {
          name: "cron",
          check: () => ({ status: "warn", message: "disabled" })
        }
      ]
    });

    const response = await server.inject({ method: "GET", url: "/readyz" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({
      ok: true,
      service: "pets-agent",
      status: "degraded",
      checks: {
        cron: { status: "warn", message: "disabled" }
      }
    });
  });

  it("serves readiness checks as not ready when a check fails", async () => {
    const server = createServer({
      messageHandler: echoHandler,
      readinessChecks: [
        {
          name: "wechat_ws",
          check: () => ({ status: "fail", message: "disconnected" })
        }
      ]
    });

    const response = await server.inject({ method: "GET", url: "/readyz" });

    expect(response.statusCode).toBe(503);
    expect(response.json()).toEqual({
      ok: false,
      service: "pets-agent",
      status: "not_ready",
      checks: {
        wechat_ws: { status: "fail", message: "disconnected" }
      }
    });
  });

  it("serves the development chat page", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({ method: "GET", url: "/" });

    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toContain("text/html");
    expect(response.body).toContain("Pets Agent");
    expect(response.body).toContain("Claude Agent SDK");
  });

  it("rejects path traversal attempts for development chat assets", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/chat/..%2F..%2Fpackage.json"
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects development UI assets from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/chat/app.js",
      remoteAddress: "10.0.0.5"
    });

    expect(response.statusCode).toBe(403);
  });

  it("routes browser chat messages via SSE streaming", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "hello"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toContain("text/event-stream");
    // SSE format: contains event and data lines
    expect(response.body).toContain("event: agent");
    expect(response.body).toContain("completed");
    expect(response.body).toContain("received: hello");
  });

  it("routes browser chat document attachments through saved metadata", async () => {
    const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-uploads-"));
    let capturedMessage: InboundMessage | undefined;
    const content = "# Notes\nUploaded document facts.";
    const server = createDevServer({
      uploadRootPath,
      messageHandler: {
        handle(message) {
          capturedMessage = message;
          return Promise.resolve({ text: "ok" });
        }
      }
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "answer from upload",
        attachments: [
          {
            name: "notes.md",
            mimeType: "text/markdown",
            sizeBytes: Buffer.byteLength(content),
            contentBase64: Buffer.from(content, "utf8").toString("base64")
          }
        ]
      }
    });

    expect(response.statusCode).toBe(200);
    expect(capturedMessage?.attachments).toHaveLength(1);
    const attachment = capturedMessage?.attachments?.[0];
    expect(attachment).toMatchObject({
      type: "document",
      name: "notes.md",
      mimeType: "text/markdown",
      sizeBytes: Buffer.byteLength(content)
    });
    expect(attachment?.storagePath.startsWith(uploadRootPath)).toBe(true);
    await expect(readFile(attachment?.storagePath ?? "", "utf8")).resolves.toBe(content);
  });

  it("routes browser chat image attachments through saved metadata", async () => {
    const uploadRootPath = await mkdtemp(path.join(tmpdir(), "pets-agent-uploads-"));
    let capturedMessage: InboundMessage | undefined;
    const content = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    const server = createDevServer({
      uploadRootPath,
      messageHandler: {
        handle(message) {
          capturedMessage = message;
          return Promise.resolve({ text: "ok" });
        }
      }
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "describe the image",
        attachments: [
          {
            name: "diagram.png",
            mimeType: "application/octet-stream",
            sizeBytes: content.length,
            contentBase64: content.toString("base64")
          }
        ]
      }
    });

    expect(response.statusCode).toBe(200);
    const attachment = capturedMessage?.attachments?.[0];
    expect(attachment).toMatchObject({
      type: "image",
      name: "diagram.png",
      mimeType: "image/png",
      sizeBytes: content.length
    });
    expect(attachment?.storagePath.startsWith(uploadRootPath)).toBe(true);
    await expect(readFile(attachment?.storagePath ?? "")).resolves.toEqual(content);
  });

  it("rejects unsupported browser chat document attachments", async () => {
    let called = false;
    const server = createDevServer({
      messageHandler: {
        handle() {
          called = true;
          return Promise.resolve({ text: "ok" });
        }
      }
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "answer from upload",
        attachments: [
          {
            name: "notes.exe",
            mimeType: "application/octet-stream",
            sizeBytes: 4,
            contentBase64: Buffer.from("test", "utf8").toString("base64")
          }
        ]
      }
    });

    expect(response.statusCode).toBe(400);
    expect(response.json()).toEqual({
      error: "Uploaded attachment notes.exe must be a .txt, .md, or supported image file."
    });
    expect(called).toBe(false);
  });

  it("rejects browser chat messages from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "hello"
      },
      remoteAddress: "10.0.0.5"
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects development events from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/events?userId=browser-user",
      remoteAddress: "10.0.0.5"
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects role listing from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/roles",
      remoteAddress: "10.0.0.5"
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects role management from non-local clients", async () => {
    const authorization = new InMemoryRoleAuthorizationService();
    const server = createDevServer({
      messageHandler: echoHandler,
      authorization
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/role",
      payload: {
        userId: "browser-user",
        role: "developer"
      },
      remoteAddress: "10.0.0.5"
    });

    expect(response.statusCode).toBe(403);
  });

  it("sets development roles from the browser", async () => {
    const authorization = new InMemoryRoleAuthorizationService();
    const server = createDevServer({
      messageHandler: echoHandler,
      authorization
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/role",
      payload: {
        userId: "browser-user",
        role: "developer"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ userId: "browser-user", role: "developer" });
    await expect(authorization.roleFor({ id: "browser-user" })).resolves.toBe("developer");
  });

  it("accepts reviewer role", async () => {
    const authorization = new InMemoryRoleAuthorizationService();
    const server = createDevServer({
      messageHandler: echoHandler,
      authorization
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/role",
      payload: {
        userId: "browser-user",
        role: "reviewer"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ userId: "browser-user", role: "reviewer" });
    await expect(authorization.roleFor({ id: "browser-user" })).resolves.toBe("reviewer");
  });

  it("does not enable dev routes by default", async () => {
    const server = createServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({ method: "GET", url: "/" });

    expect(response.statusCode).toBe(404);
  });
});

describe("feedback endpoints", () => {
  it("returns 501 when feedback store is not configured", async () => {
    const server = createDevServer({
      messageHandler: echoHandler
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1"
    });

    expect(response.statusCode).toBe(501);
  });

  it("returns 403 when user lacks feedback_view capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({ "reviewer-1": ["workspace_read"] })
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=reviewer-1"
    });

    expect(response.statusCode).toBe(403);
  });

  it("returns feedback list for user with feedback_view capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({
        "admin-1": [
          "workspace_read",
          "workspace_mutate",
          "knowledge_base_update",
          "feedback_view",
          "feedback_manage"
        ]
      })
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1"
    });

    expect(response.statusCode).toBe(200);
    const body: { feedback: unknown[] } = response.json();
    expect(body.feedback).toBeDefined();
    expect(Array.isArray(body.feedback)).toBe(true);
  });

  it("rejects feedback access from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({
        "admin-1": [
          "workspace_read",
          "workspace_mutate",
          "knowledge_base_update",
          "feedback_view",
          "feedback_manage"
        ]
      })
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1",
      remoteAddress: "10.0.0.8"
    });

    expect(response.statusCode).toBe(403);
  });

  it("returns 403 when updating feedback without feedback_manage capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({
        "dev-1": ["workspace_read", "workspace_mutate", "feedback_view"]
      })
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/1",
      payload: { status: "reviewed", userId: "dev-1" }
    });

    expect(response.statusCode).toBe(403);
  });

  it("updates feedback status with feedback_manage capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({
        "admin-1": [
          "workspace_read",
          "workspace_mutate",
          "knowledge_base_update",
          "feedback_view",
          "feedback_manage"
        ]
      })
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/1",
      payload: { status: "reviewed", userId: "admin-1" }
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ id: 1, status: "reviewed" });
  });

  it("returns 404 when updating missing feedback", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore({ existingIds: [1] }),
      authorization: makeAuthorization({
        "admin-1": [
          "workspace_read",
          "workspace_mutate",
          "knowledge_base_update",
          "feedback_view",
          "feedback_manage"
        ]
      })
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/999",
      payload: { status: "reviewed", userId: "admin-1" }
    });

    expect(response.statusCode).toBe(404);
  });
});

const echoHandler: MessageGateway = {
  handle(message) {
    const result: OutboundMessage = { text: `received: ${message.text}` };
    return Promise.resolve(result);
  }
};

function createDevServer(options: Omit<CreateServerOptions, "enableDevRoutes">) {
  return createServer({ ...options, enableDevRoutes: true });
}

function makeFeedbackStore(options: { readonly existingIds?: readonly number[] } = {}) {
  const entries: readonly FeedbackEntry[] = [
    {
      id: 1,
      userId: "user-1",
      userMessage: "update the docs",
      conversationContext: "",
      status: "pending",
      createdAt: "2026-01-01T00:00:00Z"
    }
  ];
  const existingIds = options.existingIds ?? [1];
  return {
    save: () => Promise.resolve(1),
    updateStatus: (id: number) => Promise.resolve(existingIds.includes(id)),
    getAll: () => Promise.resolve(entries)
  };
}

function makeAuthorization(
  roleCapabilities: Record<string, readonly RoleCapability[]>
): AuthorizationService {
  return {
    roleFor(user: ChannelUser): Promise<UserRole> {
      return Promise.resolve(roleCapabilities[user.id] ? user.id : "reviewer");
    },
    can(user: ChannelUser, action: AuthorizationAction): Promise<AuthorizationDecision> {
      const caps = roleCapabilities[user.id] ?? ["workspace_read"];
      const required =
        action === "mutate"
          ? "workspace_mutate"
          : action === "update_kb"
            ? "knowledge_base_update"
            : "workspace_read";
      return Promise.resolve(
        caps.includes(required)
          ? { allowed: true }
          : { allowed: false, reason: "Insufficient permissions" }
      );
    },
    hasCapability(user: ChannelUser, capability: RoleCapability): Promise<boolean> {
      const caps = roleCapabilities[user.id] ?? ["workspace_read"];
      return Promise.resolve(caps.includes(capability));
    }
  };
}
