import { describe, expect, it } from "vitest";
import type { MessageGateway, OutboundMessage, UserRole, ChannelUser } from "../core/index.js";
import type { AuthorizationService, RoleCapability, AuthorizationAction, AuthorizationDecision } from "../auth/index.js";
import type { FeedbackEntry } from "../persistence/index.js";
import { InMemoryRoleAuthorizationService } from "../auth/inMemoryRoleAuthorizationService.js";
import { createServer, type CreateServerOptions } from "./createServer.js";

describe("createServer", () => {
  it("serves health checks", async () => {
    const server = createServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({ method: "GET", url: "/health" });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ ok: true, service: "pets-agent" });
  });

  it("serves the development chat page", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({ method: "GET", url: "/" });

    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toContain("text/html");
    expect(response.body).toContain("Pets Agent");
    expect(response.body).toContain("Claude Agent SDK");
  });

  it("rejects path traversal attempts for development chat assets", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({ method: "GET", url: "/dev/chat/..%2F..%2Fpackage.json" });

    expect(response.statusCode).toBe(403);
  });

  it("rejects development UI assets from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/chat/app.js",
      remoteAddress: "10.0.0.5",
    });

    expect(response.statusCode).toBe(403);
  });

  it("routes browser chat messages via SSE streaming", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
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

  it("rejects browser chat messages from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({
      method: "POST",
      url: "/dev/chat",
      payload: {
        userId: "browser-user",
        text: "hello"
      },
      remoteAddress: "10.0.0.5",
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects development events from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/events?userId=browser-user",
      remoteAddress: "10.0.0.5",
    });

    expect(response.statusCode).toBe(403);
  });

  it("rejects role listing from non-local clients", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/roles",
      remoteAddress: "10.0.0.5",
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
      remoteAddress: "10.0.0.5",
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
      messageHandler: echoHandler,
    });

    const response = await server.inject({ method: "GET", url: "/" });

    expect(response.statusCode).toBe(404);
  });
});

describe("feedback endpoints", () => {
  it("returns 501 when feedback store is not configured", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1",
    });

    expect(response.statusCode).toBe(501);
  });

  it("returns 403 when user lacks feedback_view capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({ "reviewer-1": ["workspace_read"] }),
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=reviewer-1",
    });

    expect(response.statusCode).toBe(403);
  });

  it("returns feedback list for user with feedback_view capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({ "admin-1": ["workspace_read", "workspace_mutate", "knowledge_base_update", "feedback_view", "feedback_manage"] }),
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1",
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
      authorization: makeAuthorization({ "admin-1": ["workspace_read", "workspace_mutate", "knowledge_base_update", "feedback_view", "feedback_manage"] }),
    });

    const response = await server.inject({
      method: "GET",
      url: "/dev/feedback?userId=admin-1",
      remoteAddress: "10.0.0.8",
    });

    expect(response.statusCode).toBe(403);
  });

  it("returns 403 when updating feedback without feedback_manage capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({ "dev-1": ["workspace_read", "workspace_mutate", "feedback_view"] }),
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/1",
      payload: { status: "reviewed", userId: "dev-1" },
    });

    expect(response.statusCode).toBe(403);
  });

  it("updates feedback status with feedback_manage capability", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore(),
      authorization: makeAuthorization({ "admin-1": ["workspace_read", "workspace_mutate", "knowledge_base_update", "feedback_view", "feedback_manage"] }),
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/1",
      payload: { status: "reviewed", userId: "admin-1" },
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ id: 1, status: "reviewed" });
  });

  it("returns 404 when updating missing feedback", async () => {
    const server = createDevServer({
      messageHandler: echoHandler,
      feedbackStore: makeFeedbackStore({ existingIds: [1] }),
      authorization: makeAuthorization({ "admin-1": ["workspace_read", "workspace_mutate", "knowledge_base_update", "feedback_view", "feedback_manage"] }),
    });

    const response = await server.inject({
      method: "PATCH",
      url: "/dev/feedback/999",
      payload: { status: "reviewed", userId: "admin-1" },
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
    { id: 1, userId: "user-1", userMessage: "update the docs", conversationContext: "", status: "pending", createdAt: "2026-01-01T00:00:00Z" },
  ];
  const existingIds = options.existingIds ?? [1];
  return {
    save: () => Promise.resolve(1),
    updateStatus: (id: number) => Promise.resolve(existingIds.includes(id)),
    getAll: () => Promise.resolve(entries),
  };
}

function makeAuthorization(roleCapabilities: Record<string, readonly RoleCapability[]>): AuthorizationService {
  return {
    roleFor(user: ChannelUser): Promise<UserRole> {
      return Promise.resolve(roleCapabilities[user.id] ? user.id : "reviewer");
    },
    can(user: ChannelUser, action: AuthorizationAction): Promise<AuthorizationDecision> {
      const caps = roleCapabilities[user.id] ?? ["workspace_read"];
      const required = action === "mutate"
        ? "workspace_mutate"
        : action === "update_kb"
          ? "knowledge_base_update"
          : "workspace_read";
      return Promise.resolve(caps.includes(required) ? { allowed: true } : { allowed: false, reason: "Insufficient permissions" });
    },
    hasCapability(user: ChannelUser, capability: RoleCapability): Promise<boolean> {
      const caps = roleCapabilities[user.id] ?? ["workspace_read"];
      return Promise.resolve(caps.includes(capability));
    },
  };
}
