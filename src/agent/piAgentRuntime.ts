import { createAgentSession, DefaultResourceLoader, SessionManager, AuthStorage, type AgentSession, type AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import type { Model } from "@earendil-works/pi-ai";
import type { AgentRequest, AgentResponse, AgentRuntime, StoredRoleConfig } from "../core/contracts.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { buildWorkspacePrompt } from "./workspacePromptBuilder.js";
import {
  availableToolsForRole,
  roleCanUseFileMutationTools,
  type ToolPermissionDecider,
} from "./toolPolicy.js";
import { PiEventCollector } from "./piEventCollector.js";

// ── Runtime ──────────────────────────────────────────────────────────────────

export type PiAgentRuntimeOptions = {
  readonly roleConfig: StoredRoleConfig;
  readonly model: Model<"anthropic-messages">;
  readonly apiKey: string;
  readonly contextConfig?: ContextConfig | undefined;
  readonly rawLogger?: JsonlLogger;
  readonly toolPermissionDecider?: ToolPermissionDecider;
  readonly agentDir?: string | undefined;
};

const TOOL_NAME_MAP: Readonly<Record<string, string>> = {
  "Read": "read",
  "Bash": "bash",
  "Edit": "edit",
  "Write": "write",
  "Glob": "find",
  "Grep": "grep",
};

export class PiAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly piModel: Model<"anthropic-messages">;
  private readonly apiKey: string;
  private readonly contextConfig: ContextConfig;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly toolPermissionDecider: ToolPermissionDecider | undefined;
  private readonly agentDir: string | undefined;

  private readonly sessionCache = new Map<string, AgentSession>();

  private static readonly DEFAULT_CONTEXT_CONFIG: ContextConfig = {
    autoCompactEnabled: true,
    autoCompactWindow: 150_000,
    workspaceMaxChars: 8_000,
    historyMaxMessages: 20,
  };

  private static readonly DEFAULT_AGENT_DIR = "~/.pi/agent";

  public constructor(options: PiAgentRuntimeOptions) {
    this.name = `pi-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.piModel = options.model;
    this.apiKey = options.apiKey;
    this.contextConfig = options.contextConfig ?? PiAgentRuntime.DEFAULT_CONTEXT_CONFIG;
    this.rawLogger = options.rawLogger;
    this.toolPermissionDecider = options.toolPermissionDecider;
    this.agentDir = options.agentDir;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const prompt = await buildWorkspacePrompt(request, this.contextConfig.workspaceMaxChars);

    const sessionId = request.sessionId ?? generateSessionId();
    const session = await this.getOrCreateSession(request, sessionId);

    const collector = new PiEventCollector(request, this.rawLogger, this.name, this.roleConfig);
    collector.setSessionId(sessionId);

    // Log the request
    void this.rawLogger?.write({
      type: "llm.request",
      operation: "agent_runtime",
      runtime: this.name,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      sessionId,
      prompt,
      turn: 0,
    });

    const unsubscribe = session.subscribe((event: AgentSessionEvent) => {
      collector.onEvent(event);
    });

    try {
      await session.prompt(prompt);
    } catch (error) {
      void this.rawLogger?.write({
        type: "llm.error",
        operation: "agent_runtime",
        runtime: this.name,
        userId: request.user.id,
        workspacePath: request.workspacePath,
        sessionId,
        error: formatUnknownError(error),
        durationMs: Date.now(),
      });
      throw error;
    } finally {
      unsubscribe();
    }

    request.stream?.({ type: "completed", sessionId });

    return collector.toResponse(sessionId);
  }

  public disposeSession(sessionId: string): Promise<void> {
    const session = this.sessionCache.get(sessionId);
    if (session !== undefined) {
      session.dispose();
      this.sessionCache.delete(sessionId);
    }
    return Promise.resolve();
  }

  // ── Session management ─────────────────────────────────────────────────

  private async getOrCreateSession(request: AgentRequest, sessionId: string): Promise<AgentSession> {
    const existing = this.sessionCache.get(sessionId);
    if (existing !== undefined) {
      return existing;
    }

    const session = await this.createSession(request);
    this.sessionCache.set(sessionId, session);
    return session;
  }

  private async createSession(request: AgentRequest): Promise<AgentSession> {
    const resourceLoader = new DefaultResourceLoader({
      cwd: request.workspacePath,
      agentDir: this.agentDir ?? PiAgentRuntime.DEFAULT_AGENT_DIR,
      systemPrompt: this.roleConfig.systemPrompt,
      noExtensions: true,
      noThemes: true,
    });
    await resourceLoader.reload();

    const toolNames = piToolsForRole(this.roleConfig)
      .map((tool) => TOOL_NAME_MAP[tool] ?? tool)
      .filter((name) => name.length > 0);

    // Set up auth storage with the API key for our model's provider
    const authStorage = AuthStorage.inMemory();
    authStorage.setRuntimeApiKey(this.piModel.provider, this.apiKey);

    const sessionOptions: Parameters<typeof createAgentSession>[0] = {
      cwd: request.workspacePath,
      model: this.piModel,
      resourceLoader,
      sessionManager: SessionManager.inMemory(),
      authStorage,
    };
    sessionOptions.tools = toolNames;

    const { session } = await createAgentSession(sessionOptions);

    return session;
  }
}

// ── Helper functions ─────────────────────────────────────────────────────────

function generateSessionId(): string {
  return `pi-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function piToolsForRole(roleConfig: StoredRoleConfig): readonly string[] {
  const tools = availableToolsForRole(roleConfig);

  if (roleCanUseFileMutationTools(roleConfig)) {
    return tools;
  }

  return tools.filter((tool) => tool !== "Bash");
}

// ── Exported for testing ──────────────────────────────────────────────────────

export type { PiAgentRuntimeOptions as _PiAgentRuntimeOptions };
export { piToolsForRole as _piToolsForRole };
