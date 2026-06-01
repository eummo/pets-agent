import {
  createAgentSession,
  DefaultResourceLoader,
  SessionManager,
  AuthStorage,
  type AgentSession,
  type AgentSessionEvent
} from "@earendil-works/pi-coding-agent";
import type { ResolvedAgentSdkConfig } from "../../config/llmConfig.js";
import { AGENT_SDK_DEFAULTS } from "../../config/llmConfig.js";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import type { ContextConfig } from "../../config/runtimeConfig.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import { buildWorkspacePrompt } from "../shared/workspacePromptBuilder.js";
import {
  availableToolsForRole,
  roleCanUseFileMutationTools,
  type ToolPermissionDecider
} from "../../auth/index.js";
import { PiEventCollector } from "./piEventCollector.js";
import { formatUnknownError } from "../shared/sdkRuntimeHelpers.js";

// ── Runtime ──────────────────────────────────────────────────────────────────

export type PiAgentRuntimeOptions = {
  readonly roleConfig: StoredRoleConfig;
  readonly agentSdkConfig: ResolvedAgentSdkConfig;
  readonly maxTokens?: number | undefined;
  readonly contextConfig?: ContextConfig | undefined;
  readonly rawLogger?: JsonlLogger;
  readonly toolPermissionDecider?: ToolPermissionDecider;
};

const TOOL_NAME_MAP: Readonly<Record<string, string>> = {
  Read: "read",
  Bash: "bash",
  Edit: "edit",
  Write: "write",
  Glob: "find",
  Grep: "grep"
};

export class PiAgentRuntime implements AgentRuntime {
  public readonly name: string;
  private readonly roleConfig: StoredRoleConfig;
  private readonly agentSdkConfig: ResolvedAgentSdkConfig;
  private readonly maxTokens: number | undefined;
  private readonly contextConfig: ContextConfig;
  private readonly rawLogger: JsonlLogger | undefined;
  private readonly toolPermissionDecider: ToolPermissionDecider | undefined;

  private readonly sessionCache = new Map<string, AgentSession>();

  private static readonly DEFAULT_CONTEXT_CONFIG: ContextConfig = {
    autoCompactEnabled: true,
    autoCompactWindow: 150_000,
    workspaceMaxChars: 8_000,
    historyMaxMessages: 20
  };

  private static readonly DEFAULT_AGENT_DIR = "~/.pi/agent";

  public constructor(options: PiAgentRuntimeOptions) {
    this.name = `pi-${options.roleConfig.name}`;
    this.roleConfig = options.roleConfig;
    this.agentSdkConfig = options.agentSdkConfig;
    this.maxTokens = options.maxTokens;
    this.contextConfig = options.contextConfig ?? PiAgentRuntime.DEFAULT_CONTEXT_CONFIG;
    this.rawLogger = options.rawLogger;
    this.toolPermissionDecider = options.toolPermissionDecider;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    let prompt = await buildWorkspacePrompt(request, this.contextConfig.workspaceMaxChars);

    const sessionId = request.sessionId ?? generateSessionId();
    const session = await this.getOrCreateSession(request, sessionId);

    // When a role switch forces a new session, inject prior conversation
    // history so the new session can continue the conversation seamlessly.
    if (
      request.history !== undefined &&
      request.history.length > 0 &&
      request.sessionId === undefined
    ) {
      const historyLines = request.history
        .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
        .join("\n");
      prompt = [
        "Previous conversation (continued from a different role):",
        historyLines,
        "",
        "Continue the conversation below. The user may refer to earlier messages above.",
        "",
        prompt
      ].join("\n");
    }

    const startTime = Date.now();
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
      turn: 0
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
        durationMs: Date.now() - startTime
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

  private async getOrCreateSession(
    request: AgentRequest,
    sessionId: string
  ): Promise<AgentSession> {
    const existing = this.sessionCache.get(sessionId);
    if (existing !== undefined) {
      return existing;
    }

    const session = await this.createSession(request);
    this.sessionCache.set(sessionId, session);
    return session;
  }

  private async createSession(request: AgentRequest): Promise<AgentSession> {
    const cfg = this.agentSdkConfig;
    const model = {
      id: cfg.modelId,
      name: cfg.modelId,
      api: cfg.api ?? AGENT_SDK_DEFAULTS.api,
      provider: cfg.provider ?? AGENT_SDK_DEFAULTS.provider,
      baseUrl: cfg.baseUrl.replace(/\/+$/, ""),
      reasoning: cfg.reasoning ?? AGENT_SDK_DEFAULTS.reasoning,
      input: cfg.input ? [...cfg.input] : [...AGENT_SDK_DEFAULTS.input],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: cfg.contextWindow ?? AGENT_SDK_DEFAULTS.contextWindow,
      maxTokens: this.maxTokens ?? AGENT_SDK_DEFAULTS.maxTokens
    };

    const resourceLoader = new DefaultResourceLoader({
      cwd: request.workspacePath,
      agentDir: cfg.agentDir ?? PiAgentRuntime.DEFAULT_AGENT_DIR,
      systemPrompt: this.roleConfig.systemPrompt,
      noExtensions: true,
      noThemes: true
    });
    await resourceLoader.reload();

    const toolNames = piToolsForRole(this.roleConfig)
      .map((tool) => TOOL_NAME_MAP[tool] ?? tool)
      .filter((name) => name.length > 0);

    const authStorage = AuthStorage.inMemory();
    authStorage.setRuntimeApiKey(model.provider, cfg.apiKey);

    const sessionOptions: Parameters<typeof createAgentSession>[0] = {
      cwd: request.workspacePath,
      model,
      resourceLoader,
      sessionManager: SessionManager.inMemory(),
      authStorage
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
