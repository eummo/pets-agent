import type { ResolvedLlmConfig, ResolvedAgentSdkConfig } from "../config/llmConfig.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import type { AgentRuntime, AgentRuntimeFactory } from "./index.js";
import type { RoleConfigStore, StoredRoleConfig } from "../auth/index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claude/index.js";
import { CodebuddySdkAgentRuntime } from "./codebuddy/index.js";
import { PiAgentRuntime } from "./pi/index.js";
import type { ToolPermissionDecider } from "../auth/index.js";

export async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
  toolPermissionDecider?: ToolPermissionDecider
): Promise<Record<string, AgentRuntime>> {
  const configs = await roleConfigStore.getAll();

  return Object.fromEntries(
    configs.map((config) => [
      config.name,
      createRuntimeForSdkType(
        resolvedAgentSdkConfig,
        config,
        contextConfig,
        llmRawLogger,
        resolvedLlmConfig,
        toolPermissionDecider
      )
    ])
  );
}

export function createAgentRuntimeFactory(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
  toolPermissionDecider?: ToolPermissionDecider
): AgentRuntimeFactory {
  let warmupCache: Record<string, AgentRuntime> | undefined;

  return {
    async warmup(): Promise<Record<string, AgentRuntime>> {
      if (warmupCache !== undefined) return warmupCache;
      warmupCache = await createAgentRuntimes(
        llmRawLogger,
        roleConfigStore,
        resolvedLlmConfig,
        resolvedAgentSdkConfig,
        contextConfig,
        toolPermissionDecider
      );
      return warmupCache;
    },
    async cacheKeyForRole(role: string): Promise<string | undefined> {
      const config = await roleConfigStore.getByName(role);
      return config === undefined ? undefined : `${role}:${config.updatedAt ?? "unknown"}`;
    },
    async createRuntime(role: string): Promise<AgentRuntime | undefined> {
      const config = await roleConfigStore.getByName(role);
      if (config === undefined) return undefined;
      return createRuntimeForSdkType(
        resolvedAgentSdkConfig,
        config,
        contextConfig,
        llmRawLogger,
        resolvedLlmConfig,
        toolPermissionDecider
      );
    }
  };
}

export function setupAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
  toolPermissionDecider?: ToolPermissionDecider
): AgentRuntimeFactory {
  return createAgentRuntimeFactory(
    llmRawLogger,
    roleConfigStore,
    resolvedLlmConfig,
    resolvedAgentSdkConfig,
    contextConfig,
    toolPermissionDecider
  );
}

function createRuntimeForSdkType(
  agentSdkConfig: ResolvedAgentSdkConfig,
  roleConfig: StoredRoleConfig,
  contextConfig: ContextConfig | undefined,
  llmRawLogger: JsonlLogger,
  resolvedLlmConfig: ResolvedLlmConfig,
  toolPermissionDecider: ToolPermissionDecider | undefined
): AgentRuntime {
  switch (agentSdkConfig.type) {
    case "claude":
      return new ClaudeSdkAgentRuntime({
        roleConfig,
        contextConfig,
        rawLogger: llmRawLogger,
        model: roleConfig.model ?? agentSdkConfig.modelId,
        ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {})
      });
    case "codebuddy":
      return new CodebuddySdkAgentRuntime({
        roleConfig,
        agentSdkConfig,
        contextConfig,
        rawLogger: llmRawLogger,
        model: roleConfig.model ?? agentSdkConfig.modelId,
        ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {})
      });
    case "pi": {
      return new PiAgentRuntime({
        roleConfig,
        agentSdkConfig,
        maxTokens: resolvedLlmConfig.maxTokens,
        contextConfig,
        rawLogger: llmRawLogger,
        ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {})
      });
    }
  }
}
