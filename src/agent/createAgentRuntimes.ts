import { buildPiModel, type ResolvedLlmConfig, type ResolvedAgentSdkConfig } from "../config/llmConfig.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";
import type { AgentRuntime, AgentRuntimeFactory } from "./index.js";
import type { RoleConfigStore, StoredRoleConfig } from "../auth/index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claudeSdkAgentRuntime.js";
import { CodebuddySdkAgentRuntime } from "./codebuddySdkAgentRuntime.js";
import { PiAgentRuntime } from "./piAgentRuntime.js";
import { IntentAgentRuntime } from "./intentAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";
import { LlmIntentDetectionService } from "../intent/llmIntentDetectionService.js";
import type { ToolPermissionDecider } from "./toolPolicy.js";

export async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
): Promise<Record<string, AgentRuntime>> {
  const roleConfigs = await roleConfigStore.getAll();
  const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig, llmRawLogger);

  const configs = roleConfigs.length > 0 ? roleConfigs : DEFAULT_ROLE_CONFIGS;

  return Object.fromEntries(
    configs.map((config) => [
      config.name,
      createRuntimeForSdkType(resolvedAgentSdkConfig, config, contextConfig, llmRawLogger, resolvedLlmConfig, toolPermissionDecider),
    ])
  );
}

export function createAgentRuntimeFactory(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
): AgentRuntimeFactory {
  let warmupCache: Record<string, AgentRuntime> | undefined;

  return {
    async warmup(): Promise<Record<string, AgentRuntime>> {
      if (warmupCache !== undefined) return warmupCache;
      warmupCache = await createAgentRuntimes(llmRawLogger, roleConfigStore, resolvedLlmConfig, resolvedAgentSdkConfig, contextConfig);
      return warmupCache;
    },
    async cacheKeyForRole(role: string): Promise<string | undefined> {
      if (role === "intent") return "intent";
      const config = await roleConfigStore.getByName(role);
      return config === undefined ? undefined : `${role}:${config.updatedAt ?? "unknown"}`;
    },
    async createRuntime(role: string): Promise<AgentRuntime | undefined> {
      if (role === "intent") {
        return createIntentRuntime(resolvedLlmConfig, llmRawLogger);
      }
      const config = await roleConfigStore.getByName(role);
      if (config === undefined) return undefined;
      const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig, llmRawLogger);
      return createRuntimeForSdkType(resolvedAgentSdkConfig, config, contextConfig, llmRawLogger, resolvedLlmConfig, toolPermissionDecider);
    },
  };
}

export function setupAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  resolvedAgentSdkConfig: ResolvedAgentSdkConfig,
  contextConfig?: ContextConfig,
): AgentRuntimeFactory {
  return createAgentRuntimeFactory(llmRawLogger, roleConfigStore, resolvedLlmConfig, resolvedAgentSdkConfig, contextConfig);
}

function createRuntimeForSdkType(
  agentSdkConfig: ResolvedAgentSdkConfig,
  roleConfig: StoredRoleConfig,
  contextConfig: ContextConfig | undefined,
  llmRawLogger: JsonlLogger,
  resolvedLlmConfig: ResolvedLlmConfig,
  toolPermissionDecider: ToolPermissionDecider,
): AgentRuntime {
  switch (agentSdkConfig.type) {
    case "claude":
      return new ClaudeSdkAgentRuntime({
        roleConfig,
        contextConfig,
        rawLogger: llmRawLogger,
        model: roleConfig.model ?? agentSdkConfig.modelId,
        toolPermissionDecider,
      });
    case "codebuddy":
      return new CodebuddySdkAgentRuntime({
        roleConfig,
        agentSdkConfig,
        contextConfig,
        rawLogger: llmRawLogger,
        model: roleConfig.model ?? agentSdkConfig.modelId,
        toolPermissionDecider,
      });
    case "pi": {
      return new PiAgentRuntime({
        roleConfig,
        agentSdkConfig,
        maxTokens: resolvedLlmConfig.maxTokens,
        contextConfig,
        rawLogger: llmRawLogger,
        toolPermissionDecider,
      });
    }
  }
}

function createToolPermissionDecider(resolvedLlmConfig: ResolvedLlmConfig, llmRawLogger: JsonlLogger) {
  const permissionModel = buildPiModel(resolvedLlmConfig);
  return new LlmBashPermissionDecider(permissionModel, resolvedLlmConfig.apiKey, llmRawLogger).decide;
}

function createIntentRuntime(resolvedLlmConfig: ResolvedLlmConfig, llmRawLogger: JsonlLogger): IntentAgentRuntime {
  const intentModel = buildPiModel(resolvedLlmConfig);
  const detector = new LlmIntentDetectionService(intentModel, resolvedLlmConfig.apiKey, llmRawLogger);
  return new IntentAgentRuntime(detector);
}
