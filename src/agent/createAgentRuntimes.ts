import { buildPiModel, type ResolvedLlmConfig } from "../config/llmConfig.js";
import type { ContextConfig } from "../config/runtimeConfig.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";
import type { AgentRuntime, AgentRuntimeFactory, RoleConfigStore } from "../core/contracts.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claudeSdkAgentRuntime.js";
import { IntentAgentRuntime } from "./intentAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";
import { LlmIntentDetectionService } from "../intent/llmIntentDetectionService.js";

export async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  contextConfig?: ContextConfig,
): Promise<Record<string, AgentRuntime>> {
  const roleConfigs = await roleConfigStore.getAll();
  const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig, llmRawLogger);

  if (roleConfigs.length > 0) {
    return Object.fromEntries(
      roleConfigs.map((config) => [
        config.name,
        new ClaudeSdkAgentRuntime({
          roleConfig: config,
          contextConfig,
          rawLogger: llmRawLogger,
          model: config.model ?? resolvedLlmConfig.modelId,
          toolPermissionDecider,
        }),
      ])
    );
  }

  return Object.fromEntries(
    DEFAULT_ROLE_CONFIGS.map((config) => [
      config.name,
      new ClaudeSdkAgentRuntime({
        roleConfig: config,
        contextConfig,
        rawLogger: llmRawLogger,
        model: config.model ?? resolvedLlmConfig.modelId,
        toolPermissionDecider,
      }),
    ])
  );
}

export function createAgentRuntimeFactory(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  contextConfig?: ContextConfig,
): AgentRuntimeFactory {
  let warmupCache: Record<string, AgentRuntime> | undefined;

  return {
    async warmup(): Promise<Record<string, AgentRuntime>> {
      if (warmupCache !== undefined) return warmupCache;
      warmupCache = await createAgentRuntimes(llmRawLogger, roleConfigStore, resolvedLlmConfig, contextConfig);
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
      return new ClaudeSdkAgentRuntime({
        roleConfig: config,
        contextConfig,
        rawLogger: llmRawLogger,
        model: config.model ?? resolvedLlmConfig.modelId,
        toolPermissionDecider,
      });
    },
  };
}

export function setupAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
  contextConfig?: ContextConfig,
): AgentRuntimeFactory {
  return createAgentRuntimeFactory(llmRawLogger, roleConfigStore, resolvedLlmConfig, contextConfig);
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
