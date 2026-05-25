import { buildPiModel, type ResolvedLlmConfig } from "../config/llmConfig.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";
import type { AgentRuntime, AgentRuntimeFactory, RoleConfigStore } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claudeSdkAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";

export async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig,
): Promise<Record<string, AgentRuntime>> {
  const roleConfigs = await roleConfigStore.getAll();
  const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig);

  if (roleConfigs.length > 0) {
    return Object.fromEntries(
      roleConfigs.map((config) => [
        config.name,
        new ClaudeSdkAgentRuntime({
          roleConfig: config,
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
): AgentRuntimeFactory {
  return {
    async cacheKeyForRole(role: string): Promise<string | undefined> {
      const config = await roleConfigStore.getByName(role);
      return config === undefined ? undefined : `${role}:${config.updatedAt ?? "unknown"}`;
    },
    async createRuntime(role: string): Promise<AgentRuntime | undefined> {
      const config = await roleConfigStore.getByName(role);
      if (config === undefined) return undefined;
      const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig);
      return new ClaudeSdkAgentRuntime({
        roleConfig: config,
        rawLogger: llmRawLogger,
        model: config.model ?? resolvedLlmConfig.modelId,
        toolPermissionDecider,
      });
    },
  };
}

function createToolPermissionDecider(resolvedLlmConfig: ResolvedLlmConfig) {
  const permissionModel = buildPiModel(resolvedLlmConfig);
  return new LlmBashPermissionDecider(permissionModel, resolvedLlmConfig.apiKey).decide;
}
