import { buildPiModel, type ResolvedLlmConfig } from "../config/llmConfig.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";
import type { AgentRuntime, AgentRuntimeFactory, RoleConfigStore } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claudeSdkAgentRuntime.js";
import { EchoAgentRuntime } from "./echoAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";

export async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig | undefined,
): Promise<Record<string, AgentRuntime>> {
  const roleConfigs = await roleConfigStore.getAll();
  const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig);

  if (roleConfigs.length > 0 && resolvedLlmConfig !== undefined) {
    return Object.fromEntries(
      roleConfigs.map((config) => [
        config.name,
        new ClaudeSdkAgentRuntime({
          roleConfig: config,
          rawLogger: llmRawLogger,
          model: config.model ?? resolvedLlmConfig.modelId,
          ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {}),
        }),
      ])
    );
  }

  if (resolvedLlmConfig !== undefined) {
    return Object.fromEntries(
      DEFAULT_ROLE_CONFIGS.map((config) => [
        config.name,
        new ClaudeSdkAgentRuntime({
          roleConfig: config,
          rawLogger: llmRawLogger,
          model: config.model ?? resolvedLlmConfig.modelId,
          ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {}),
        }),
      ])
    );
  }

  console.warn("using echo runtime because real LLM runtime is not configured");
  return {
    reviewer: new EchoAgentRuntime(),
    developer: new EchoAgentRuntime(),
    admin: new EchoAgentRuntime(),
  };
}

export function createAgentRuntimeFactory(
  llmRawLogger: JsonlLogger,
  roleConfigStore: RoleConfigStore,
  resolvedLlmConfig: ResolvedLlmConfig | undefined,
): AgentRuntimeFactory {
  return {
    async cacheKeyForRole(role: string): Promise<string | undefined> {
      if (resolvedLlmConfig === undefined) return role;
      const config = await roleConfigStore.getByName(role);
      return config === undefined ? undefined : `${role}:${config.updatedAt ?? "unknown"}`;
    },
    async createRuntime(role: string): Promise<AgentRuntime | undefined> {
      if (resolvedLlmConfig === undefined) return undefined;
      const config = await roleConfigStore.getByName(role);
      if (config === undefined) return undefined;
      const toolPermissionDecider = createToolPermissionDecider(resolvedLlmConfig);
      return new ClaudeSdkAgentRuntime({
        roleConfig: config,
        rawLogger: llmRawLogger,
        model: config.model ?? resolvedLlmConfig.modelId,
        ...(toolPermissionDecider !== undefined ? { toolPermissionDecider } : {}),
      });
    },
  };
}

function createToolPermissionDecider(resolvedLlmConfig: ResolvedLlmConfig | undefined) {
  if (resolvedLlmConfig === undefined) {
    return undefined;
  }
  const permissionModel = buildPiModel(resolvedLlmConfig);
  return new LlmBashPermissionDecider(permissionModel, resolvedLlmConfig.apiKey).decide;
}
