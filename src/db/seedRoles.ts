import type { RoleConfigStore, StoredRoleConfig } from "../core/ports.js";
import { REVIEWER_CONFIG, DEVELOPER_CONFIG, ADMIN_CONFIG } from "../agent/claudeSdkAgentRuntime.js";

export async function seedDefaultRoles(store: RoleConfigStore): Promise<void> {
  const existing = await store.getAll();
  const existingByName = new Map(existing.map((config) => [config.name, config]));

  const defaults: readonly StoredRoleConfig[] = [
    {
      name: REVIEWER_CONFIG.name,
      systemPrompt: REVIEWER_CONFIG.systemPrompt,
      allowedTools: [...REVIEWER_CONFIG.allowedTools],
      permissionMode: REVIEWER_CONFIG.permissionMode,
      ...(REVIEWER_CONFIG.maxTurns !== undefined ? { maxTurns: REVIEWER_CONFIG.maxTurns } : {}),
      capabilities: ["workspace_read"],
    },
    {
      name: DEVELOPER_CONFIG.name,
      systemPrompt: DEVELOPER_CONFIG.systemPrompt,
      allowedTools: [...DEVELOPER_CONFIG.allowedTools],
      permissionMode: DEVELOPER_CONFIG.permissionMode,
      ...(DEVELOPER_CONFIG.maxTurns !== undefined ? { maxTurns: DEVELOPER_CONFIG.maxTurns } : {}),
      capabilities: ["workspace_read", "workspace_mutate"],
    },
    {
      name: ADMIN_CONFIG.name,
      systemPrompt: ADMIN_CONFIG.systemPrompt,
      allowedTools: [...ADMIN_CONFIG.allowedTools],
      permissionMode: ADMIN_CONFIG.permissionMode,
      ...(ADMIN_CONFIG.maxTurns !== undefined ? { maxTurns: ADMIN_CONFIG.maxTurns } : {}),
      capabilities: ["workspace_read", "workspace_mutate", "feedback_view", "feedback_manage"],
    },
  ];

  for (const config of defaults) {
    const existingConfig = existingByName.get(config.name);
    if (existingConfig === undefined) {
      await store.upsert(config);
      continue;
    }

    if (config.name === REVIEWER_CONFIG.name && shouldRaiseMaxTurns(existingConfig, config)) {
      await store.upsert({
        ...existingConfig,
        ...reviewerRuntimeDefaultsToRaise(existingConfig, config),
      });
    }
  }
}

function reviewerRuntimeDefaultsToRaise(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig,
): Pick<StoredRoleConfig, "allowedTools" | "permissionMode"> & { readonly maxTurns?: number } {
  const maxTurns = maxTurnsToRaise(existing, nextDefault);

  return {
    allowedTools: mergeTools(existing.allowedTools, nextDefault.allowedTools),
    permissionMode: existing.permissionMode === "dontAsk" ? nextDefault.permissionMode : existing.permissionMode,
    ...(maxTurns !== undefined ? { maxTurns } : {}),
  };
}

function shouldRaiseMaxTurns(existing: StoredRoleConfig, nextDefault: StoredRoleConfig): boolean {
  return maxTurnsToRaise(existing, nextDefault) !== existing.maxTurns
    || existing.permissionMode === "dontAsk"
    || !existing.allowedTools.includes("Bash");
}

function maxTurnsToRaise(existing: StoredRoleConfig, nextDefault: StoredRoleConfig): number | undefined {
  if (nextDefault.maxTurns === undefined || existing.maxTurns !== undefined && existing.maxTurns >= nextDefault.maxTurns) {
    return existing.maxTurns;
  }

  return nextDefault.maxTurns;
}

function mergeTools(left: readonly string[], right: readonly string[]): readonly string[] {
  return [...new Set([...left, ...right])];
}
