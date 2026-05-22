import type { RoleConfigStore, StoredRoleConfig } from "../core/ports.js";
import { REVIEWER_CONFIG, DEVELOPER_CONFIG } from "../agent/claudeSdkAgentRuntime.js";

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
    },
    {
      name: DEVELOPER_CONFIG.name,
      systemPrompt: DEVELOPER_CONFIG.systemPrompt,
      allowedTools: [...DEVELOPER_CONFIG.allowedTools],
      permissionMode: DEVELOPER_CONFIG.permissionMode,
      ...(DEVELOPER_CONFIG.maxTurns !== undefined ? { maxTurns: DEVELOPER_CONFIG.maxTurns } : {}),
    },
  ];

  for (const config of defaults) {
    const existingConfig = existingByName.get(config.name);
    if (existingConfig === undefined) {
      await store.upsert(config);
      continue;
    }

    if (config.name === REVIEWER_CONFIG.name && shouldRaiseMaxTurns(existingConfig, config)) {
      const raisedMaxTurns = config.maxTurns;
      if (raisedMaxTurns === undefined) {
        continue;
      }
      await store.upsert({
        ...existingConfig,
        maxTurns: raisedMaxTurns,
      });
    }
  }
}

function shouldRaiseMaxTurns(existing: StoredRoleConfig, nextDefault: StoredRoleConfig): boolean {
  if (nextDefault.maxTurns === undefined) {
    return false;
  }

  return existing.maxTurns === undefined || existing.maxTurns < nextDefault.maxTurns;
}
