import type { RoleConfigStore, StoredRoleConfig } from "../core/ports.js";
import { REVIEWER_CONFIG, DEVELOPER_CONFIG } from "../agent/claudeSdkAgentRuntime.js";

export async function seedDefaultRoles(store: RoleConfigStore): Promise<void> {
  const existing = await store.getAll();
  if (existing.length > 0) return;

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
    await store.upsert(config);
  }
}
