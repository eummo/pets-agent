import type { RoleConfigStore, StoredRoleConfig } from "../auth/index.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";

export async function seedDefaultRoles(store: RoleConfigStore): Promise<void> {
  const existing = await store.getAll();
  const existingByName = new Map(existing.map((config) => [config.name, config]));

  const defaults = DEFAULT_ROLE_CONFIGS;

  for (const config of defaults) {
    const existingConfig = existingByName.get(config.name);
    if (existingConfig === undefined) {
      await store.upsert(config);
      continue;
    }

    const missingCapabilities = missingDefaultCapabilities(existingConfig, config);
    if ((config.name === "reviewer" && shouldRaiseReviewerRuntimeDefaults(existingConfig, config)) || missingCapabilities.length > 0) {
      await store.upsert({
        ...existingConfig,
        ...(config.name === "reviewer" ? reviewerRuntimeDefaultsToRaise(existingConfig, config) : {}),
        ...(missingCapabilities.length > 0 ? { capabilities: [...(existingConfig.capabilities ?? []), ...missingCapabilities] } : {}),
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

function shouldRaiseReviewerRuntimeDefaults(existing: StoredRoleConfig, nextDefault: StoredRoleConfig): boolean {
  return maxTurnsToRaise(existing, nextDefault) !== existing.maxTurns
    || existing.permissionMode === "dontAsk"
    || !existing.allowedTools.includes("Bash");
}

function missingDefaultCapabilities(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig,
): readonly NonNullable<StoredRoleConfig["capabilities"]>[number][] {
  if (nextDefault.capabilities === undefined || nextDefault.capabilities.length === 0) {
    return [];
  }

  const existingCapabilities = new Set(existing.capabilities ?? []);
  return nextDefault.capabilities.filter((capability) => !existingCapabilities.has(capability));
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

