import type { RoleConfigStore, StoredRoleConfig } from "../core/contracts.js";
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

    if (config.name === "reviewer" && shouldRaiseMaxTurns(existingConfig, config)) {
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

