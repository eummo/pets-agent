import type { RoleCapability, RoleConfigStore, StoredRoleConfig } from "../auth/index.js";
import { DEFAULT_ROLE_CONFIGS } from "../core/defaultRoles.js";

export async function seedDefaultRoles(store: RoleConfigStore): Promise<void> {
  const existing = await store.getAll();
  const existingByName = new Map(existing.map((config) => [config.name, config]));

  for (const config of DEFAULT_ROLE_CONFIGS) {
    const existingConfig = existingByName.get(config.name);
    if (existingConfig === undefined) {
      await store.upsert(config);
      continue;
    }

    const missingCapabilities = missingDefaultCapabilities(existingConfig, config);
    const needsReviewerFix =
      config.name === "reviewer" && shouldRaiseReviewerRuntimeDefaults(existingConfig, config);
    const defaultMetadataOverrides = missingDefaultMetadata(existingConfig, config);

    if (
      !needsReviewerFix &&
      missingCapabilities.length === 0 &&
      Object.keys(defaultMetadataOverrides).length === 0
    ) {
      continue;
    }

    const reviewerOverrides = needsReviewerFix
      ? reviewerRuntimeDefaultsToRaise(existingConfig, config)
      : {};
    const capabilityOverrides =
      missingCapabilities.length > 0
        ? { capabilities: [...(existingConfig.capabilities ?? []), ...missingCapabilities] }
        : {};

    const updated: StoredRoleConfig = {
      ...existingConfig,
      ...reviewerOverrides,
      ...capabilityOverrides,
      ...defaultMetadataOverrides
    };
    await store.upsert(updated);
  }
}

type ReviewerOverrides = {
  readonly allowedTools: readonly string[];
  readonly permissionMode: StoredRoleConfig["permissionMode"];
  readonly maxTurns?: number;
};

function reviewerRuntimeDefaultsToRaise(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig
): ReviewerOverrides {
  const maxTurns = maxTurnsToRaise(existing, nextDefault);

  return {
    allowedTools: mergeTools(existing.allowedTools, nextDefault.allowedTools),
    permissionMode:
      existing.permissionMode === "dontAsk" ? nextDefault.permissionMode : existing.permissionMode,
    ...(maxTurns !== undefined ? { maxTurns } : {})
  };
}

function shouldRaiseReviewerRuntimeDefaults(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig
): boolean {
  return (
    maxTurnsToRaise(existing, nextDefault) !== existing.maxTurns ||
    existing.permissionMode === "dontAsk" ||
    !existing.allowedTools.includes("Bash")
  );
}

function missingDefaultCapabilities(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig
): RoleCapability[] {
  const existingCapabilities = new Set<RoleCapability>(existing.capabilities ?? []);
  return (nextDefault.capabilities ?? []).filter(
    (c): c is RoleCapability => !existingCapabilities.has(c)
  );
}

function missingDefaultMetadata(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig
): Partial<Pick<StoredRoleConfig, "enableWorkflows" | "planModeInstructions">> {
  return {
    ...(existing.enableWorkflows === undefined && nextDefault.enableWorkflows !== undefined
      ? { enableWorkflows: nextDefault.enableWorkflows }
      : {}),
    ...(existing.planModeInstructions === undefined &&
    nextDefault.planModeInstructions !== undefined
      ? { planModeInstructions: nextDefault.planModeInstructions }
      : {})
  };
}

function maxTurnsToRaise(
  existing: StoredRoleConfig,
  nextDefault: StoredRoleConfig
): number | undefined {
  if (
    nextDefault.maxTurns === undefined ||
    (existing.maxTurns !== undefined && existing.maxTurns >= nextDefault.maxTurns)
  ) {
    return existing.maxTurns;
  }

  return nextDefault.maxTurns;
}

function mergeTools(left: readonly string[], right: readonly string[]): readonly string[] {
  return [...new Set([...left, ...right])];
}
