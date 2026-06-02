import type {
  RoleCapability,
  RoleConfigStore,
  SettingSource,
  StoredRoleConfig
} from "../auth/index.js";
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
): Partial<Pick<StoredRoleConfig, "enableWorkflows" | "planModeInstructions" | "settingSources">> {
  const settingSources = mergeMissingSettingSources(
    existing.settingSources,
    nextDefault.settingSources
  );

  return {
    ...(existing.enableWorkflows === undefined && nextDefault.enableWorkflows !== undefined
      ? { enableWorkflows: nextDefault.enableWorkflows }
      : {}),
    ...(existing.planModeInstructions === undefined &&
    nextDefault.planModeInstructions !== undefined
      ? { planModeInstructions: nextDefault.planModeInstructions }
      : {}),
    ...(settingSources !== undefined ? { settingSources } : {})
  };
}

function mergeMissingSettingSources(
  existing: readonly SettingSource[] | undefined,
  nextDefault: readonly SettingSource[] | undefined
): readonly SettingSource[] | undefined {
  if (nextDefault === undefined) return undefined;
  if (existing === undefined) return nextDefault;

  const existingSources = new Set(existing);
  const missingSources = nextDefault.filter((source) => !existingSources.has(source));
  if (missingSources.length === 0) return undefined;

  return [...existing, ...missingSources];
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
