export type StartupBannerOptions = {
  readonly serverUrl: string;
  readonly devRoutesEnabled: boolean;
  readonly agentSdk: StartupModelSummary;
  readonly intentLlm: StartupModelSummary;
  readonly runtimes: readonly StartupRuntimeSummary[];
  readonly wechat: StartupWechatSummary;
  readonly cron: StartupCronSummary;
  readonly paths: StartupPathSummary;
};

export type StartupModelSummary = {
  readonly type?: string;
  readonly modelId: string;
  readonly baseUrl: string;
};

export type StartupRuntimeSummary = {
  readonly role: string;
  readonly runtimeName: string;
};

export type StartupWechatSummary = {
  readonly status: "connecting" | "disabled";
  readonly wsUrl: string;
};

export type StartupCronSummary = {
  readonly enabled: boolean;
  readonly tickIntervalMs?: number;
  readonly staleGraceMs?: number;
  readonly leaderLeaseTtlMs?: number;
  readonly deliveryMode?: "app-message" | "smart-bot-fallback" | "disabled";
};

export type StartupPathSummary = {
  readonly knowledgeBasePath: string;
  readonly conversationLogPath: string;
  readonly llmRawLogPath: string;
  readonly systemLogPath: string;
  readonly databasePath: string;
  readonly sessionStorePath: string;
  readonly historyStorePath: string;
  readonly cronJobStorePath?: string;
  readonly cronLeaderLeasePath?: string;
};

const DEFAULT_WECOM_WSS_URL = "wss://openws.work.weixin.qq.com";

export function formatStartupBanner(options: StartupBannerOptions): string {
  return [
    "pets-agent startup",
    `- server: ${options.serverUrl} (devRoutes=${formatEnabled(options.devRoutesEnabled)})`,
    "- health: /healthz /readyz",
    `- agent sdk: ${formatModelSummary(options.agentSdk)}`,
    `- intent llm: ${formatModelSummary(options.intentLlm)}`,
    `- runtimes: ${options.runtimes.map(formatRuntimeSummary).join(", ")}`,
    `- wechat wss: ${options.wechat.status} (${options.wechat.wsUrl || DEFAULT_WECOM_WSS_URL})`,
    `- cron: ${formatCronSummary(options.cron)}`,
    `- knowledge base: ${options.paths.knowledgeBasePath}`,
    [
      "- logs:",
      `conversation=${options.paths.conversationLogPath}`,
      `llmRaw=${options.paths.llmRawLogPath}`,
      `system=${options.paths.systemLogPath}`
    ].join(" "),
    [
      "- state:",
      `db=${options.paths.databasePath}`,
      `sessions=${options.paths.sessionStorePath}`,
      `history=${options.paths.historyStorePath}`,
      ...(options.paths.cronJobStorePath !== undefined
        ? [`cronJobs=${options.paths.cronJobStorePath}`]
        : []),
      ...(options.paths.cronLeaderLeasePath !== undefined
        ? [`cronLeader=${options.paths.cronLeaderLeasePath}`]
        : [])
    ].join(" ")
  ].join("\n");
}

function formatEnabled(value: boolean): "on" | "off" {
  return value ? "on" : "off";
}

function formatModelSummary(summary: StartupModelSummary): string {
  const prefix = summary.type === undefined ? "" : `${summary.type} `;
  return `${prefix}${summary.modelId} at ${summary.baseUrl}`;
}

function formatRuntimeSummary(summary: StartupRuntimeSummary): string {
  return `${summary.role}=${summary.runtimeName}`;
}

function formatCronSummary(summary: StartupCronSummary): string {
  if (!summary.enabled) {
    return "disabled";
  }

  const delivery = summary.deliveryMode ?? "disabled";
  const leaderLease =
    summary.leaderLeaseTtlMs === undefined ? "" : `, leaderLeaseTtl=${summary.leaderLeaseTtlMs}ms`;
  return `enabled (tick=${summary.tickIntervalMs}ms, grace=${summary.staleGraceMs}ms${leaderLease}, delivery=${delivery})`;
}
