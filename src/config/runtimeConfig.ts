import path from "node:path";

export type RuntimeConfig = {
  readonly port: number;
  readonly knowledgeBasePath: string;
  readonly wechatToken: string;
  readonly logDir: string;
  readonly sessionStorePath: string;
  readonly historyStorePath: string;
  readonly dbPath: string;
  readonly enableDevRoutes: boolean;
};

export function loadRuntimeConfig(env: NodeJS.ProcessEnv = process.env): RuntimeConfig {
  return {
    port: Number.parseInt(env["PORT"] ?? "3000", 10),
    knowledgeBasePath: env["KNOWLEDGE_BASE_PATH"] ?? ".harness/knowledge-base",
    wechatToken: env["WECHAT_TOKEN"] ?? "dev-token",
    logDir: env["LOG_DIR"] ?? path.resolve(".harness", "logs"),
    sessionStorePath: env["SESSION_STORE_PATH"] ?? path.resolve(".harness", "state", "sessions.json"),
    historyStorePath: env["HISTORY_STORE_PATH"] ?? path.resolve(".harness", "state", "history.json"),
    dbPath: env["DB_PATH"] ?? path.resolve(".harness", "state", "agent.db"),
    enableDevRoutes: env["NODE_ENV"] !== "production",
  };
}
