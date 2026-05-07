import * as fs from "fs";
import * as path from "path";
import * as yaml from "js-yaml";

export interface LLMProviderConfig {
  model_id: string;
  base_url: string;
  api_key_env: string;
}

export interface Config {
  llm: {
    provider: string;
    providers: Record<string, LLMProviderConfig>;
    max_retries: number;
    retry_delay: number;
  };
  learning?: {
    file: string;
  };
  skills?: {
    dirs: string[];
  };
}

let cachedConfig: Config | null = null;
let configLoadError: Error | null = null;

export function loadConfig(): Config {
  if (configLoadError) throw configLoadError;
  if (cachedConfig) return cachedConfig;

  const configPath = path.join(process.cwd(), "config/config.yaml");
  try {
    const configYaml = fs.readFileSync(configPath, "utf8");
    cachedConfig = yaml.load(configYaml) as Config;
    return cachedConfig;
  } catch (err) {
    configLoadError = err instanceof Error ? err : new Error(String(err));
    throw configLoadError;
  }
}

export function getProvider(): LLMProviderConfig {
  const config = loadConfig();
  const provider = config.llm.providers[config.llm.provider];
  if (!provider) {
    throw new Error(`Provider "${config.llm.provider}" not found in config`);
  }
  return provider;
}

export function getApiKey(): string {
  const provider = getProvider();
  const key = process.env[provider.api_key_env];
  if (!key) {
    throw new Error(`Environment variable ${provider.api_key_env} is not set. Please configure your API key.`);
  }
  return key;
}

export function validateConfig(): void {
  const config = loadConfig();
  if (!config.llm?.provider) {
    throw new Error("Config error: llm.provider is required");
  }
  if (!config.llm?.providers?.[config.llm.provider]) {
    throw new Error(`Config error: provider "${config.llm.provider}" not found in llm.providers`);
  }
  // Validate API key is present
  try {
    getApiKey();
  } catch (e) {
    console.warn("Warning:", (e as Error).message);
  }
}
