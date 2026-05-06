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

export function loadConfig(): Config {
  if (cachedConfig) return cachedConfig;

  const configPath = path.join(process.cwd(), "config/config.yaml");
  const configYaml = fs.readFileSync(configPath, "utf8");
  cachedConfig = yaml.load(configYaml) as Config;
  return cachedConfig;
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
  return process.env[provider.api_key_env] ?? "";
}
