# 配置文件
# - API密钥等敏感信息从环境变量读取（通过 config.yaml 指定环境变量名）
# - 其他配置从 config/config.yaml 读取

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env 文件
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)


def load_yaml_config() -> dict[str, Any]:
    """加载 config/config.yaml 文件"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "config/config.yaml 不存在，请从 config-example.yaml 复制并重命名为 pets_agent/config/config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


yaml_config = load_yaml_config()

# LLM 配置
llm_config: dict[str, Any] = yaml_config["llm"]
LLM_PROVIDER: str = str(llm_config["provider"])
LLM_PROVIDERS: dict[str, dict[str, Any]] = llm_config["providers"]
LLM_MAX_RETRIES: int = int(llm_config["max_retries"])
LLM_RETRY_DELAY: float = float(llm_config["retry_delay"])


# 获取当前使用提供商的配置
def get_current_llm_config() -> dict[str, Any]:
    """获取当前 LLM 提供商的配置"""
    if LLM_PROVIDER not in LLM_PROVIDERS:
        raise ValueError(f"未配置的 LLM 提供商: {LLM_PROVIDER}")
    return LLM_PROVIDERS[LLM_PROVIDER]


def get_llm_api_key() -> str:
    """从环境变量获取当前提供商的 API Key"""
    provider_config = get_current_llm_config()
    env_var_name: str = str(provider_config.get("api_key_env", "LLM_API_KEY"))
    api_key = os.getenv(env_var_name, "")
    if not api_key:
        raise ValueError(f"未设置环境变量: {env_var_name}")
    return api_key


# 知识库配置 (从 config.yaml)
learning_config: dict[str, Any] = yaml_config["learning"]
LEARNING_FILE = PROJECT_ROOT / str(learning_config["file"])
