from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError

from pets_agent.llm.base import LLMClientBase
from pets_agent.logger import logger


class OpenAIClient(LLMClientBase):
    """OpenAI 兼容接口的 LLM 客户端"""

    def __init__(self, model: str, api_key: str, base_url: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAI 客户端初始化: model={model}, base_url={base_url}")

    @property
    def model(self) -> str:
        return self._model

    def _call_llm(self, messages: list[dict]) -> str:
        """实际调用 OpenAI API"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content

    def _validate_response(self, response: str) -> bool:
        """验证响应是否有效"""
        # 如果响应为空或包含错误标记，认为无效
        if not response:
            return False
        if response.startswith("错误:"):
            return False
        return True

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 OpenAI 兼容 API 生成响应"""
        print("正在调用大语言模型...")
        logger.info("开始调用 LLM API")

        try:
            return super().generate(prompt, system_prompt)

        except AuthenticationError as e:
            error_msg = f"认证失败: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except RateLimitError as e:
            error_msg = f"请求频率超限: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except APITimeoutError as e:
            error_msg = f"请求超时: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except APIError as e:
            error_msg = f"API 错误: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except Exception as e:
            error_msg = f"调用LLM API时发生未知错误: {type(e).__name__}: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"
