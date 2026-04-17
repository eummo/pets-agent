import anthropic
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError, RateLimitError, AuthenticationError

from pets_agent.llm.base import LLMClientBase
from pets_agent.logger import logger


class AnthropicClient(LLMClientBase):
    """Anthropic Claude API 客户端"""

    def __init__(self, model: str, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Anthropic 客户端初始化: model={model}")

    @property
    def model(self) -> str:
        return self._model

    def _call_llm(self, messages: list[dict]) -> str:
        """实际调用 Anthropic API"""
        # 从 messages 中提取最后一个 user 消息
        user_content = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                user_content = msg['content']
                break

        system_prompt = None
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
                break

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content}
            ]
        )
        return response.content[0].text

    def _validate_response(self, response: str) -> bool:
        """验证响应是否有效"""
        if not response:
            return False
        if response.startswith("错误:"):
            return False
        return True

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 Anthropic API 生成响应"""
        print("正在调用大语言模型...")
        logger.info("开始调用 Anthropic LLM API")

        try:
            return super().generate(prompt, system_prompt)

        except AuthenticationError as e:
            error_msg = f"Anthropic 认证失败: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except RateLimitError as e:
            error_msg = f"Anthropic 请求频率超限: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except APITimeoutError as e:
            error_msg = f"Anthropic 请求超时: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except AnthropicAPIError as e:
            error_msg = f"Anthropic API 错误: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"

        except Exception as e:
            error_msg = f"调用 Anthropic API 时发生未知错误: {type(e).__name__}: {e}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return f"错误: {error_msg}"
