from abc import ABC, abstractmethod
import time

from pets_agent.logger import logger


class LLMClientBase(ABC):
    """LLM 客户端抽象基类，支持重试机制"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @abstractmethod
    def _call_llm(self, messages: list[dict]) -> str:
        """
        实际的 LLM API 调用，由子类实现

        Args:
            messages: 消息列表

        Returns:
            LLM 生成的响应文本
        """
        pass

    @abstractmethod
    def _validate_response(self, response: str) -> bool:
        """
        验证响应是否有效（用于判断是否需要重试）

        Args:
            response: LLM 响应

        Returns:
            True 表示有效，False 表示需要重试
        """
        pass

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用 LLM 生成响应，支持重试

        Args:
            prompt: 用户输入的提示
            system_prompt: 系统提示词

        Returns:
            LLM 生成的响应文本，错误时返回以 "错误:" 开头的字符串
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]

        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(messages)

                if self._validate_response(response):
                    return response

                # 响应无效（非错误消息），返回
                if not response.startswith("错误:"):
                    return response

                last_error = response
                logger.warning(f"LLM 返回无效响应（第 {attempt + 1} 次尝试）")

            except Exception as e:
                last_error = f"调用 LLM API 时发生异常: {type(e).__name__}: {e}"
                logger.warning(f"{last_error}（第 {attempt + 1} 次尝试）")

            # 重试前等待
            if attempt < self._max_retries - 1:
                logger.info(f"等待 {self._retry_delay} 秒后重试...")
                time.sleep(self._retry_delay)

        # 所有重试都失败
        error_msg = last_error or "LLM 调用失败"
        return f"错误: {error_msg}"

    @property
    @abstractmethod
    def model(self) -> str:
        """返回当前使用的模型名称"""
        pass
