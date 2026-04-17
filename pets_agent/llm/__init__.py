from pets_agent.llm.base import LLMClientBase
from pets_agent.llm.clients import OpenAIClient, AnthropicClient

__all__ = ["LLMClientBase", "OpenAIClient", "AnthropicClient", "create_llm_client"]


def create_llm_client(client_type: str, **kwargs):
    """
    LLM 客户端工厂函数

    Args:
        client_type: 客户端类型 ("openai" 或 "anthropic")
        **kwargs: 传递给客户端的参数

    Returns:
        LLMClientBase 实例
    """
    if client_type == "openai":
        return OpenAIClient(**kwargs)
    elif client_type == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"不支持的客户端类型: {client_type}")
