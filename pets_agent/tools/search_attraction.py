import os

from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    api_key = os.environ.get("TAVILY_API_KEY") or ""
    if not api_key:
        return "错误:未配置 TAVILY_API_KEY 环境变量，请检查 .env 文件。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        results = response.get("results", [])
        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        formatted = "\n".join(f"- {r['title']}: {r['content']}" for r in results)
        return f"根据搜索，为您找到以下信息:\n{formatted}"

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
