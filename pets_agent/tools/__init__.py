# 将所有工具函数放入一个字典，方便后续调用
from pets_agent.tools.fetch_weather import get_weather
from pets_agent.tools.search_attraction import get_attraction

available_tools = {"get_weather": get_weather,"get_attraction": get_attraction,}