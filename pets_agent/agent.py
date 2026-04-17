import re

from pets_agent.config import (
    LLM_PROVIDER,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    get_current_llm_config,
    get_llm_api_key,
)
from pets_agent.llm import create_llm_client
from pets_agent.logger import logger
from pets_agent.tools import available_tools


def format_final_answer(text: str) -> str:
    """格式化最终答案，移除思考过程等干扰内容"""
    # 移除 think 标签内容
    text = re.sub(r'<think>.*?</think>', '\n', text, flags=re.DOTALL)
    # 移除 Thought/Action 标记及其内容
    text = re.sub(r'Thought:.*?(?=\nAction:|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'Action:.*?(?=\nThought:|\nObservation:|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'Observation:.*?(?=\nThought:|\Z)', '', text, flags=re.DOTALL)
    # 移除 finish 标记
    text = re.sub(r'finish\(answer="(.*?)"\)', r'\1', text, flags=re.DOTALL)
    # 清理多余空白
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""

llm_config = get_current_llm_config()
llm_kwargs = {
    "model": llm_config["model_id"],
    "api_key": get_llm_api_key(),
    "max_retries": LLM_MAX_RETRIES,
    "retry_delay": LLM_RETRY_DELAY,
}
# OpenAI 需要 base_url，Anthropic 不需要
if "base_url" in llm_config:
    llm_kwargs["base_url"] = llm_config["base_url"]

llm = create_llm_client(
    client_type=LLM_PROVIDER,
    **llm_kwargs,
)

user_prompt = "你好，请帮我查询一下今天深圳的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "=" * 40)
logger.info(f"用户输入: {user_prompt}")

for i in range(5):
    print(f"--- 循环 {i+1} ---\n")
    logger.info(f"=== 循环 {i+1} ===")

    full_prompt = "\n".join(prompt_history)

    logger.info("调用 LLM")
    logger.info("========== LLM 输入 ==========\n" + full_prompt + "\n==============================")
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    logger.info("========== LLM 输出 ==========\n" + llm_output + "\n==============================")

    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
            logger.info("已截断多余的 Thought-Action 对")

    print(f"模型输出:\n{llm_output}\n")

    # 如果 LLM 返回错误，直接结束
    if llm_output.startswith("错误:"):
        logger.error(f"LLM 调用失败，结束执行: {llm_output}")
        print(f"LLM 调用失败，结束执行。")
        break

    prompt_history.append(llm_output)

    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        # 如果 LLM 直接输出了答案（没有 Action 格式），尝试提取并结束
        if "Observation:" not in llm_output and ("。" in llm_output or "答案" in llm_output):
            answer = format_final_answer(llm_output)
            print(f"\n{'='*40}\n{answer}\n{'='*40}")
            logger.info(f"任务完成，最终答案:\n{answer}")
            break
        print("解析错误: 模型输出中未找到 Action。")
        logger.error("解析错误: 模型输出中未找到 Action")
        break

    action_str = action_match.group(1).strip()
    if action_str.startswith("finish"):
        final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
        answer = format_final_answer(final_answer)
        print(f"\n{'='*40}\n任务完成！\n\n{answer}\n{'='*40}")
        logger.info(f"任务完成，最终答案:\n{answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    logger.info(f"调用工具: {tool_name}, 参数: {kwargs}")

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
        logger.info(f"工具 {tool_name} 返回:\n{observation}")
    else:
        observation = f"错误: 未定义的工具 '{tool_name}'"
        logger.error(f"未定义的工具: {tool_name}")

    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
