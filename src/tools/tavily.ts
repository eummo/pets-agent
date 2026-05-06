import axios from "axios";

export async function searchAttraction(
  city: string,
  weather: string,
): Promise<string> {
  const tavilyKey = process.env.TAVILY_API_KEY;
  if (!tavilyKey) {
    return "未配置 TAVILY_API_KEY 环境变量，无法搜索景点";
  }
  try {
    const response = await axios.post(
      "https://api.tavily.com/search",
      {
        query: `'${city}' 在'${weather}'天气下最值得去的旅游景点推荐及理由`,
        search_depth: "basic",
        include_answer: true,
      },
      {
        headers: {
          Authorization: `Bearer ${tavilyKey}`,
          "Content-Type": "application/json",
        },
      },
    );
    const data = response.data;
    if (data.answer) return data.answer;
    const results = data.results || [];
    if (results.length === 0) return "没有找到相关的旅游景点推荐";
    return results.map((r: any) => `- ${r.title}: ${r.content}`).join("\n");
  } catch (e) {
    return `搜索景点时遇到问题 - ${e instanceof Error ? e.message : String(e)}`;
  }
}
