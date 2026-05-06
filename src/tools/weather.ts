import axios from "axios";

export async function getWeather(city: string): Promise<string> {
  try {
    const response = await axios.get(
      `https://wttr.in/${encodeURIComponent(city)}?format=j1`,
    );
    const current = response.data.current_condition[0];
    return `${city}当前天气:${current.weatherDesc[0].value}，气温${current.temp_C}摄氏度`;
  } catch (e) {
    return `查询天气时遇到问题 - ${e instanceof Error ? e.message : String(e)}`;
  }
}
