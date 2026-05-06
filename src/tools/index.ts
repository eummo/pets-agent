import { registry } from "./registry.js";
import { getWeather } from "./weather.js";
import { searchAttraction } from "./tavily.js";
import type { ToolDef } from "./registry.js";

function makeTool(def: Omit<ToolDef, "execute"> & { execute: ToolDef["execute"] }): ToolDef {
  return def as ToolDef;
}

const weatherTool: ToolDef = makeTool({
  name: "get_weather",
  label: "查询天气",
  description: "查询指定城市的实时天气",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "城市名称" },
    },
    required: ["city"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { city: string };
  },
  async execute(_toolCallId, params) {
    const p = params as { city: string };
    const result = await getWeather(p.city);
    return { content: [{ type: "text", text: result }], details: { city: p.city, result } };
  },
});

const attractionTool: ToolDef = makeTool({
  name: "get_attraction",
  label: "搜索景点",
  description: "根据城市和天气搜索推荐的旅游景点",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "城市名称" },
      weather: { type: "string", description: "当前天气情况" },
    },
    required: ["city", "weather"],
  },
  prepareArguments(args: unknown) {
    if (typeof args === "string") args = JSON.parse(args);
    return args as { city: string; weather: string };
  },
  async execute(_toolCallId, params) {
    const p = params as { city: string; weather: string };
    const result = await searchAttraction(p.city, p.weather);
    return {
      content: [{ type: "text", text: result }],
      details: { city: p.city, weather: p.weather, result },
    };
  },
});

export function registerAllTools(): void {
  registry.register(weatherTool);
  registry.register(attractionTool);
}

export { registry };
