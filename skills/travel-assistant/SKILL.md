---
name: travel-assistant
description: 旅行助手技能。用于查询天气、搜索景点、提供旅游建议。当用户询问旅行、天气、景点推荐时使用此技能。
---

# 旅行助手技能

## 功能

- 查询任意城市的实时天气
- 根据城市和天气条件搜索推荐景点
- 提供旅游建议和行程规划

## 使用工具

- `get_weather(city)` - 查询城市天气
- `get_attraction(city, weather)` - 根据天气搜索景点

## 工作流程

1. 首先使用 `get_weather` 查询目的地天气
2. 根据天气使用 `get_attraction` 搜索景点
3. 综合天气和景点给出旅游建议

## 示例

用户: "北京天气怎么样？有什么好玩的地方？"

1. 调用 `get_weather(city="北京")`
2. 调用 `get_attraction(city="北京", weather="晴天")`
3. 综合结果给出建议
