# Pets Agent 项目指令

## 项目概述

Pets Agent 是一个基于 pi-mono 框架的智能旅行助手 Agent，支持工具调用、Skill 加载和 AGENTS.md 指令注入。

## 目录结构

```
pets-agent/
├── src/
│   ├── index.ts       # 主入口，REPL 交互
│   ├── agent.ts      # Agent 创建与配置
│   ├── config.ts     # YAML 配置加载
│   ├── tools/        # 工具系统
│   │   ├── registry.ts   # 工具注册表
│   │   ├── weather.ts   # 天气查询工具
│   │   ├── tavily.ts    # Tavily 搜索工具
│   │   └── index.ts     # 统一导出
│   └── skills/
│       └── loader.ts     # Skill 加载器
├── skills/           # Agent Skills (SKILL.md 格式)
├── config/
│   └── config.yaml   # LLM 提供商配置
└── .env              # 环境变量
```

## 可用工具

- `get_weather(city)` - 查询城市天气
- `get_attraction(city, weather)` - 根据天气搜索景点

## Skill 系统

Skills 放在 `skills/` 目录，每个 Skill 是一个 `SKILL.md` 文件。

### Skill 格式

```markdown
---
name: my-skill
description: 描述这个 skill 的用途和使用场景
---

# My Skill

## 使用方法

...
```

### 加载规则

- `skills/` 目录下的 `SKILL.md` 文件
- `skills/<name>/SKILL.md` 嵌套目录
- Skills 自动注入到 system prompt

## 添加新工具

1. 在 `src/tools/` 创建 `xxx.ts`
2. 实现工具函数
3. 在 `src/tools/index.ts` 注册
4. 工具自动出现在 system prompt
