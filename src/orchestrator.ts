import * as path from "path";
import * as fs from "fs";
import { homedir } from "os";
import { Agent } from "@mariozechner/pi-agent-core";
import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import { streamSimple, getModel } from "@mariozechner/pi-ai";
import { getApiKey, loadConfig } from "./config.js";
import { registry, registerAllTools } from "./tools/index.js";
import { registerAgentManagerTools } from "./tools/agent-manager.js";
import {
  loadSkills,
  formatSkillsForPrompt,
} from "./skills/loader.js";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { AgentEvent } from "@mariozechner/pi-agent-core";

export function createOrchestratorAgent(extraSystemPrompt = ""): Agent {
  registerAllTools();
  registerAgentManagerTools();

  const config = loadConfig();
  const apiKey = getApiKey();

  // Resolve full model from registry (includes `api` field needed by streamSimple)
  const providerName = config.llm.provider as "minimax-cn";
  const modelId = config.llm.providers[config.llm.provider].model_id;
  const resolvedModel = getModel(providerName, modelId as any);

  const defaultDirs = [
    path.join(homedir(), ".pets-agent", "skills"),
    path.join(process.cwd(), "skills"),
  ];
  const skillsDirs = config.skills?.dirs ?? defaultDirs;
  const result = loadSkills(skillsDirs);
  const skillsSection = formatSkillsForPrompt(result.skills);

  const agentsMd = loadAgentsMd();

  const toolDescs = registry.list().map((t) => {
    const params = t.parameters as any;
    const propNames = params.properties ? Object.keys(params.properties).join(", ") : "";
    return `- ${t.name}(${propNames}): ${t.description}`;
  });

  const orchestrationPrompt = `
你是一个 agent 编排平台。你可以同时管理多个子 agent（Claude Code、Codex、Kiro 等）。
每个子 agent 在独立进程中运行，你通过 spawn_agent 工具启动它们，通过 list_tasks / get_task 监控进度。

示例工作流：
1. 用户要求实现一个功能 → spawn_agent("claude-code", "实现 xxx 功能")
2. 用 get_task 查看进度
3. 如果需要可以 kill_task 停止

当前活跃任务可通过 list_tasks 查看。

编排能力：
- 可以同时启动多个子 agent 并行工作
- 可以监控子 agent 的实时输出
- 可以随时停止不需要的子 agent
- 每个子 agent 都是独立的进程，有自己的状态和输出流
`.trim();

  const systemPrompt = [
    `你是一个智能旅行助手，同时也是 agent 编排平台。`,
    ``,
    `可用工具:`,
    ...toolDescs,
    ``,
    `工作流程 (旅行相关):`,
    `1. 先用 get_weather 查询天气`,
    `2. 再用 get_attraction 根据天气推荐景点`,
    `3. 给出具体旅游建议`,
    ``,
    extraSystemPrompt,
    agentsMd,
    skillsSection,
    ``,
    `--- 编排能力 ---`,
    orchestrationPrompt,
  ]
    .filter(Boolean)
    .join("\n");

  const tools = registry.getToolDefinitions() as AgentTool[];

  const agent = new Agent({
    initialState: {
      systemPrompt,
      model: resolvedModel as any,
      tools,
      thinkingLevel: "off" as ThinkingLevel,
    },
    streamFn: streamSimple as any,
    getApiKey: () => apiKey,
    toolExecution: "sequential",
  });

  return agent;
}

function loadAgentsMd(): string {
  const paths = [
    path.join(process.cwd(), "AGENTS.md"),
    path.join(process.cwd(), "CLAUDE.md"),
  ];
  for (const p of paths) {
    try {
      if (fs.existsSync(p)) {
        return "\n\n" + fs.readFileSync(p, "utf8");
      }
    } catch {}
  }
  return "";
}

export function subscribeToOrchestrator(agent: Agent): void {
  agent.subscribe((event: AgentEvent, _signal: any) => {
    switch (event.type) {
      case "tool_execution_start":
        console.log(`>>> 调用工具: ${event.toolName}`);
        break;
      case "tool_execution_end":
        console.log(
          `<<< 工具完成: ${event.toolName} ${event.isError ? "(错误)" : "(成功)"}`,
        );
        break;
      case "message_end": {
        const msg = event.message as any;
        if (msg?.role === "assistant") {
          const text = (msg.content as any[])
            .filter((c) => c.type === "text" || c.type === "toolCall")
            .map((c: any) => {
              if (c.type === "text") return c.text;
              if (c.type === "toolCall")
                return `[调用 ${c.name}: ${JSON.stringify(c.arguments)}]`;
              return "";
            })
            .join("");
          if (text) console.log(`\n[助手] ${text}`);
        }
        break;
      }
    }
  });
}
