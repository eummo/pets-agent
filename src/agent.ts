import * as path from "path";
import * as fs from "fs";
import { homedir } from "os";
import { Agent } from "@mariozechner/pi-agent-core";
import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import { streamSimple, type Model } from "@mariozechner/pi-ai";
import { getProvider, getApiKey, loadConfig } from "./config.js";
import { registry, registerAllTools } from "./tools/index.js";
import {
  loadSkills,
  formatSkillsForPrompt,
} from "./skills/loader.js";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { AgentEvent } from "@mariozechner/pi-agent-core";

export function createAgent(extraSystemPrompt = ""): Agent {
  registerAllTools();

  const config = loadConfig();
  const provider = getProvider();
  const apiKey = getApiKey();

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

  const systemPrompt = [
    `你是一个智能旅行助手。`,
    ``,
    `可用工具:`,
    ...toolDescs,
    ``,
    `工作流程:`,
    `1. 先用 get_weather 查询天气`,
    `2. 再用 get_attraction 根据天气推荐景点`,
    `3. 给出具体旅游建议`,
    ``,
    extraSystemPrompt,
    agentsMd,
    skillsSection,
  ]
    .filter(Boolean)
    .join("\n");

  const tools = registry.getToolDefinitions() as AgentTool[];

  const agent = new Agent({
    initialState: {
      systemPrompt,
      model: {
        id: provider.model_id,
        provider: config.llm.provider,
      } as any,
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

export function subscribeToAgent(agent: Agent): void {
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
