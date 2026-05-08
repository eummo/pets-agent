import * as path from "path";
import * as fs from "fs";
import { homedir } from "os";
import { Agent } from "@earendil-works/pi-agent-core";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import { streamSimple, getModel } from "@earendil-works/pi-ai";
import { getApiKey, loadConfig } from "./config.js";
import { registry, registerAllTools } from "./tools/index.js";
import { registerAgentManagerTools } from "./tools/agent-manager.js";
import {
  loadSkills,
  formatSkillsForPrompt,
} from "./skills/loader.js";
import type { AgentTool } from "@earendil-works/pi-agent-core";

export function createOrchestratorAgent(extraSystemPrompt = ""): Agent {
  registerAllTools();
  registerAgentManagerTools();

  const config = loadConfig();
  const apiKey = getApiKey();

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
你是一个 agent 编排平台，**优先使用 Claude Code** 处理编程任务。

**Agent 选择策略（按优先级）：**
1. **claude-code** (最高优先级) - 通用编程、代码修改、调试、文件操作
2. **pi-agent** - 当需要 pi-mono 框架内置能力时使用
3. **codex / kiro** - 当 claude-code 不可用时的备选

**为什么优先 claude-code：**
- Claude Code 是成熟的编程 agent，功能完整
- 在 WSL 环境中经过验证，稳定性高
- 支持 -p (print mode) 非交互模式，适合自动化

**spawn_agent 使用示例：**
1. 用户要求实现功能 → spawn_agent("claude-code", "实现 xxx 功能")
2. 用户要求代码审查 → spawn_agent("claude-code", "审查代码")
3. 用户要求调试 → spawn_agent("claude-code", "调试并修复问题")
4. 需要 pi-mono 框架能力 → spawn_agent("pi-agent", "使用 pi agent 完成...")

**当前活跃任务可通过 list_tasks 查看。**

**编排能力：**
- 可以同时启动多个子 agent 并行工作
- 可以监控子 agent 的实时输出
- 可以随时停止不需要的子 agent
- 每个子 agent 都是独立的进程/会话，有自己的状态和输出流

**任务分解 (decompose_task) — 复杂任务用：**
当任务涉及多个独立步骤、需要不同专业领域、或规模较大时，使用 decompose_task 工具分解。

典型场景：
- "实现 XXX 系统" → 分解为前端、后端、测试等多个子任务并行
- "调研并对比 A vs B" → 分解为两个独立调研任务并行，再汇总
- "分析项目问题" → 分解为代码扫描、日志分析、复现验证等步骤

分解模式：
1. 调用 decompose_task，传入 subtasks 数组（每个含 title、agentType、prompt）
2. 所有子任务自动并行启动
3. 监控子任务状态：list_tasks / get_task
4. 所有子任务完成后，父任务自动收到 task_complete 事件
5. 可选：在父任务下继续 spawn 后续汇总任务

简单任务（一步搞定）直接 spawn_agent，不需要分解。
`.trim();

  const systemPrompt = [
    `你是一个智能开发助手，同时也是 agent 编排平台。`,

    `可用工具:`,
    ...toolDescs,

    extraSystemPrompt,
    agentsMd,
    skillsSection,

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

export type LogLine = {
  text: string;
  style?: "info" | "user" | "agent" | "tool_start" | "tool_end" | "error" | "task_update" | "task_exit";
};

type SubscribeOptions = {
  onLog: (line: LogLine) => void;
};

export function subscribeToOrchestrator(agent: Agent, options: SubscribeOptions): void {
  agent.subscribe((event: any, _signal: any) => {
    switch (event.type) {
      case "tool_execution_start":
        options.onLog({ text: `>>> 调用工具: ${event.toolName}`, style: "tool_start" });
        break;
      case "tool_execution_end":
        options.onLog({
          text: `<<< 工具完成: ${event.toolName} ${event.isError ? "(错误)" : "(成功)"}`,
          style: event.isError ? "error" : "tool_end",
        });
        break;
      case "message_end": {
        const msg = event.message as any;
        if (msg?.role === "assistant") {
          const text = (msg.content as any[])
            .filter((c: any) => c.type === "text" || c.type === "toolCall")
            .map((c: any) => {
              if (c.type === "text") return c.text;
              if (c.type === "toolCall")
                return `[调用 ${c.name}: ${JSON.stringify(c.arguments)}]`;
              return "";
            })
            .join("");
          if (text) options.onLog({ text, style: "agent" });
        }
        break;
      }
      default: {
        const e = event as any;
        if (e.message && typeof e.message === "string") {
          options.onLog({ text: e.message, style: "error" });
        }
        break;
      }
    }
  });
}