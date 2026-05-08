import type { AgentTool } from "@earendil-works/pi-agent-core";

export interface ToolDef {
  name: string;
  label: string;
  description: string;
  parameters: AgentTool["parameters"];
  prepareArguments?: (args: unknown) => unknown;
  execute: (
    toolCallId: string,
    params: unknown,
    signal?: AbortSignal,
  ) => Promise<{ content: { type: "text"; text: string }[]; details: unknown }>;
}

class ToolRegistry {
  private tools = new Map<string, ToolDef>();

  register(tool: ToolDef): void {
    this.tools.set(tool.name, tool);
  }

  get(name: string): ToolDef | undefined {
    return this.tools.get(name);
  }

  list(): ToolDef[] {
    return Array.from(this.tools.values());
  }

  getToolDefinitions(): AgentTool[] {
    return this.list().map((t) => ({
      name: t.name,
      label: t.label,
      description: t.description,
      parameters: t.parameters,
      prepareArguments: t.prepareArguments as AgentTool["prepareArguments"],
      async execute(toolCallId: string, params: unknown, signal?: AbortSignal) {
        const p = t.prepareArguments ? t.prepareArguments(params) : params;
        return t.execute(toolCallId, p, signal);
      },
    }));
  }
}

export const registry = new ToolRegistry();
