import type { AgentRequest, AgentResponse, AgentRuntime } from "../core/ports.js";

export class EchoAgentRuntime implements AgentRuntime {
  public readonly name = "echo";

  public run(request: AgentRequest): Promise<AgentResponse> {
    return Promise.resolve({
      text: [
        "开发 harness 已连接。",
        `工作空间: ${request.workspacePath}`,
        `用户: ${request.user.id}`,
        `输入: ${request.text}`
      ].join("\n")
    });
  }

  public disposeSession(): Promise<void> {
    return Promise.resolve();
  }
}
