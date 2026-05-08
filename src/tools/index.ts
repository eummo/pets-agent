import { registry } from "./registry.js";
import { registerAgentManagerTools } from "./agent-manager.js";

export function registerAllTools(): void {
  // Only agent orchestration tools
  registerAgentManagerTools();
}

export { registry };
