import type { AgentRuntime, AgentRuntimeFactory } from "../agent/index.js";

export class RuntimeCache {
  private readonly cache: Map<string, AgentRuntime>;
  private readonly cacheOrder: string[];

  public constructor(
    agentRuntimes: Record<string, AgentRuntime>,
    private readonly runtimeFactory: AgentRuntimeFactory | undefined,
    private readonly maxCacheSize = 16,
  ) {
    this.cache = new Map(Object.entries(agentRuntimes));
    this.cacheOrder = Object.keys(agentRuntimes);
  }

  public async resolve(role: string): Promise<AgentRuntime | undefined> {
    const cacheKey = await this.cacheKeyForRole(role);

    const cached = this.cache.get(cacheKey);
    if (cached !== undefined) return cached;

    if (this.runtimeFactory !== undefined) {
      const created = await this.runtimeFactory.createRuntime(role);
      if (created !== undefined) {
        this.cacheRuntime(cacheKey, created);
        return created;
      }
    }

    return this.cache.get(role);
  }

  private cacheRuntime(role: string, runtime: AgentRuntime): void {
    this.cache.set(role, runtime);
    this.cacheOrder.push(role);
    while (this.cacheOrder.length > this.maxCacheSize) {
      const oldest = this.cacheOrder.shift();
      if (oldest !== undefined) {
        this.cache.delete(oldest);
      }
    }
  }

  private async cacheKeyForRole(role: string): Promise<string> {
    if (this.runtimeFactory?.cacheKeyForRole === undefined) {
      return role;
    }

    return await this.runtimeFactory.cacheKeyForRole(role) ?? role;
  }
}
