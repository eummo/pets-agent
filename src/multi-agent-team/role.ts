/**
 * Role — base class for all team roles.
 *
 * Template Method pattern: subclasses implement _prepare, _execute, _document.
 * The run() method orchestrates the SOP lifecycle.
 */

import type { TeamRole } from "./types.js";

export interface RoleContext {
  projectId: string;
  phase: string;
  input: unknown;
  workdir: string;
  /** If set, the LLM call should honour this signal for cancellation */
  signal?: AbortSignal;
  /** Timeout in ms for LLM call (used by BaseRoleLLM subclasses). Default: no timeout */
  timeoutMs?: number;
}

export abstract class Role {
  abstract readonly name: TeamRole;
  abstract readonly description: string;

  /**
   * Template method — do not override.
   * Implements the standard SOP lifecycle:
   * 1. Prepare — gather context, check prerequisites
   * 2. Execute — do the actual work
   * 3. Document — create artifacts
   * 4. Report — return structured result
   */
  async run(ctx: RoleContext): Promise<RoleResult> {
    const start = Date.now();

    // 1. Prepare
    const prep = await this._prepare(ctx);
    const prepBlocked = !prep.ok;
    if (prepBlocked) {
      const reason: string = (prep as { ok: false; reason: string; suggestions?: string[] }).reason;
      const suggestions: string[] = (prep as { ok: false; suggestions?: string[] }).suggestions ?? [];
      return {
        role: this.name,
        durationMs: Date.now() - start,
        status: "blocked",
        blockedReason: reason,
        artifacts: [],
        summary: `Blocked: ${reason}`,
        nextActions: suggestions,
      };
    }

    // 2. Execute
    const exec = await this._execute(ctx, prep.data);
    const execBlocked = exec.status === "blocked";
    if (execBlocked) {
      const reason: string = exec.reason ?? "unknown";
      const suggestions: string[] = exec.suggestions ?? [];
      return {
        role: this.name,
        durationMs: Date.now() - start,
        status: "blocked",
        blockedReason: reason,
        artifacts: [],
        summary: `Blocked: ${reason}`,
        nextActions: suggestions,
      };
    }

    // 3. Document
    const docs = await this._document(ctx, exec.data);

    // 4. Report
    return {
      role: this.name,
      durationMs: Date.now() - start,
      status: "completed",
      artifacts: docs.artifacts,
      summary: exec.summary ?? this.defaultSummary(exec.data),
      nextActions: exec.nextActions ?? [],
    };
  }

  protected async _prepare(_ctx: RoleContext): Promise<{ ok: true; data: unknown } | { ok: false; reason: string; suggestions?: string[] }> {
    return { ok: true, data: null };
  }

  protected abstract _execute(ctx: RoleContext, prepData: unknown): Promise<ExecuteResult>;
  protected async _document(_ctx: RoleContext, _execData: unknown): Promise<DocumentResult> {
    return { artifacts: [] };
  }

  protected defaultSummary(_data: unknown): string {
    return `${this.name} completed successfully.`;
  }
}

// ============================================================================
// Result Types
// ============================================================================

export type ExecuteStatus = "ok" | "blocked" | "partial";

export interface ExecuteResult {
  status: ExecuteStatus;
  reason?: string;
  suggestions?: string[];
  data?: unknown;
  summary?: string;
  nextActions?: string[];
}

export interface DocumentResult {
  artifacts: Array<{
    type: string;
    title: string;
    content: string;
  }>;
}

export interface RoleResult {
  role: TeamRole;
  durationMs: number;
  status: "completed" | "blocked";
  blockedReason?: string;
  artifacts: Array<{ type: string; title: string; content: string }>;
  summary: string;
  nextActions: string[];
}
