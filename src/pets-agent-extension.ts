/**
 * Pets-Agent Extension for pi-coding-agent
 *
 * Registers orchestrator tools (spawn_agent, decompose_task, list_tasks, etc.)
 * that delegate to sub-agents (claude-code, codex, pi-agent, kiro) via
 * the existing AgentManager process-spawning engine.
 *
 * Usage: pi --extension pets-agent
 */

import { type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerTaskTools } from "./tools/task-tools.js";
import { registerMemoryTools } from "./tools/memory-tools.js";
import { registerTeamTools } from "./tools/team-tools.js";
import { memoryInjector } from "./memory/injector.js";

export default function petsAgentExtension(pi: ExtensionAPI): void {
  registerTaskTools(pi);
  registerMemoryTools(pi);
  registerTeamTools(pi);

  // System prompt injection
  pi.on("before_agent_start", (event) => {
    const workdir = event.systemPromptOptions.cwd ?? process.cwd();
    const memoryBlock = memoryInjector.buildBlock({ workdir, includeSkills: true });

    const orchestratorSection = `
## Orchestration Capabilities

You are a development assistant with agent orchestration capabilities.

**Orchestrator Tools:**
- spawn_agent(agentType, prompt) — launch a sub-agent (claude-code preferred for coding)
- list_tasks — view all running/completed tasks
- get_task(taskId) — view task details and recent output
- kill_task(taskId) — stop a running task
- list_task_history — query past task executions
- decompose_task(taskDescription, subtasks) — split complex tasks into parallel subtasks

**Memory Tools:**
- remember_pattern(pattern, tags?) — save a successful command/workflow
- remember_prefs(agentType, taskPrompt, success, exitCode?, durationSec?) — record agent performance
- remember_project(workdir, content, tags?) — save per-project context
- refresh_memory(workdir?) — force reload memory snapshots from disk
- get_memory(type?, workdir?, query?) — view/search memory
- forget_memory(type, idOrText) — remove a memory entry

**Skill Tools:**
- list_skills(query?) — list all available skills
- view_skill(name) — view full content of a specific skill

**Project Team Tools:**
- create_project(name, description, target?, successCriteria?)
- list_projects(status?)
- get_project(projectId)
- plan_phase(projectId, phase)
- run_role(projectId, role, phase, input?, workdir?)
- create_artifact(projectId, type, title, content, phase)
- review_artifact(projectId, artifactId, verdict, comment?)
- advance_phase(projectId)
- make_decision(projectId, topic, options, rationale, selected, madeBy)
- team_meeting(projectId, topic, participants, notes?)
- generate_doc(type, projectName, input?)

**Agent Selection:**
1. claude-code — general coding, file operations, debugging
2. pi-agent — when pi-mono framework capabilities are needed
3. codex / kiro — fallback options

**Task Decomposition:**
When a task spans multiple domains, requires independent steps, or is large in scope,
use decompose_task to split it into parallel subtasks, then monitor with list_tasks.
Simple single-step tasks should use spawn_agent directly.
`.trim();

    return {
      systemPrompt: `${event.systemPrompt}${memoryBlock}\n\n${orchestratorSection}`,
    };
  });

  // Custom branding header
  pi.on("session_start", async (_event, ctx) => {
    if (ctx.hasUI) {
      ctx.ui.setHeader((_tui, theme) => {
        const muted = (text: string) => theme.fg("muted", text);
        const dim = (text: string) => theme.fg("dim", text);
        return {
          render(_width: number): string[] {
            const width = 43;
            const content = "   Pets-Agent  ·  开发助手";
            const contentCells = 27;
            const pad = width - contentCells;
            const left = " ".repeat(pad >> 1);
            const right = " ".repeat(pad - (pad >> 1));
            return [
              "",
              left + content + right,
              "",
              `${muted("/commands")} ${dim("· /help for available slash commands")}`,
              "",
            ];
          },
          invalidate() {},
        };
      });
    }
  });
}
