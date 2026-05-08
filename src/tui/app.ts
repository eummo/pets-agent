/**
 * PetsApp - TUI app builder
 *
 * Composes pi-tui components + custom TUI components into a running app.
 * All special logic (thinking animation, slash commands) lives here.
 */

import "dotenv/config";
import {
  Container,
  Input,
  KeybindingsManager,
  ProcessTerminal,
  setKeybindings,
  Spacer,
  TUI,
  Text,
} from "@earendil-works/pi-tui";

import { APP_KEYBINDINGS } from "./keybindings.js";
import {
  ChatLog,
  ThinkingSpinner,
  buildHeader,
} from "./components.js";
import { handleSlashCommand } from "./commands.js";
import { createOrchestratorAgent, subscribeToOrchestrator } from "../orchestrator.js";
import { agentManager } from "../tasks/agent-manager.js";
import { taskHistory } from "../tasks/task-history.js";
import type { LogLine } from "../orchestrator.js";

// Ensure history is persisted on unexpected exit
process.on("exit", () => taskHistory.flush());
process.on("SIGTERM", () => {
  taskHistory.flush();
  agentManager.destroy();
});

export interface PetsApp {
  tui: TUI;
  chatLog: ChatLog;
  userInput: Input;
  agent: ReturnType<typeof createOrchestratorAgent>;
}

export function createPetsApp(): PetsApp {
  // Create terminal and TUI
  const terminal = new ProcessTerminal();
  const tui = new TUI(terminal);

  // Setup keybindings
  const keybindings = new KeybindingsManager(APP_KEYBINDINGS);
  setKeybindings(keybindings);

  // Create components
  const chatLog = new ChatLog();
  const userInput = new Input();
  const headerText = new Text(buildHeader(), 0, 0);
  const spacer = new Spacer();
  const thinkingSpinner = new ThinkingSpinner();

  // Layout container
  const root = new Container();
  root.addChild(headerText);
  root.addChild(spacer);
  root.addChild(chatLog);
  root.addChild(thinkingSpinner);
  root.addChild(userInput);

  tui.addChild(root);
  tui.setFocus(userInput);

  // Create agent
  const agent = createOrchestratorAgent();
  subscribeToOrchestrator(agent, {
    onLog: (line: LogLine) => {
      if (line.style === "tool_start" || line.style === "tool_end") {
        chatLog.pushTool(line.text);
      } else if (line.style === "agent") {
        chatLog.pushAgent(line.text);
      }
      tui.requestRender();
    },
  });

  // Track agent state
  let agentBusy = false;

  agent.subscribe((event: any) => {
    switch (event.type) {
      case "message_start":
        if (event.message?.role === "assistant") {
          agentBusy = true;
          thinkingSpinner.start();
        }
        break;
      case "agent_end":
        thinkingSpinner.stop();
        agentBusy = false;
        break;
    }
  });

  // Task manager events
  agentManager.on("update", () => {
    tui.requestRender();
  });

  // Input handling
  userInput.onSubmit = async (value: string) => {
    const trimmed = value.trim();
    userInput.setValue("");

    if (!trimmed) return;

    if (trimmed.startsWith("/")) {
      await handleSlashCommand(trimmed, chatLog, tui, createOrchestratorAgent);
      return;
    }

    if (agentBusy) {
      chatLog.pushTool("Agent is busy, please wait...");
      tui.requestRender();
      return;
    }

    chatLog.pushUser(trimmed);
    tui.requestRender();

    try {
      await agent.prompt(trimmed);
      await agent.waitForIdle();
    } catch (err: any) {
      chatLog.pushTool(`Error: ${err.message}`);
    }

    tui.requestRender();
  };

  // Escape to interrupt
  userInput.onEscape = () => {
    if (agentBusy) {
      agent.abort();
      chatLog.pushTool("Interrupted.");
      agentBusy = false;
      thinkingSpinner.stop();
      tui.requestRender();
    }
  };

  return { tui, chatLog, userInput, agent };
}

export function startPetsApp(app: PetsApp): void {
  app.tui.start();
}
