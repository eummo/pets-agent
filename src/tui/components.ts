/**
 * TUI Components - custom Component implementations
 *
 * Reuses pi-tui: Text, Container, Input, Spacer, TUI, etc.
 * Extends with: ChatLog, ThinkingSpinner, SlashCommandHelp
 */

import {
  type Component,
  Container,
  Input,
  setKeybindings,
  Spacer,
  Text,
  truncateToWidth,
  visibleWidth,
  KeybindingsManager,
} from "@earendil-works/pi-tui";
import chalk from "chalk";
import type { LogLine } from "../orchestrator.js";

// ============================================================================
// Theme helpers
// ============================================================================

const DIM = chalk.dim;
const BOLD = chalk.bold;
const GREEN = chalk.green;
const MAGENTA = chalk.magenta;

// ============================================================================
// ChatLog - scrollable chat history
// ============================================================================

export interface ChatEntry {
  role: "user" | "agent" | "tool";
  text: string;
  timestamp: number;
}

export class ChatLog implements Component {
  private entries: ChatEntry[] = [];
  private maxEntries = 200;

  invalidate(): void {}

  clear(): void {
    this.entries = [];
  }

  pushUser(text: string): void {
    this.entries.push({ role: "user", text, timestamp: Date.now() });
    this.trim();
  }

  pushAgent(text: string): void {
    this.entries.push({ role: "agent", text, timestamp: Date.now() });
  }

  pushTool(text: string): void {
    this.entries.push({ role: "tool", text, timestamp: Date.now() });
    this.trim();
  }

  private trim(): void {
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries);
    }
  }

  render(width: number): string[] {
    if (this.entries.length === 0) return [];

    const lines: string[] = [];
    const visibleEntries = this.entries.slice(-40);

    for (const entry of visibleEntries) {
      if (entry.role === "user") {
        const truncated = truncateToWidth(entry.text, width - 2, "");
        lines.push(GREEN(BOLD("> ")) + truncated);
        lines.push("");
      } else if (entry.role === "agent") {
        for (const l of entry.text.split("\n")) {
          const truncated = truncateToWidth(l, width, "");
          lines.push(MAGENTA(truncated));
        }
        lines.push("");
      } else if (entry.role === "tool") {
        const truncated = truncateToWidth(entry.text, width - 4, "");
        lines.push(DIM(">>> ") + truncated);
        lines.push("");
      }
    }

    return lines;
  }
}

// ============================================================================
// ThinkingSpinner - braille dot spinner animation
// ============================================================================

export class ThinkingSpinner implements Component {
  private text = "";
  private interval: NodeJS.Timeout | null = null;
  private frameIndex = 0;
  private readonly frames = ["⠙", "⠹", "⠼", "⠴", "⠦", "⠧", "⠇", "⠋"];
  private readonly intervalMs = 300;

  invalidate(): void {}

  start(): void {
    if (this.interval) return;
    this.frameIndex = 0;
    this.text = DIM(this.frames[this.frameIndex]) + " " + DIM("thinking");
    this.interval = setInterval(() => {
      this.frameIndex = (this.frameIndex + 1) % this.frames.length;
      this.text = DIM(this.frames[this.frameIndex]) + " " + DIM("thinking");
    }, this.intervalMs);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.text = "";
  }

  render(_width: number): string[] {
    return this.text ? [this.text] : [];
  }
}

// ============================================================================
// SlashCommandHelp - multi-line help text (agent response)
// ============================================================================

export function buildHelpText(): string {
  return [
    BOLD("Commands:"),
    "  /quit, /exit   Quit the application",
    "  /clear         Clear the screen",
    "  /tasks         List running tasks",
    "  /history       Show command history",
    "  /logs [id]     Show task logs",
    "  /help          Show this help",
    "",
    BOLD("Input:"),
    "  Type text and press Enter to chat",
    "  /command to execute a slash command",
  ].join("\n");
}

// ============================================================================
// Header component
// ============================================================================

export function buildHeader(): string {
  return BOLD(GREEN("Pets Agent")) + DIM(" · ^C/^D clear/exit · / commands · ! bash");
}
