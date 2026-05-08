/**
 * App keybindings - extends TUI_KEYBINDINGS with app-level bindings
 */

import {
  TUI_KEYBINDINGS,
  type KeybindingDefinitions,
  type KeyId,
} from "@earendil-works/pi-tui";

export const APP_KEYBINDINGS: KeybindingDefinitions = {
  ...TUI_KEYBINDINGS,
  "app.interrupt": { defaultKeys: "escape" as KeyId, description: "Cancel or abort" },
  "app.clear": { defaultKeys: "ctrl+c" as KeyId, description: "Clear editor" },
  "app.exit": { defaultKeys: "ctrl+d" as KeyId, description: "Exit when editor is empty" },
  "app.suspend": { defaultKeys: "ctrl+z" as KeyId, description: "Suspend to background" },
};

export type AppKeybinding = keyof typeof APP_KEYBINDINGS;

export function keyText(keybinding: AppKeybinding): string {
  const binding = APP_KEYBINDINGS[keybinding];
  if (!binding) return keybinding;
  const keys = Array.isArray(binding.defaultKeys) ? binding.defaultKeys : [binding.defaultKeys];
  return keys.map((k) => {
    if (k === "escape") return "ctrl+[";
    if (k === "ctrl+c") return "^C";
    if (k === "ctrl+d") return "^D";
    if (k === "ctrl+z") return "^Z";
    if (k === "ctrl+o") return "^O";
    if (k === "ctrl+l") return "^L";
    if (k === "ctrl+p") return "^P";
    if (k === "ctrl+t") return "^T";
    if (k === "ctrl+y") return "^Y";
    if (k === "ctrl+a") return "^A";
    if (k === "ctrl+e") return "^E";
    if (k === "ctrl+k") return "^K";
    if (k === "ctrl+u") return "^U";
    if (k === "ctrl+w") return "^W";
    if (k === "ctrl+n") return "^N";
    if (k === "alt+enter") return "Alt+Enter";
    if (k === "alt+up") return "Alt+Up";
    if (k === "alt+v") return "Alt+V";
    if (k === "shift+tab") return "Shift+Tab";
    if (k === "shift+enter") return "Shift+Enter";
    if (k === "tab") return "Tab";
    if (k === "enter") return "Enter";
    if (k === "backspace") return "⌫";
    if (k === "delete") return "Del";
    if (k === "up") return "↑";
    if (k === "down") return "↓";
    if (k === "left") return "←";
    if (k === "right") return "→";
    if (k === "home") return "Home";
    if (k === "end") return "End";
    if (k === "pageUp") return "PgUp";
    if (k === "pageDown") return "PgDn";
    return k;
  }).join("/");
}

export function hint(keybinding: AppKeybinding, description: string): string {
  const { chalk } = require("chalk");
  return chalk.dim(keyText(keybinding)) + " " + description;
}
