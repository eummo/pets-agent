import type { InboundMessage, OutboundMessage } from "./contracts.js";

export function handleCommandWithoutWorkspace(message: InboundMessage): OutboundMessage | undefined {
  const normalizedText = message.text.trim().toLowerCase();

  if (normalizedText === "/help") {
    return {
      text: [
        "Available commands:",
        "/new - start a fresh conversation",
        "/help - show this help message"
      ].join("\n")
    };
  }

  return undefined;
}

export function isNewConversationCommand(message: InboundMessage): boolean {
  return message.text.trim().toLowerCase() === "/new";
}
