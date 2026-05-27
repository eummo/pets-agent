import type { InboundMessage } from "../core/contracts.js";

export type KnowledgeWorkspace = {
  readonly kind: "knowledge-base" | "source-repository";
  readonly id: string;
  readonly path: string;
};

export type KnowledgeWorkspaceResolver = {
  resolve(message: InboundMessage): Promise<readonly KnowledgeWorkspace[]>;
};