import type { AuthorizationService, FeedbackStore, MessageGateway, RoleConfigStore } from "../core/contracts.js";
import type { SseProgressBroker } from "./sseProgressBroker.js";

export type DevRoutesOptions = {
  readonly messageHandler: MessageGateway;
  readonly roleConfigStore?: RoleConfigStore | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly authorization?: AuthorizationService | undefined;
  readonly progressBroker?: SseProgressBroker | undefined;
};
