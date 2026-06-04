import type { MessageGateway } from "../core/index.js";
import type { AuthorizationService, RoleConfigStore } from "../auth/index.js";
import type { FeedbackStore } from "../persistence/index.js";
import type { SseProgressBroker } from "./sseProgressBroker.js";

export type DevRoutesOptions = {
  readonly messageHandler: MessageGateway;
  readonly roleConfigStore?: RoleConfigStore | undefined;
  readonly feedbackStore?: FeedbackStore | undefined;
  readonly authorization?: AuthorizationService | undefined;
  readonly progressBroker?: SseProgressBroker | undefined;
  readonly uploadRootPath?: string | undefined;
};
