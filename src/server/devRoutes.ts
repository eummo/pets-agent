import type { FastifyInstance } from "fastify";
import { registerDevChatRoutes } from "./devChatRoutes.js";
import { registerDevEventRoutes } from "./devEventRoutes.js";
import { registerDevFeedbackRoutes } from "./devFeedbackRoutes.js";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { registerDevRoleRoutes } from "./devRoleRoutes.js";
import { registerDevStaticRoutes } from "./devStaticRoutes.js";

export type { DevRoutesOptions } from "./devRouteOptions.js";

export function registerDevRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  registerDevStaticRoutes(server);
  registerDevRoleRoutes(server, options);
  registerDevEventRoutes(server, options);
  registerDevChatRoutes(server, options);
  registerDevFeedbackRoutes(server, options);
}
