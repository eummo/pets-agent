import type { FastifyInstance } from "fastify";
import type { DevRoutesOptions } from "./devRouteOptions.js";
import { isLocalRequest, normalizeOptionalText } from "./serverUtils.js";

type DevRoleBody = {
  readonly userId?: string;
  readonly role?: string;
};

export function registerDevRoleRoutes(server: FastifyInstance, options: DevRoutesOptions): void {
  server.get("/dev/roles", async () => {
    if (options.roleConfigStore === undefined) {
      return { roles: [] };
    }
    const configs = await options.roleConfigStore.getAll();
    return { roles: configs.map((c) => ({ name: c.name, capabilities: c.capabilities ?? [] })) };
  });

  server.post<{ Body: DevRoleBody }>("/dev/role", async (request, reply) => {
    if (!isLocalRequest(request.ip)) {
      return reply.status(403).send({ error: "Role management is only available from localhost." });
    }

    const userId = normalizeOptionalText(request.body.userId) ?? "browser-user";
    const role = request.body.role;

    if (typeof role !== "string" || role.trim().length === 0) {
      return reply.status(400).send({ error: "Role must be a non-empty string." });
    }

    if (options.roleConfigStore !== undefined) {
      const config = await options.roleConfigStore.getByName(role);
      if (config === undefined) {
        return reply.status(400).send({ error: `Unknown role: ${role}` });
      }
    }

    if (options.authorization?.setRole === undefined) {
      return reply.status(501).send({ error: "Role management is not supported by this authorization service." });
    }
    options.authorization.setRole(userId, role);

    const currentRole = await options.authorization.roleFor({ id: userId });
    return reply.send({
      userId,
      role: currentRole
    });
  });
}
