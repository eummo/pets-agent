import type { ServerResponse } from "node:http";
import type { FastifyReply } from "fastify";

const SSE_HEADERS: Record<string, string> = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache, no-transform",
  connection: "keep-alive",
  "x-accel-buffering": "no",
};

export function setupSseResponse(reply: FastifyReply): void {
  reply.raw.writeHead(200, SSE_HEADERS);
  reply.raw.write("\n");
}

export function writeSse(response: ServerResponse, event: string, data: unknown): void {
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}
