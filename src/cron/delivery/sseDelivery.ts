import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";
import type { SseProgressBroker } from "../../server/sseProgressBroker.js";

/**
 * Delivers cron job results through the SSE progress broker.
 * Browser clients subscribed to the user's SSE stream receive the result
 * as a `cron.result` progress event.
 *
 * Target format: "sse:<userId>"
 */
export class SseDeliveryChannel implements DeliveryChannel {
  public readonly prefix = "sse";

  public constructor(private readonly broker: SseProgressBroker) {}

  public async deliver(target: string, payload: DeliveryPayload): Promise<void> {
    const userId = target.slice(this.prefix.length + 1);
    if (userId.length === 0) {
      throw new Error(`Invalid SSE delivery target: "${target}". Expected "sse:<userId>"`);
    }
    await this.broker.publish(
      { id: userId },
      {
        stage: "cron.result",
        message: payload.jobName,
        data: { output: payload.output, error: payload.error, template: payload.template },
      }
    );
  }
}
