import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";

/**
 * Routes each delivery target to the matching DeliveryChannel by prefix.
 * The scheduler calls this composite channel with the full list of targets
 * from a job's DeliveryTarget.channels, and each target is dispatched
 * independently to the appropriate handler.
 */
export class CompositeDeliveryChannel implements DeliveryChannel {
  public readonly prefix = "";

  public constructor(private readonly channels: readonly DeliveryChannel[]) {}

  public async deliver(target: string, payload: DeliveryPayload): Promise<void> {
    const handler = this.channels.find((c) => target.startsWith(`${c.prefix}:`));
    if (handler === undefined) {
      throw new Error(
        `No delivery channel found for target "${target}". ` +
          `Available prefixes: ${this.channels.map((c) => c.prefix).join(", ")}`
      );
    }
    await handler.deliver(target, payload);
  }

  /**
   * Deliver to all channels listed in the DeliveryTarget.
   * Errors in one channel do not prevent delivery to others;
   * the first error is thrown after all channels have been attempted.
   */
  public async deliverAll(
    channels: readonly string[],
    payload: DeliveryPayload
  ): Promise<void> {
    let firstError: Error | undefined;
    for (const target of channels) {
      try {
        await this.deliver(target, payload);
      } catch (error) {
        firstError ??= error instanceof Error ? error : new Error(String(error));
      }
    }
    if (firstError !== undefined) {
      throw firstError;
    }
  }
}
