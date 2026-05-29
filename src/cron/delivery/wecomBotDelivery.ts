import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";
import type { WechatSmartBotAdapter } from "../../wechat/wechatSmartBotAdapter.js";

/**
 * Delivers cron job results via the WeChat Smart Bot (WebSocket long connection).
 *
 * This channel reuses the same bot connection used for incoming messages,
 * so no additional corpId/corpSecret/agentId configuration is needed.
 *
 * Target formats:
 * - "wecom:user:<userId>"   — send to a specific user
 * - "wecom:chat:<chatId>"   — send to a specific group chat
 */
export class WecomBotDeliveryChannel implements DeliveryChannel {
  public readonly prefix = "wecom";

  public constructor(private readonly adapter: WechatSmartBotAdapter) {}

  public async deliver(target: string, payload: DeliveryPayload): Promise<void> {
    const targetId = this.resolveTarget(target);
    if (targetId === undefined) {
      throw new Error(
        `Invalid WeCom delivery target: "${target}". Expected "wecom:user:<id>" or "wecom:chat:<id>"`
      );
    }

    const content = this.formatContent(payload);
    await this.adapter.sendProactiveMessage(targetId, content);
  }

  private resolveTarget(target: string): string | undefined {
    const rest = target.slice(this.prefix.length + 1);
    const userMatch = /^user:(.+)$/.exec(rest);
    if (userMatch?.[1] !== undefined) {
      return userMatch[1];
    }
    const chatMatch = /^chat:(.+)$/.exec(rest);
    if (chatMatch?.[1] !== undefined) {
      return chatMatch[1];
    }
    return undefined;
  }

  private formatContent(payload: DeliveryPayload): string {
    if (payload.template !== undefined && payload.template.length > 0) {
      return payload.template.replace(/\{output\}/g, payload.output);
    }
    const header = `# ${payload.jobName}`;
    if (payload.error !== undefined && payload.error.length > 0) {
      return `${header}\n> status: error\n\n**Error:** ${payload.error}`;
    }
    return `${header}\n> status: success\n\n${payload.output}`;
  }
}
