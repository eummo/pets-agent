import type { DeliveryChannel, DeliveryPayload } from "../cronTypes.js";

/**
 * Delivers cron job results via the WeCom application message API.
 *
 * Target formats:
 * - "wecom:user:<userId>"   — send to a specific user
 * - "wecom:chat:<chatId>"   — send to a specific group chat
 * - "wecom:@all"            — send to all users in the application
 *
 * Requires corpid, corpsecret, and agentid configuration.
 * Access token is cached and refreshed automatically (valid for 7200s).
 */
export type WecomDeliveryConfig = {
  readonly corpId: string;
  readonly corpSecret: string;
  readonly agentId: string;
  readonly tokenCacheMs?: number;
};

type TokenResponse = {
  readonly errcode: number;
  readonly errmsg: string;
  readonly access_token?: string;
  readonly expires_in?: number;
};

type SendMessageResponse = {
  readonly errcode: number;
  readonly errmsg: string;
};

type WecomMessageBody = {
  readonly msgtype: string;
  readonly agentid: number;
  readonly markdown: { readonly content: string };
  readonly touser: string;
};

export class WecomAppMessageDeliveryChannel implements DeliveryChannel {
  public readonly prefix = "wecom";

  private readonly tokenCacheMs: number;
  private cachedToken: string | undefined;
  private tokenExpiresAt = 0;

  public constructor(private readonly config: WecomDeliveryConfig) {
    this.tokenCacheMs = config.tokenCacheMs ?? 7_200_000;
  }

  public async deliver(target: string, payload: DeliveryPayload): Promise<void> {
    const resolved = this.resolveTarget(target);
    if (resolved === undefined) {
      throw new Error(`Invalid WeCom delivery target: "${target}". Expected "wecom:user:<id>", "wecom:chat:<id>", or "wecom:@all"`);
    }

    const content = this.formatContent(payload);
    const accessToken = await this.getAccessToken();

    const touser = resolved.type === "all" ? "@all" : resolved.id;

    const body: WecomMessageBody = {
      msgtype: "markdown",
      agentid: Number(this.config.agentId),
      markdown: { content },
      touser,
    };

    const url = `https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=${accessToken}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const result = (await response.json()) as SendMessageResponse;
    if (result.errcode !== 0) {
      throw new Error(`WeCom message send failed: errcode=${result.errcode}, errmsg=${result.errmsg}`);
    }
  }

  private resolveTarget(target: string): { type: "user" | "chat" | "all"; id: string } | undefined {
    const rest = target.slice(this.prefix.length + 1);
    if (rest === "@all") {
      return { type: "all", id: "@all" };
    }
    const userMatch = /^user:(.+)$/.exec(rest);
    if (userMatch?.[1] !== undefined) {
      return { type: "user", id: userMatch[1] };
    }
    const chatMatch = /^chat:(.+)$/.exec(rest);
    if (chatMatch?.[1] !== undefined) {
      return { type: "chat", id: chatMatch[1] };
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

  private async getAccessToken(): Promise<string> {
    if (this.cachedToken !== undefined && Date.now() < this.tokenExpiresAt) {
      return this.cachedToken;
    }

    const url = `https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=${this.config.corpId}&corpsecret=${this.config.corpSecret}`;
    const response = await fetch(url);
    const result = (await response.json()) as TokenResponse;

    if (result.errcode !== 0 || result.access_token === undefined) {
      throw new Error(`WeCom gettoken failed: errcode=${result.errcode}, errmsg=${result.errmsg}`);
    }

    this.cachedToken = result.access_token;
    this.tokenExpiresAt = Date.now() + (result.expires_in ?? 7200) * 1000 - 60_000; // refresh 1 min early
    return this.cachedToken;
  }
}
