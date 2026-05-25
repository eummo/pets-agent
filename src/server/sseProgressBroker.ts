import type { ServerResponse } from "node:http";
import type { AgentProgressEvent, ChannelUser, ProgressReporter } from "../core/contracts.js";
import { writeSse } from "./sseUtils.js";

type Subscriber = {
  readonly id: string;
  readonly response: ServerResponse;
};

export class SseProgressBroker implements ProgressReporter {
  private readonly subscribers = new Map<string, Map<string, Subscriber>>();

  public subscribe(userId: string, response: ServerResponse): () => void {
    const subscriberId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const userSubscribers = this.subscribers.get(userId) ?? new Map<string, Subscriber>();
    userSubscribers.set(subscriberId, { id: subscriberId, response });
    this.subscribers.set(userId, userSubscribers);

    writeSse(response, "progress", {
      stage: "events.connected",
      message: "实时进度通道已连接。"
    });

    return () => {
      userSubscribers.delete(subscriberId);
      if (userSubscribers.size === 0) {
        this.subscribers.delete(userId);
      }
    };
  }

  public publish(user: ChannelUser, event: AgentProgressEvent): Promise<void> {
    const subscribers = this.subscribers.get(user.id);
    if (subscribers === undefined) {
      return Promise.resolve();
    }

    const payload = {
      ...event,
      timestamp: new Date().toISOString()
    };

    for (const subscriber of subscribers.values()) {
      writeSse(subscriber.response, "progress", payload);
    }

    return Promise.resolve();
  }
}

