import { describe, expect, it } from "vitest";
import { formatStartupBanner } from "./startupBanner.js";

describe("formatStartupBanner", () => {
  it("prints startup status without secrets", () => {
    const banner = formatStartupBanner({
      serverUrl: "http://127.0.0.1:3000",
      devRoutesEnabled: true,
      agentSdk: {
        type: "claude",
        modelId: "MiniMax-M3",
        baseUrl: "https://api.example.test/anthropic"
      },
      intentLlm: {
        modelId: "MiniMax-M3",
        baseUrl: "https://api.example.test/anthropic"
      },
      runtimes: [
        { role: "reviewer", runtimeName: "claude-sdk-reviewer" },
        { role: "developer", runtimeName: "claude-sdk-developer" },
        { role: "admin", runtimeName: "claude-sdk-admin" }
      ],
      wechat: {
        status: "connecting",
        wsUrl: "wss://openws.work.weixin.qq.com"
      },
      cron: {
        enabled: true,
        tickIntervalMs: 60_000,
        staleGraceMs: 300_000,
        leaderLeaseTtlMs: 180_000,
        deliveryMode: "smart-bot-fallback"
      },
      paths: {
        knowledgeBasePath: ".harness/knowledge-base",
        conversationLogPath: ".harness/logs/conversation.jsonl",
        llmRawLogPath: ".harness/logs/llm-raw.jsonl",
        systemLogPath: ".harness/logs/system.jsonl",
        databasePath: ".harness/state/agent.db",
        sessionStorePath: ".harness/state/sessions.json",
        historyStorePath: ".harness/state/history.json",
        cronJobStorePath: ".harness/state/cron-jobs.json",
        cronLeaderLeasePath: ".harness/state/cron-jobs.json.leader"
      }
    });

    expect(banner).toContain("pets-agent startup");
    expect(banner).toContain("server: http://127.0.0.1:3000");
    expect(banner).toContain("agent sdk: claude MiniMax-M3");
    expect(banner).toContain("wechat wss: connecting");
    expect(banner).toContain("cron: enabled");
    expect(banner).toContain("leaderLeaseTtl=180000ms");
    expect(banner).toContain("delivery=smart-bot-fallback");
    expect(banner).toContain("knowledge base: .harness/knowledge-base");
    expect(banner).toContain("conversation=.harness/logs/conversation.jsonl");
    expect(banner).toContain("cronLeader=.harness/state/cron-jobs.json.leader");
    expect(banner).not.toMatch(/secret|api[_-]?key|authorization|token/i);
  });

  it("prints disabled cron compactly", () => {
    const banner = formatStartupBanner({
      serverUrl: "http://127.0.0.1:3000",
      devRoutesEnabled: false,
      agentSdk: { type: "pi", modelId: "model-a", baseUrl: "https://model.test" },
      intentLlm: { modelId: "intent-a", baseUrl: "https://intent.test" },
      runtimes: [{ role: "reviewer", runtimeName: "pi-reviewer" }],
      wechat: { status: "connecting", wsUrl: "wss://openws.work.weixin.qq.com" },
      cron: { enabled: false },
      paths: {
        knowledgeBasePath: ".harness/knowledge-base",
        conversationLogPath: ".harness/logs/conversation.jsonl",
        llmRawLogPath: ".harness/logs/llm-raw.jsonl",
        systemLogPath: ".harness/logs/system.jsonl",
        databasePath: ".harness/state/agent.db",
        sessionStorePath: ".harness/state/sessions.json",
        historyStorePath: ".harness/state/history.json"
      }
    });

    expect(banner).toContain("devRoutes=off");
    expect(banner).toContain("cron: disabled");
    expect(banner).not.toContain("cronJobs=");
    expect(banner).not.toContain("cronLeader=");
  });
});
