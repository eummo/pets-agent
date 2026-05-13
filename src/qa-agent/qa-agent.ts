/**
 * QAAgent — Retrieve-then-Generate QA agent powered by MiniMax LLM.
 *
 * Queries the Memory system for relevant context, then generates
 * a natural language answer using the LLM.
 */

import { MemoryRetriever } from "./memory-retriever.js";

export interface QAAgentConfig {
  apiKey: string;
  model?: string;
  baseURL?: string;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

const DEFAULT_BASE_URL = "https://api.minimax.chat/v1";
const DEFAULT_MODEL = "MiniMax-Text-01";

const SYSTEM_PROMPT = `你是 pets-agent 的知识库问答助手。你的职责是基于检索到的知识回答用户问题。

行为准则：
- 优先使用检索到的知识回答问题
- 如果检索到的知识与问题相关，基于知识给出详细、准确的回答
- 如果知识库中没有相关信息，明确告知用户"知识库中暂无相关信息"
- 不要编造或猜测知识库中不存在的内容
- 回答使用中文，技术术语可用英文
- 回答格式使用 Markdown 结构化输出`;

export class QAAgent {
  private config: Required<QAAgentConfig>;
  private retriever: MemoryRetriever;
  private history: ChatMessage[] = [];

  constructor(config: QAAgentConfig) {
    this.config = {
      apiKey: config.apiKey,
      model: config.model ?? process.env.LLM_MODEL ?? DEFAULT_MODEL,
      baseURL: config.baseURL ?? process.env.LLM_BASE_URL ?? DEFAULT_BASE_URL,
    };
    this.retriever = new MemoryRetriever();
  }

  async init(): Promise<void> {
    await this.retriever.init();
  }

  /**
   * Ask a question. Supports multi-turn conversation via internal history.
   */
  async ask(question: string): Promise<string> {
    // Retrieve relevant context from memory
    const context = this.retriever.retrieve(question);

    // Build system message with retrieved context
    const systemContent = context
      ? SYSTEM_PROMPT + "\n\n" + context
      : SYSTEM_PROMPT + "\n\n知识库中暂无与该问题相关的信息，请如实告知用户。";

    // Build messages array
    const messages: ChatMessage[] = [
      { role: "system", content: systemContent },
      ...this.history,
      { role: "user", content: question },
    ];

    // Call LLM
    const answer = await this.callLLM(messages);

    // Update history
    this.history.push({ role: "user", content: question });
    this.history.push({ role: "assistant", content: answer });

    return answer;
  }

  /**
   * List all knowledge in the memory system.
   */
  async listKnowledge(): Promise<string> {
    return this.retriever.listAll();
  }

  /**
   * Clear conversation history.
   */
  clearHistory(): void {
    this.history = [];
  }

  private async callLLM(messages: ChatMessage[]): Promise<string> {
    const MAX_RETRIES = 2;
    const BASE_DELAY_MS = 500;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 60_000);

        const res = await fetch(`${this.config.baseURL}/text/chatcompletion_v2`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.config.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: this.config.model,
            messages: messages.map((m) => ({ role: m.role, content: m.content })),
            temperature: 0.4,
            max_tokens: 4096,
          }),
          signal: controller.signal,
        });

        clearTimeout(timer);

        if (!res.ok) {
          const transient = res.status >= 500 || res.status === 429 || res.status === 408;
          if (transient && attempt < MAX_RETRIES) {
            const delay = BASE_DELAY_MS * Math.pow(2, attempt);
            await new Promise((r) => setTimeout(r, delay));
            continue;
          }
          const errText = await res.text();
          throw new Error(`LLM API error ${res.status}: ${errText}`);
        }

        const json = (await res.json()) as {
          choices?: Array<{ message?: { content?: string } }>;
        };
        return json.choices?.[0]?.message?.content ?? "";
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          return "[请求超时]";
        }
        if (err instanceof TypeError && attempt < MAX_RETRIES) {
          const delay = BASE_DELAY_MS * Math.pow(2, attempt);
          await new Promise((r) => setTimeout(r, delay));
          continue;
        }
        throw err;
      }
    }
    throw new Error("LLM call exhausted retries");
  }
}
