#!/usr/bin/env node
/**
 * QA Agent CLI — query the pets-agent knowledge base.
 *
 * Usage:
 *   npx tsx src/qa-agent/cli.ts "问题"     # 单次问答
 *   npx tsx src/qa-agent/cli.ts -i          # 交互式 REPL
 *   npx tsx src/qa-agent/cli.ts --list      # 列出所有知识概览
 */

import * as readline from "readline";
import { QAAgent } from "./qa-agent.js";

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    printUsage();
    process.exit(0);
  }

  const apiKey = process.env.MINIMAX_API_KEY ?? process.env.MINIMAX_KEY;
  if (!apiKey) {
    console.error("错误：请设置 MINIMAX_API_KEY 环境变量");
    process.exit(1);
  }

  const agent = new QAAgent({ apiKey });

  console.log("正在加载知识库...");
  await agent.init();
  console.log("知识库加载完成。\n");

  // --list: show all knowledge
  if (args[0] === "--list") {
    const overview = await agent.listKnowledge();
    console.log(overview);
    return;
  }

  // -i: interactive REPL
  if (args[0] === "-i" || args[0] === "--interactive") {
    await repl(agent);
    return;
  }

  // One-shot query
  const question = args.join(" ");
  const answer = await agent.ask(question);
  console.log(answer);
}

async function repl(agent: QAAgent): Promise<void> {
  console.log("pets-agent 知识库问答 (输入问题后按回车，输入 exit 退出，输入 clear 清除对话历史)");
  console.log("---");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (): Promise<string> =>
    new Promise((resolve) => {
      rl.question("你: ", (answer) => resolve(answer));
    });

  while (true) {
    const input = (await prompt()).trim();

    if (!input) continue;

    if (input === "exit" || input === "quit") {
      console.log("再见！");
      rl.close();
      break;
    }

    if (input === "clear") {
      agent.clearHistory();
      console.log("对话历史已清除。\n");
      continue;
    }

    try {
      const answer = await agent.ask(input);
      console.log(`\n助手: ${answer}\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`\n错误: ${msg}\n`);
    }
  }
}

function printUsage(): void {
  console.log(`pets-agent 知识库问答

用法:
  npx tsx src/qa-agent/cli.ts "问题"     单次问答
  npx tsx src/qa-agent/cli.ts -i          交互式 REPL
  npx tsx src/qa-agent/cli.ts --list      列出所有知识概览

环境变量:
  MINIMAX_API_KEY   MiniMax API 密钥（必需）
  LLM_MODEL         模型名称（默认: MiniMax-Text-01）
  LLM_BASE_URL      API 基础 URL（默认: https://api.minimax.chat/v1）`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
