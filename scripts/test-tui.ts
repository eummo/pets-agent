#!/usr/bin/env npx tsx
/**
 * pets-agent TUI 交互测试
 * 用法: npx tsx scripts/test-tui.ts
 * 或: npx tsx scripts/test-tui.ts --agent claude-code
 */
import { spawn, execSync } from 'child_process';
import { openpty } from '@pipeline/sual';

const SLEEP = (ms: number) => new Promise(r => setTimeout(r, ms));

async function run() {
  const cmd = 'npx';
  const args = ['tsx', 'src/index.ts'];
  const pty = openpty(cmd, args);

  let output = '';
  pty.onData((data: string) => {
    output += data;
    process.stdout.write(data);
  });

  await SLEEP(500);

  // 测试 slash commands
  const tests = [
    { cmd: '/help\n', expect: '可用命令', label: '/help' },
    { cmd: '/tasks\n', expect: '任务', label: '/tasks' },
    { cmd: '/history\n', expect: '历史', label: '/history' },
    { cmd: '/clear\n', expect: '', label: '/clear (no output)' },
    { cmd: '/quit\n', expect: '', label: '/quit' },
  ];

  for (const t of tests) {
    pty.write(t.cmd);
    await SLEEP(300);
  }

  // 测试用户输入
  pty.write('你好\n');
  await SLEEP(3000);
  pty.write('/quit\n');
  await SLEEP(500);
  pty.destroy();

  console.log('\n=== OUTPUT ===\n' + output);
}

run().catch(console.error);
