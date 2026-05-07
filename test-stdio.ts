import { spawn } from "child_process";
import * as fs from "fs";

// Test: does tsx foreground affect stdin inheritance?
// When running `npx tsx test.ts` in foreground, stdin stays open.
// Try explicitly closing stdin in child process

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";

// This is what agent-manager does - but in a tsx foreground session stdin is inherited
// Let's see if explicitly ignoring stdin helps
const child = spawn(exe,
  ["-p", "--dangerously-skip-permissions", "--bare", "echo TSX_TEST"],
  {
    cwd: "/tmp",
    // Try different stdio configs
    stdio: ["ignore", "pipe", "pipe"],
    // Also test: explicitly set stdin to /dev/null via env
    env: { ...process.env }
  }
);

let out = "", err = "";
child.stdout.on("data", (d) => { out += d.toString(); console.log("out:", d.toString().trim()); });
child.stderr.on("data", (d) => { err += d.toString(); console.log("err:", d.toString().trim()); });
child.on("close", (c) => { console.log("exit:", c); process.exit(0); });
child.on("error", (e) => { console.log("error:", e.message); process.exit(1); });
