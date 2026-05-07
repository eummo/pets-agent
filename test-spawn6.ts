import { spawn } from "child_process";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";

// Test just the exe from /tmp
const child = spawn(exe,
  ["-p", "--dangerously-skip-permissions", "--bare", "echo FROM_TMP"],
  { cwd: "/tmp", stdio: ["pipe", "pipe", "pipe"] }
);

const t = setInterval(() => { console.log("still alive..."); }, 2000);
child.on("close", (c) => {
  clearInterval(t);
  console.log("closed:", c);
  process.exit(0);
});
child.on("error", (e) => { clearInterval(t); console.log("error:", e.message); process.exit(1); });
