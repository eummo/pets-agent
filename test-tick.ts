import { spawn } from "child_process";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";

console.log("cwd:", process.cwd());
console.log("exe:", exe);

const child = spawn(exe,
  ["-p", "--dangerously-skip-permissions", "--bare", "echo TICTAC"],
  { cwd: "/tmp", stdio: ["pipe", "pipe", "pipe"] }
);

const timer = setInterval(() => console.log("tick"), 2000);
child.on("close", (c) => { clearInterval(timer); console.log("exit:", c); process.exit(0); });
child.on("error", (e) => { clearInterval(timer); console.log("error:", e.message); process.exit(1); });
