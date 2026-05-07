import { spawn } from "child_process";

// Test 1: WSL path with real cwd
console.log("=== Test 1: WSL path from ~/code/pets-agent ===");
spawn("/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe",
  ["-p", "--dangerously-skip-permissions", "--bare", "echo TEST1"],
  { stdio: ["pipe", "pipe", "pipe"] }
).on("close", (c) => console.log("exit:", c));

// Test 2: same from /tmp
spawn("/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe",
  ["-p", "--dangerously-skip-permissions", "--bare", "echo TEST2"],
  { cwd: "/tmp", stdio: ["pipe", "pipe", "pipe"] }
).on("close", (c) => console.log("exit:", c));

// Test 3: cmd.exe wrapper
spawn("cmd.exe", ["/c",
  "C:\\Users\\jadenli\\AppData\\Roaming\\npm\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe",
  "-p", "--dangerously-skip-permissions", "--bare", "echo TEST3"
], { stdio: ["pipe", "pipe", "pipe"] }).on("close", (c) => console.log("cmd.exe exit:", c));

// Test 4: cmd.exe with WSL path
spawn("cmd.exe", ["/c",
  "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe",
  "-p", "--dangerously-skip-permissions", "--bare", "echo TEST4"
], { stdio: ["pipe", "pipe", "pipe"] }).on("close", (c) => console.log("cmd.exe WSL path exit:", c));

setTimeout(() => {
  console.log("Done waiting");
  process.exit(0);
}, 5000);
