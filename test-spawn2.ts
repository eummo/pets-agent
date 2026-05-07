import { spawn, execSync } from "child_process";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";

console.log("Testing different spawn approaches:");

// Test 1: current approach
try {
  const child = spawn(exe, ["-p", "--dangerously-skip-permissions", "echo test"], {
    cwd: "/tmp",
    stdio: ["pipe", "pipe", "pipe"]
  });
  child.on("error", (e) => console.log("Approach 1 error:", e.message));
  child.on("close", (c) => console.log("Approach 1 exit:", c));
  setTimeout(() => child.kill(), 2000);
} catch (e) {
  console.log("Approach 1 exception:", e.message);
}

// Test 2: with win32 path style
try {
  const child = spawn("C:\\Users\\jadenli\\AppData\\Roaming\\npm\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe", ["-p", "--dangerously-skip-permissions", "echo test"], {
    cwd: "/tmp",
    stdio: ["pipe", "pipe", "pipe"]
  });
  child.on("error", (e) => console.log("Approach 2 error:", e.message));
  child.on("close", (c) => console.log("Approach 2 exit:", c));
  setTimeout(() => child.kill(), 2000);
} catch (e) {
  console.log("Approach 2 exception:", e.message);
}

// Test 3: verify the exe exists and is executable
try {
  const stat = execSync(`stat "${exe}"`).toString();
  console.log("File stat:", stat.slice(0, 100));
} catch (e) {
  console.log("stat failed:", e.message);
}

// Test 4: try calling via cmd.exe
try {
  const child = spawn("cmd.exe", ["/c", exe, "-p", "--dangerously-skip-permissions", "echo test"], {
    cwd: "/tmp",
    stdio: ["pipe", "pipe", "pipe"]
  });
  child.on("error", (e) => console.log("Approach 4 error:", e.message));
  child.on("close", (c) => console.log("Approach 4 exit:", c));
  setTimeout(() => child.kill(), 2000);
} catch (e) {
  console.log("Approach 4 exception:", e.message);
}
