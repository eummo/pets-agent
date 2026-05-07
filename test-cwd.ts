import { spawn } from "child_process";
import * as fs from "fs";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";
console.log("exe exists:", fs.existsSync(exe));

spawn(exe, ["-p", "--dangerously-skip-permissions", "--bare", "echo OK"],
  { cwd: "/tmp/todo-app", stdio: ["ignore", "pipe", "pipe"] }
).on("error", (e) => console.log("error with non-existent cwd:", e.message))
 .on("close", (c) => console.log("close:", c));

setTimeout(() => process.exit(0), 2000);
