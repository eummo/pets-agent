import { spawn } from "child_process";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";
const prompt = "echo PWD_OK and exit immediately";

// Use bare mode and check output after 5s
const child = spawn(exe,
  ["-p", "--dangerously-skip-permissions", "--no-session-persistence", "--bare", prompt],
  { cwd: "/tmp", stdio: ["pipe", "pipe", "pipe"] }
);

let out = "", err = "";
const start = Date.now();
child.stdout.on("data", (d) => { const s = d.toString(); console.log("stdout chunk:", JSON.stringify(s)); out += s; });
child.stderr.on("data", (d) => { const s = d.toString(); console.log("stderr chunk:", JSON.stringify(s)); err += s; });
child.on("error", (e) => { console.log("SPAWN ERROR:", e.message); });
child.on("close", (c) => { console.log(`close after ${Date.now()-start}ms, code: ${c}`); console.log("final out:", JSON.stringify(out)); console.log("final err:", JSON.stringify(err)); });

// Just exit after 6s
setTimeout(() => {
  console.log("6s reached, killing...");
  child.kill();
  process.exit(0);
}, 6000);
