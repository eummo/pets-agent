import { spawn } from "child_process";

const exe = "/mnt/c/Users/jadenli/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe";
console.log("exe:", exe);
console.log("cwd:", process.cwd());

const child = spawn(exe,
  ["-p", "--dangerously-skip-permissions", "--bare", "echo PWD_TEST_OK"],
  { cwd: "/tmp", stdio: ["pipe", "pipe", "pipe"] }
);

let out = "", err = "";
child.stdout.on("data", (d) => { out += d.toString(); });
child.stderr.on("data", (d) => { err += d.toString(); });
child.on("error", (e) => { console.log("SPAWN ERROR:", e.message); });
child.on("close", (c) => {
  console.log("exit:", c);
  console.log("stdout:", JSON.stringify(out));
  console.log("stderr:", JSON.stringify(err));
});
