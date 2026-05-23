import { spawn } from "node:child_process";
import path from "node:path";
import { removeServerPidFile, writeServerPidFile } from "./serverPid.js";

const serverPidPath = process.env["SERVER_PID_PATH"] ?? path.resolve(".harness", "state", "server.pid");
const tsxCliPath = path.resolve("node_modules", "tsx", "dist", "cli.mjs");
const signalExitCodes: Partial<Record<NodeJS.Signals, number>> = {
  SIGINT: 130,
  SIGTERM: 143,
};

await writeServerPidFile(serverPidPath);

const serverProcess = spawn(process.execPath, [tsxCliPath, "src/index.ts"], {
  cwd: process.cwd(),
  env: process.env,
  stdio: "inherit",
});

let shuttingDown = false;

serverProcess.once("exit", (code, signal) => {
  void removeServerPidFile(serverPidPath).finally(() => {
    process.exit(code ?? signalExitCodes[signal ?? "SIGTERM"] ?? 1);
  });
});

process.once("SIGINT", () => {
  stopServer("SIGINT");
});
process.once("SIGTERM", () => {
  stopServer("SIGTERM");
});
process.once("exit", () => {
  void removeServerPidFile(serverPidPath);
});

function stopServer(signal: NodeJS.Signals): void {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  if (serverProcess.exitCode === null && serverProcess.signalCode === null) {
    serverProcess.kill(signal);
    return;
  }

  void removeServerPidFile(serverPidPath).finally(() => {
    process.exit(signalExitCodes[signal] ?? 1);
  });
}
