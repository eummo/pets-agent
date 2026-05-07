import { agentManager } from "./src/tasks/agent-manager.js";
import * as fs from "fs";

const logFile = "/tmp/llog";
const log = (m: string) => {
  fs.appendFileSync(logFile, m + "\n");
  console.error(m);
};

const t = agentManager.spawn("claude-code", "回复四个字：成功", { name: "test", workdir: "/tmp" });
log("spawned: " + t.id);

setTimeout(() => {
  const x = agentManager.get(t.id);
  log("done: " + x?.status + " lines:" + (x?.progress.length ?? 0));
  if (x?.progress.length) log("last: " + x.progress[x.progress.length - 1].slice(0, 200));
  process.exit(0);
}, 15000);
