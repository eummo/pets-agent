import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const STOP_TIMEOUT_MS = 5_000;
const STOP_POLL_INTERVAL_MS = 50;

export async function writeServerPidFile(pidFilePath: string, pid = process.pid): Promise<void> {
  await mkdir(path.dirname(pidFilePath), { recursive: true });
  await writeFile(pidFilePath, `${pid}\n`, "utf8");
}

export async function removeServerPidFile(pidFilePath: string): Promise<void> {
  await rm(pidFilePath, { force: true });
}

export async function stopServerFromPidFile(pidFilePath: string): Promise<void> {
  const pid = await readPidFile(pidFilePath);
  if (pid === undefined) {
    return;
  }

  if (!isProcessAlive(pid)) {
    await removeServerPidFile(pidFilePath);
    return;
  }

  if (pid === process.pid) {
    throw new Error(`Refusing to stop the current process from PID file: ${pidFilePath}`);
  }

  process.kill(pid, "SIGTERM");
  const stopped = await waitForProcessExit(pid, STOP_TIMEOUT_MS);

  if (!stopped) {
    throw new Error(`Timed out waiting for server process ${pid} from ${pidFilePath} to stop.`);
  }

  await removeServerPidFile(pidFilePath);
}

async function readPidFile(pidFilePath: string): Promise<number | undefined> {
  let content: string;
  try {
    content = await readFile(pidFilePath, "utf8");
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }

  const pid = Number.parseInt(content.trim(), 10);
  if (!Number.isInteger(pid) || pid <= 0) {
    await removeServerPidFile(pidFilePath);
    return undefined;
  }

  return pid;
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitForProcessExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (!isProcessAlive(pid)) {
      return true;
    }
    await sleep(STOP_POLL_INTERVAL_MS);
  }

  return !isProcessAlive(pid);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
