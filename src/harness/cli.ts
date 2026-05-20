import path from "node:path";
import { createHarnessEnvironment } from "./createHarness.js";

type ParsedArgs = {
  readonly root: string;
  readonly reset: boolean;
};

function parseArgs(argv: readonly string[]): ParsedArgs {
  let root = ".harness";
  let reset = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--reset") {
      reset = true;
      continue;
    }
    if (arg === "--root") {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error("--root requires a path value.");
      }
      root = value;
      index += 1;
      continue;
    }
    if (arg !== undefined) {
      throw new Error(`Unknown harness argument: ${arg}`);
    }
  }

  return { root, reset };
}

const args = parseArgs(process.argv.slice(2));
const environment = await createHarnessEnvironment({
  root: args.root,
  reset: args.reset
});

console.info("Harness environment ready:");
console.info(`- root: ${environment.root}`);
console.info(`- knowledge base: ${environment.knowledgeBasePath}`);
console.info("- repositories:");
for (const repository of environment.repositories) {
  console.info(`  - ${repository.name}: ${path.join(environment.root, repository.relativePath)}`);
}
