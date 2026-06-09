import {
  backupSqliteDatabase,
  restoreSqliteDatabase,
  verifySqliteDatabase
} from "./sqliteMaintenance.js";

type Command = "backup" | "restore" | "verify";

type ParsedArgs =
  | {
      readonly command: "backup";
      readonly dbPath: string;
      readonly backupPath: string;
    }
  | {
      readonly command: "restore";
      readonly dbPath: string;
      readonly backupPath: string;
      readonly force: boolean;
    }
  | {
      readonly command: "verify";
      readonly dbPath: string;
    };

async function main(argv: readonly string[]): Promise<void> {
  const args = parseArgs(argv);
  if (args.command === "backup") {
    const result = await backupSqliteDatabase({
      sourcePath: args.dbPath,
      backupPath: args.backupPath
    });
    console.info(
      `SQLite backup written: ${result.backupPath} (${result.totalPages} pages, integrity=${result.integrityCheck})`
    );
    return;
  }

  if (args.command === "restore") {
    const result = await restoreSqliteDatabase({
      backupPath: args.backupPath,
      targetPath: args.dbPath,
      force: args.force
    });
    console.info(
      `SQLite backup restored: ${result.targetPath} (integrity=${result.integrityCheck})`
    );
    return;
  }

  const integrityCheck = await verifySqliteDatabase(args.dbPath);
  console.info(`SQLite database verified: ${args.dbPath} (integrity=${integrityCheck})`);
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  const command = argv[0] as Command | undefined;
  if (command !== "backup" && command !== "restore" && command !== "verify") {
    throw new Error(usage());
  }

  const values = new Map<string, string>();
  let force = false;
  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--force") {
      force = true;
      continue;
    }
    if (arg === "--db" || arg === "--out" || arg === "--backup") {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${arg} requires a path value.`);
      }
      values.set(arg, value);
      index += 1;
      continue;
    }
    throw new Error(`Unknown SQLite maintenance argument: ${arg ?? ""}\n${usage()}`);
  }

  const dbPath = values.get("--db");
  if (dbPath === undefined) {
    throw new Error(`--db is required.\n${usage()}`);
  }

  if (command === "backup") {
    const backupPath = values.get("--out");
    if (backupPath === undefined) {
      throw new Error(`--out is required for backup.\n${usage()}`);
    }
    return { command, dbPath, backupPath };
  }

  if (command === "restore") {
    const backupPath = values.get("--backup");
    if (backupPath === undefined) {
      throw new Error(`--backup is required for restore.\n${usage()}`);
    }
    return { command, dbPath, backupPath, force };
  }

  return { command, dbPath };
}

function usage(): string {
  return [
    "Usage:",
    "  npm run db:backup -- --db <path> --out <backup.db>",
    "  npm run db:restore -- --backup <backup.db> --db <path> --force",
    "  npm run db:verify -- --db <path>"
  ].join("\n");
}

try {
  await main(process.argv.slice(2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
