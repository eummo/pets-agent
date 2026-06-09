import Database from "better-sqlite3";
import { constants } from "node:fs";
import { access, copyFile, mkdir, rm, stat } from "node:fs/promises";
import path from "node:path";

export type SqliteBackupResult = {
  readonly sourcePath: string;
  readonly backupPath: string;
  readonly totalPages: number;
  readonly remainingPages: number;
  readonly integrityCheck: "ok";
};

export type SqliteRestoreResult = {
  readonly backupPath: string;
  readonly targetPath: string;
  readonly integrityCheck: "ok";
};

export async function backupSqliteDatabase(options: {
  readonly sourcePath: string;
  readonly backupPath: string;
}): Promise<SqliteBackupResult> {
  const sourcePath = path.resolve(options.sourcePath);
  const backupPath = path.resolve(options.backupPath);
  assertDifferentPaths(sourcePath, backupPath, "backup destination");
  await assertReadableFile(sourcePath, "SQLite source");
  await mkdir(path.dirname(backupPath), { recursive: true });

  const db = new Database(sourcePath, { readonly: true, fileMustExist: true });
  try {
    const metadata = await db.backup(backupPath);
    const integrityCheck = await verifySqliteDatabase(backupPath);
    return {
      sourcePath,
      backupPath,
      totalPages: metadata.totalPages,
      remainingPages: metadata.remainingPages,
      integrityCheck
    };
  } finally {
    db.close();
  }
}

export async function restoreSqliteDatabase(options: {
  readonly backupPath: string;
  readonly targetPath: string;
  readonly force?: boolean;
}): Promise<SqliteRestoreResult> {
  const backupPath = path.resolve(options.backupPath);
  const targetPath = path.resolve(options.targetPath);
  assertDifferentPaths(backupPath, targetPath, "restore target");
  await assertReadableFile(backupPath, "SQLite backup");
  const integrityCheck = await verifySqliteDatabase(backupPath);

  if (!options.force && (await fileExists(targetPath))) {
    throw new Error(
      `Refusing to overwrite existing SQLite database without --force: ${targetPath}`
    );
  }

  await mkdir(path.dirname(targetPath), { recursive: true });
  await removeSqliteSidecarFiles(targetPath);
  await copyFile(backupPath, targetPath);
  await removeSqliteSidecarFiles(targetPath);
  await verifySqliteDatabase(targetPath);
  return { backupPath, targetPath, integrityCheck };
}

export async function verifySqliteDatabase(dbPath: string): Promise<"ok"> {
  const resolvedPath = path.resolve(dbPath);
  await assertReadableFile(resolvedPath, "SQLite database");
  const db = new Database(resolvedPath, { readonly: true, fileMustExist: true });
  try {
    let row: { readonly integrity_check: string } | undefined;
    try {
      row = db.prepare("PRAGMA integrity_check").get() as
        | { readonly integrity_check: string }
        | undefined;
    } catch (error) {
      throw new Error(
        `SQLite integrity check failed for ${resolvedPath}: ${
          error instanceof Error ? error.message : String(error)
        }`,
        { cause: error }
      );
    }
    if (row?.integrity_check !== "ok") {
      throw new Error(
        `SQLite integrity check failed for ${resolvedPath}: ${row?.integrity_check ?? "missing result"}`
      );
    }
    return "ok";
  } finally {
    db.close();
  }
}

async function assertReadableFile(filePath: string, label: string): Promise<void> {
  try {
    await access(filePath, constants.R_OK);
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) {
      throw new Error(`${label} is not a file: ${filePath}`);
    }
  } catch (error) {
    throw new Error(
      `${label} is not readable at ${filePath}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error }
    );
  }
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function removeSqliteSidecarFiles(dbPath: string): Promise<void> {
  await Promise.all([rm(`${dbPath}-wal`, { force: true }), rm(`${dbPath}-shm`, { force: true })]);
}

function assertDifferentPaths(left: string, right: string, label: string): void {
  if (left === right) {
    throw new Error(`SQLite source and ${label} must be different paths: ${left}`);
  }
}
