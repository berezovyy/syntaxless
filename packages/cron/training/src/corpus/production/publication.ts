/** Immutable directory publication and atomic JSON state/report replacement. */
import { Database } from "bun:sqlite";
import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  createReadStream,
  existsSync,
  fsyncSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  renameSync,
  rmSync,
  statfsSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

export async function hashArtifact(path: string): Promise<string> {
  const hash = createHash("sha256");

  for await (const chunk of createReadStream(path)) hash.update(chunk);

  return hash.digest("hex");
}

export function diskPreflight(
  directory: string,
  requiredBytes: number,
  reserveBytes = 2 * 1024 ** 3,
): { availableBytes: number; requiredBytes: number; reserveBytes: number } {
  if (
    !Number.isSafeInteger(requiredBytes) ||
    requiredBytes <= 0 ||
    !Number.isSafeInteger(reserveBytes) ||
    reserveBytes < 0
  ) {
    throw new Error(
      "Disk estimates must be positive safe-integer bytes with a nonnegative reserve",
    );
  }

  const stats = statfsSync(directory, { bigint: true });
  const available = stats.bavail * stats.bsize;

  if (available < BigInt(requiredBytes) + BigInt(reserveBytes)) {
    throw new Error(
      `Insufficient disk space: ${available} available, ${requiredBytes} required plus ${reserveBytes} reserve; prior artifacts will not be deleted`,
    );
  }

  return { availableBytes: Number(available), requiredBytes, reserveBytes };
}

function syncDirectory(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

/** Replace owned JSON metadata; callers coordinate writes to the destination. */
export function replaceJson(path: string, value: unknown): void {
  const target = resolve(path);
  const payload = `${JSON.stringify(value, null, 2)}\n`;
  const parent = dirname(target);
  const temporary = join(parent, `.${basename(target)}.${randomUUID()}.tmp`);
  const fd = openSync(temporary, "wx", 0o600);

  using _temporary = {
    [Symbol.dispose]() {
      rmSync(temporary, { force: true });
    },
  };

  try {
    writeFileSync(fd, payload);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }

  renameSync(temporary, target);
  syncDirectory(parent);
}

/** Closing the connection or process releases ownership. Keep the inode:
 * unlinking a lock while another connection holds it would admit two owners. */
export function acquirePreparationLock(output: string): Disposable {
  const path = join(dirname(output), `.${basename(output)}.lock.sqlite`);
  const lock = new Database(path, { create: true });

  try {
    lock.exec("PRAGMA busy_timeout=0; BEGIN EXCLUSIVE;");

    return lock;
  } catch (error) {
    lock.close(true);
    throw new Error(`Cannot acquire production preparation lock: ${path}`, {
      cause: error,
    });
  }
}

export async function publishDirectory<T>(
  output: string,
  requiredBytes: number,
  build: (staging: string) => Promise<T>,
): Promise<T> {
  const target = resolve(output);

  if (existsSync(target)) throw new Error(`Refusing to overwrite published path: ${target}`);
  const parent = dirname(target);

  mkdirSync(parent, { recursive: true });
  diskPreflight(parent, requiredBytes);
  using _lock = acquirePreparationLock(target);

  if (existsSync(target)) {
    throw new Error(`Published path appeared while taking the lock: ${target}`);
  }

  const staging = mkdtempSync(join(parent, `.${basename(target)}.building-`));

  using _staging = {
    [Symbol.dispose]() {
      // This name is uniquely ours; after publication it no longer exists.
      rmSync(staging, { recursive: true, force: true });
    },
  };

  const value = await build(staging);

  if (!existsSync(join(staging, "manifest.json"))) {
    throw new Error("Publication requires a completed manifest.json");
  }

  if (existsSync(target)) {
    throw new Error(`Refusing to replace a path created during generation: ${target}`);
  }

  syncDirectory(staging);
  renameSync(staging, target);
  syncDirectory(parent);

  return value;
}
