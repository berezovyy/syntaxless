/** Same-filesystem staged publication: an existing version is never modified. */
import { createHash } from "node:crypto";
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
  const lock = join(parent, `.${basename(target)}.lock`);
  // O_EXCL prevents cooperating generators from publishing the same version.
  const lockFd = openSync(lock, "wx", 0o600);
  let staging: string | undefined;

  try {
    writeFileSync(lockFd, JSON.stringify({ pid: process.pid, target }));
    fsyncSync(lockFd);

    if (existsSync(target)) {
      throw new Error(`Published path appeared while taking the lock: ${target}`);
    }

    staging = mkdtempSync(join(parent, `.${basename(target)}.building-`));
    const value = await build(staging);

    if (!existsSync(join(staging, "manifest.json"))) {
      throw new Error("Publication requires a completed manifest.json");
    }

    if (existsSync(target)) {
      throw new Error(`Refusing to replace a path created during generation: ${target}`);
    }

    syncDirectory(staging);
    renameSync(staging, target);
    staging = undefined;
    syncDirectory(parent);

    return value;
  } finally {
    // Only our uniquely allocated staging directory is removed on failure.
    if (staging) rmSync(staging, { recursive: true, force: true });
    closeSync(lockFd);
    rmSync(lock);
  }
}
