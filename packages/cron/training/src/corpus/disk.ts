/** Free-space preflight and atomic JSON replacement for run artifacts. */
import { randomUUID } from "node:crypto";
import {
  closeSync,
  fsyncSync,
  openSync,
  renameSync,
  rmSync,
  statfsSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

/** Throw unless `directory` has `requiredBytes` plus a reserve available. */
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
      `Insufficient disk space: ${available} available, ${requiredBytes} required ` +
        `plus ${reserveBytes} reserve; prior artifacts will not be deleted`,
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
