import { afterAll } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const directories: string[] = [];

/** Isolated scratch space removed even when a test assertion fails. */
export function tempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));

  directories.push(directory);

  return directory;
}

afterAll(() => {
  for (const directory of directories) rmSync(directory, { recursive: true, force: true });
});
