/** Pin the complete local generation/validation implementation, not selected files. */
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

export function fileHash(path: string | URL): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function corpusCodeHashes(): Record<string, string> {
  const cron = fileURLToPath(new URL("../../../../", import.meta.url));
  const result: Record<string, string> = {};

  function walk(path: string): void {
    for (const entry of readdirSync(path, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name, "en"),
    )) {
      const file = join(path, entry.name);

      if (entry.isDirectory()) {
        walk(file);
      } else if (entry.isFile() && entry.name.endsWith(".ts") && !entry.name.endsWith(".test.ts")) {
        result[relative(cron, file)] = fileHash(file);
      }
    }
  }

  walk(join(cron, "src"));
  walk(join(cron, "training/src"));

  for (const path of [
    join(cron, "package.json"),
    join(cron, "training/package.json"),
    join(cron, "../../package.json"),
    join(cron, "../../bun.lock"),
  ]) {
    result[relative(cron, path)] = fileHash(path);
  }

  return result;
}
