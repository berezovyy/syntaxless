/** Pin a producer's local imports and the locked execution/consumption contract. */
import { createHash } from "node:crypto";
import { closeSync, openSync, readFileSync, readSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

/** SHA-256 of a file through one fixed buffer; inputs can be corpus-sized. */
export function fileHash(path: string | URL): string {
  const hash = createHash("sha256");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  const fd = openSync(path, "r");

  try {
    for (;;) {
      const read = readSync(fd, buffer, 0, buffer.length, null);

      if (read === 0) break;
      hash.update(buffer.subarray(0, read));
    }
  } finally {
    closeSync(fd);
  }

  return hash.digest("hex");
}

export function corpusCodeHashes(entryUrl: string): Record<string, string> {
  const cron = fileURLToPath(new URL("../../../", import.meta.url));
  const result: Record<string, string> = {};
  const visited = new Set<string>();
  const scanner = new Bun.Transpiler({ loader: "ts" });

  function visit(file: string): void {
    if (visited.has(file)) return;
    visited.add(file);
    const source = readFileSync(file);

    result[relative(cron, file)] = createHash("sha256").update(source).digest("hex");
    if (!/\.[cm]?[jt]sx?$/.test(file)) return;

    // Scan without executing imports. Bun's resolver also handles .js -> .ts
    // source imports. Local data imports are hashed without parsing them.
    for (const item of scanner.scan(source.toString("utf8")).imports) {
      if (item.path.startsWith(".")) visit(Bun.resolveSync(item.path, dirname(file)));
    }
  }

  visit(fileURLToPath(entryUrl));
  result["$bun.version"] = createHash("sha256").update(Bun.version).digest("hex");

  // Bare package imports are pinned by the lockfiles. The Python shard loader
  // is pinned too because it defines how the written features are consumed.
  for (const path of [
    join(cron, "package.json"),
    join(cron, "training/package.json"),
    join(cron, "../../package.json"),
    join(cron, "../../bun.lock"),
    join(cron, "../../training/cron/dataset.py"),
    join(cron, "../../training/cron/uv.lock"),
  ]) {
    result[relative(cron, path)] = fileHash(path);
  }

  return Object.fromEntries(
    Object.entries(result).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
  );
}
