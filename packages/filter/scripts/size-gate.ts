/** Measure every emitted JavaScript chunk, including the lazy model payload. */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { brotliCompressSync, constants } from "node:zlib";

const dist = join(import.meta.dir, "../dist");
const RUNTIME_BUDGET = 20 * 1024;
const MODEL_BUDGET = 64 * 1024;

function files(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);

    return entry.isDirectory() ? files(path) : entry.name.endsWith(".js") ? [path] : [];
  });
}

export function brotliSize(file: string): number {
  return brotliCompressSync(readFileSync(file), {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
  }).length;
}

export function runSizeGate(directory = dist): {
  runtimeBytes: number;
  modelBytes: number;
} {
  let runtimeBytes = 0;
  let modelBytes = 0;

  for (const path of files(directory)) {
    const bytes = brotliSize(path);

    if (relative(directory, path) === "weights.gen.js") modelBytes += bytes;
    else runtimeBytes += bytes;
  }

  const runtimeUsage = `${runtimeBytes}/${RUNTIME_BUDGET} B`;
  const modelUsage = `${modelBytes}/${MODEL_BUDGET} B`;

  console.log(`size:gate aggregate runtime ${runtimeUsage}; model ${modelUsage}`);

  if (runtimeBytes > RUNTIME_BUDGET || modelBytes > MODEL_BUDGET) {
    throw new Error("aggregate package size budget exceeded");
  }

  return { runtimeBytes, modelBytes };
}

if (import.meta.main) runSizeGate();
