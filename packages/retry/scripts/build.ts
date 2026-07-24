/**
 * Package build: code-split JavaScript, reachable public types, and size gates.
 *
 * `tsc` alone is unbounded and would inline the weight artifact into the
 * compiler entry. This script instead:
 *   1. cleans `dist/`;
 *   2. lets esbuild emit a code-split, minified ESM bundle where the compiler
 *      entry (`dist/index.js`) and the lazily imported model chunks
 *      (`dist/model-*.js`) are separate;
 *   3. runs `tsc` for declarations only, then prunes the internal declarations
 *      that the public entry does not reach (notably the generated weights);
 *   4. runs the Brotli size gate, which fails the build on a breach.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

import { emitDeclarations } from "../../../scripts/declarations.js";
import { runSizeGate } from "./size-gate.ts";

const packageRoot = join(import.meta.dirname, "..");
const srcDir = join(packageRoot, "src");

// Candidate builds go to a disposable package directory without touching source
// weights or the normal distribution. Positional arguments are internal only.
const weightFile = process.argv[2]
  ? resolve(process.argv[2])
  : join(srcDir, "model", "weights.gen.ts");

const distDir = process.argv[3] ? resolve(process.argv[3]) : join(packageRoot, "dist");
const esbuildCli = join(packageRoot, "node_modules", "esbuild", "bin", "esbuild");

const javascriptFlags = [
  "--format=esm",
  "--platform=neutral",
  "--target=es2022",
  "--minify",
  "--legal-comments=none",
];

function cleanDist(): void {
  rmSync(distDir, { recursive: true, force: true });
  mkdirSync(distDir, { recursive: true });
}

/**
 * Code-split ESM. The dynamic imports in `src/model/engine.ts` become the
 * `model-*` chunks, so the weight artifact never reaches `dist/index.js`.
 */
function bundle(): void {
  execFileSync(
    process.execPath,
    [
      esbuildCli,
      join(srcDir, "index.ts"),
      `--outdir=${distDir}`,
      "--bundle",
      "--splitting",
      "--tree-shaking=true",
      "--entry-names=[name]",
      "--chunk-names=model-[hash]",
      "--external:./weights.gen.js",
      ...javascriptFlags,
    ],
    { cwd: packageRoot, stdio: "inherit" },
  );

  // All emitted JS files are siblings, so their original relative weights
  // import works unchanged. No resolver plugin or module replacement is needed.
  execFileSync(
    process.execPath,
    [esbuildCli, weightFile, `--outfile=${join(distDir, "weights.gen.js")}`, ...javascriptFlags],
    { cwd: packageRoot, stdio: "inherit" },
  );
}

function main(): void {
  cleanDist();
  bundle();
  emitDeclarations(packageRoot, distDir);
  runSizeGate(distDir);
}

main();
