import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

/**
 * Package build (Phase 21).
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
import { build } from "esbuild";

import { runSizeGate } from "./size-gate.ts";

const packageRoot = join(import.meta.dirname, "..");
const srcDir = join(packageRoot, "src");
const distDir = join(packageRoot, "dist");

function cleanDist(): void {
  rmSync(distDir, { recursive: true, force: true });
  mkdirSync(distDir, { recursive: true });
}

/**
 * Code-split ESM. The dynamic imports in `src/model/engine.ts` become the
 * `model-*` chunks, so the weight artifact never reaches `dist/index.js`.
 */
async function bundle(): Promise<void> {
  await build({
    entryPoints: { index: join(srcDir, "index.ts") },
    outdir: distDir,
    bundle: true,
    splitting: true,
    format: "esm",
    platform: "neutral",
    target: "es2022",
    minify: true,
    legalComments: "none",
    treeShaking: true,
    entryNames: "[name]",
    chunkNames: "model-[hash]",
    write: true,
  });
}

/** Emit declarations only, so tsc cannot clobber the esbuild output. */
function emitDeclarations(): void {
  const tsc = join(packageRoot, "node_modules", "typescript", "bin", "tsc");

  execFileSync(
    process.execPath,
    [
      tsc,
      "-p",
      "tsconfig.build.json",
      "--emitDeclarationOnly",
      "--declarationMap",
      "false",
      "--sourceMap",
      "false",
    ],
    { cwd: packageRoot, stdio: "inherit" },
  );
}

function walkDeclarations(dir: string): string[] {
  const files: string[] = [];

  for (const name of readdirSync(dir)) {
    const path = join(dir, name);

    if (statSync(path).isDirectory()) files.push(...walkDeclarations(path));
    else if (name.endsWith(".d.ts")) files.push(path);
  }

  return files;
}

/**
 * Keep only the declarations transitively reachable from the public
 * `dist/index.d.ts`. Internal modules (the generated weights, the bytes
 * decoder, the asset policy, compiler validation) are dropped.
 */
function pruneInternalDeclarations(): void {
  const declarations = walkDeclarations(distDir);
  const reachable = new Set<string>();
  const stack = [join(distDir, "index.d.ts")];

  while (stack.length > 0) {
    const file = stack.pop()!;

    if (reachable.has(file)) continue;
    reachable.add(file);
    const source = readFileSync(file, "utf8");

    for (const match of source.matchAll(/(?:from\s*|import\(\s*)["']([^"']+)["']/g)) {
      const specifier = match[1]!;

      if (!specifier.startsWith(".") || !specifier.endsWith(".js")) continue;
      const target = resolve(dirname(file), `${specifier.slice(0, -3)}.d.ts`);

      if (declarations.includes(target)) stack.push(target);
    }
  }

  const pruned = declarations.filter((file) => !reachable.has(file));

  for (const file of pruned) rmSync(file);
  console.log(`build: pruned ${pruned.length} internal declaration file(s)`);
}

async function main(): Promise<void> {
  cleanDist();
  await bundle();
  emitDeclarations();
  pruneInternalDeclarations();
  runSizeGate();
}

await main();
