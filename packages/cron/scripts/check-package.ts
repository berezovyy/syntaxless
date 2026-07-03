/**
 * Packaged-artifact check (Phase 21).
 *
 * `bun pm pack` is the source of truth for what ships. This script packs the
 * workspace into a temp directory, asserts the exact file list (rejecting extra
 * files, not only missing ones), proves the weight artifact stays out of
 * `dist/index.js`, then installs the tarball into a throwaway consumer and
 * exercises `compile()` (synchronous, no initialization) and `parse()` from the
 * packed runtime. A version-mismatch must remain a hard error in the bundle.
 */
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

const packageRoot = join(import.meta.dirname, "..");
const tscBinary = join(packageRoot, "node_modules", "typescript", "bin", "tsc");

interface PackageManifest {
  files: string[];
}

const manifest = JSON.parse(
  readFileSync(join(packageRoot, "package.json"), "utf8"),
) as PackageManifest;

function walk(dir: string): string[] {
  const files: string[] = [];

  for (const name of readdirSync(dir)) {
    const path = join(dir, name);

    if (statSync(path).isDirectory()) files.push(...walk(path));
    else files.push(path);
  }

  return files;
}

/** Expected shipped paths (tarball-relative), derived from disk + `files`. */
function expectedPackedFiles(): Set<string> {
  const expected = new Set<string>(["package/package.json"]);

  for (const entry of manifest.files) {
    const path = join(packageRoot, entry);

    if (!existsSync(path)) throw new Error(`check:package: files entry missing: ${entry}`);

    if (statSync(path).isDirectory()) {
      for (const file of walk(path)) expected.add(`package/${relative(packageRoot, file)}`);
    } else {
      expected.add(`package/${entry}`);
    }
  }

  return expected;
}

function assertExactPackedFiles(tarball: string): void {
  const actual = new Set(
    execFileSync("tar", ["-tzf", tarball], { encoding: "utf8" })
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.endsWith("/")),
  );

  const expected = expectedPackedFiles();
  const missing = [...expected].filter((file) => !actual.has(file)).sort();
  const extra = [...actual].filter((file) => !expected.has(file)).sort();

  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      [
        "check:package: packed file list mismatch",
        ...missing.map((file) => `  missing: ${file}`),
        ...extra.map((file) => `  unexpected: ${file}`),
      ].join("\n"),
    );
  }

  console.log(`check:package: ${actual.size} packed files verified`);
}

/** The packed entry/chunks must never inline the replaceable weight artifact. */
function assertWeightsIsolated(extractDir: string): void {
  const weightsSource = readFileSync(join(packageRoot, "src", "model", "weights.gen.ts"), "utf8");
  const q = /"q":\s*"([^"]+)"/.exec(weightsSource)?.[1];

  if (q === undefined) throw new Error("check:package: could not locate the weight payload");
  const needle = q.slice(0, 96);

  const distDir = join(extractDir, "package", "dist");
  const indexSource = readFileSync(join(distDir, "index.js"), "utf8");

  if (indexSource.includes(needle)) {
    throw new Error("check:package: dist/index.js inlines the weight string");
  }

  const modelChunks = readdirSync(distDir).filter((name) => /^model-.*\.js$/.test(name));

  const holders = modelChunks.filter((name) =>
    readFileSync(join(distDir, name), "utf8").includes(needle),
  );

  if (holders.length > 0) {
    throw new Error(`check:package: ${holders.join(", ")} unexpectedly inline the weight string`);
  }

  const dedicated = join(distDir, "model", "weights.gen.js");

  if (!readFileSync(dedicated, "utf8").includes(needle)) {
    throw new Error("check:package: dedicated weights module lacks the payload");
  }

  console.log("check:package: weights isolated in dist/model/weights.gen.js");
}

const CONSUMER = `import { compile, getEngine, parse } from "@syntaxless/cron";

const weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"] as const;
const compiled = compile({
  frequency: { unit: "week", interval: 1 },
  weekdays: [...weekdays],
  time: { hour: 9, minute: 0 },
});
if (typeof (compiled as { then?: unknown }).then === "function") {
  throw new Error("compile must be synchronous");
}
if (!compiled.supported || compiled.expression !== "0 9 * * 1-5") {
  throw new Error("compile mismatch: " + JSON.stringify(compiled));
}

const parsed = await parse("every day");
if (!parsed.ok) throw new Error("packed parse failed: " + parsed.reason);
if (parsed.intent.frequency.unit !== "day" || parsed.intent.frequency.interval !== 1) {
  throw new Error("packed parse intent mismatch: " + JSON.stringify(parsed.intent));
}

let rejected = false;
try {
  await getEngine({ weights: new TextEncoder().encode(JSON.stringify({ version: 2 })) });
} catch (error) {
  rejected = (error as Error).name === "unsupported-weights-version";
}
if (!rejected) throw new Error("packed runtime silently accepted an unsupported weights version");

console.log("check:package: consumer runtime ok");
`;

const CONSUMER_TSCONFIG = {
  compilerOptions: {
    target: "ES2023",
    module: "ESNext",
    moduleResolution: "Bundler",
    lib: ["ES2023", "DOM", "DOM.Iterable"],
    strict: true,
    noEmit: true,
    skipLibCheck: true,
    types: [],
  },
  include: ["consumer.ts"],
};

function runConsumer(tarball: string, root: string): void {
  const consumerDir = join(root, "consumer");

  mkdirSync(consumerDir);

  writeFileSync(
    join(consumerDir, "package.json"),
    `${JSON.stringify({ name: "cron-consumer", private: true, type: "module" }, null, 2)}\n`,
  );

  writeFileSync(join(consumerDir, "consumer.ts"), CONSUMER);

  writeFileSync(
    join(consumerDir, "tsconfig.json"),
    `${JSON.stringify(CONSUMER_TSCONFIG, null, 2)}\n`,
  );

  execFileSync("bun", ["add", tarball], { cwd: consumerDir, stdio: "inherit" });
  execFileSync("bun", ["run", "consumer.ts"], { cwd: consumerDir, stdio: "inherit" });

  execFileSync(process.execPath, [tscBinary, "-p", "tsconfig.json"], {
    cwd: consumerDir,
    stdio: "inherit",
  });
}

function main(): void {
  const root = mkdtempSync(join(tmpdir(), "syntaxless-pack-"));

  try {
    const packed = execFileSync(
      "bun",
      ["pm", "pack", "--destination", root, "--ignore-scripts", "--quiet"],
      { cwd: packageRoot, encoding: "utf8" },
    );

    const tarball = packed
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .pop();

    if (tarball === undefined) throw new Error("check:package: bun pm pack produced no tarball");

    assertExactPackedFiles(tarball);

    const extractDir = join(root, "extracted");

    mkdirSync(extractDir);
    execFileSync("tar", ["-xzf", tarball, "-C", extractDir]);
    assertWeightsIsolated(extractDir);

    runConsumer(tarball, root);
    console.log("check:package: ok");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

main();
