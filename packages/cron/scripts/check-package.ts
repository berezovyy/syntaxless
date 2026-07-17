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
import { pathToFileURL } from "node:url";

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

  const dedicated = join(distDir, "weights.gen.js");

  if (!readFileSync(dedicated, "utf8").includes(needle)) {
    throw new Error("check:package: dedicated weights module lacks the payload");
  }

  console.log("check:package: weights isolated in dist/weights.gen.js");
}

const CONSUMER = `import {
  clarify,
  resolveClarification,
  compile,
  cron,
  explain,
  validate,
  recurrences,
  IntentError,
  getEngine,
  parse,
} from "@syntaxless/cron";
import type { MinuteOfHour, ScheduleClarification, ScheduleIntent } from "@syntaxless/cron";
import { createTemporalCalendar } from "@syntaxless/cron/temporal";
import type { TemporalProvider } from "@syntaxless/cron/temporal";

globalThis.fetch = async () => {
  throw new Error("packed consumer must run offline");
};

const minute: MinuteOfHour = { minute: 15 };
const hourly: ScheduleIntent = { frequency: { unit: "hour", interval: 2 }, time: minute };

validate(hourly);

if (!explain(hourly).includes("15 */2 * * *")) {
  throw new Error("packed hourly explanation mismatch");
}

const bounded: ScheduleIntent = {
  frequency: { unit: "day", interval: 1 },
  bounds: { start: "2028-02-29" },
};

validate(bounded);

if (!explain(bounded).includes("inclusive local start date is 2028-02-29")) {
  throw new Error("packed bounds explanation mismatch");
}

const unsupported = compile(bounded);

if (unsupported.supported || unsupported.code !== "bounds") {
  throw new Error("packed compiler discarded bounds");
}

let invalidRejected = false;

try {
  validate({ ...bounded, bounds: { start: "2026-02-29" } });
} catch (error) {
  invalidRejected = error instanceof IntentError && error.code === "invalid-bounds";
}

if (!invalidRejected) {
  throw new Error("packed validator accepted an invalid date");
}

const incomplete: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  time: { hour: 9, minute: 25 },
};

const question: ScheduleClarification | null = clarify(incomplete);

if (question?.field !== "weekdays" || question.allowed.length !== 7) {
  throw new Error("packed weekday clarification mismatch");
}

const resolved = resolveClarification(incomplete, { weekdays: ["tuesday", "thursday"] });
const clarified = compile(resolved);

if (
  !clarified.supported ||
  clarified.expression !== "25 9 * * 2,4" ||
  incomplete.weekdays !== undefined
) {
  throw new Error("packed clarification discarded or mutated facts");
}

const civil = recurrences(
  { frequency: { unit: "month", interval: 1 }, dayOfMonth: 31 },
  {
    timezone: "UTC",
    reference: "2028-01-31T12:00:00Z",
    count: 2,
    policies: { monthEnd: "constrain" },
  },
);

const civilInstants = "2028-02-29T00:00:00.000Z,2028-03-31T00:00:00.000Z";

if (!civil.ok || civil.occurrences.map((item) => item.instant).join(",") !== civilInstants) {
  throw new Error("packed Gregorian recurrence mismatch");
}

const temporal = (globalThis as unknown as { Temporal: TemporalProvider }).Temporal;

const zoned = recurrences(
  { frequency: { unit: "day", interval: 1 }, time: { hour: 2, minute: 30 } },
  {
    timezone: "America/New_York",
    reference: "2026-03-07T12:00:00Z",
    count: 1,
    calendar: createTemporalCalendar(temporal),
    policies: { gap: "later" },
  },
);

if (!zoned.ok || zoned.occurrences[0]?.instant !== "2026-03-08T07:30:00.000Z") {
  throw new Error("packed zoned recurrence mismatch");
}

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

if (!parsed.ok) {
  throw new Error("packed parse failed: " + parsed.reason);
}

if (parsed.intent.frequency.unit !== "day" || parsed.intent.frequency.interval !== 1) {
  throw new Error("packed parse intent mismatch: " + JSON.stringify(parsed.intent));
}

const missingMonth = await cron("every month");

if (
  missingMonth.supported ||
  missingMonth.clarification?.field !== "dayOfMonth" ||
  missingMonth.intent?.frequency.unit !== "month"
) {
  throw new Error("packed convenience clarification mismatch");
}

let rejected = false;

try {
  await getEngine({ weights: new TextEncoder().encode(JSON.stringify({ version: 2 })) });
} catch (error) {
  rejected = (error as Error).name === "unsupported-weights-version";
}

if (!rejected) {
  throw new Error("packed runtime silently accepted an unsupported weights version");
}

console.log("check:package: consumer runtime ok");
`;

const CONSUMER_TSCONFIG = {
  compilerOptions: {
    target: "ES2023",
    module: "ESNext",
    moduleResolution: "Bundler",
    lib: ["ES2023", "DOM", "DOM.Iterable"],
    strict: true,
    outDir: "emitted",
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

  execFileSync("bun", ["add", tarball], {
    cwd: consumerDir,
    stdio: "inherit",
    env: {
      ...process.env,
      BUN_INSTALL_CACHE_DIR: join(root, "cache"),
      BUN_TMPDIR: root,
    },
  });

  // Check public declarations even if the current model later fails inference.
  execFileSync(process.execPath, [tscBinary, "-p", "tsconfig.json"], {
    cwd: consumerDir,
    stdio: "inherit",
  });

  execFileSync("bun", ["run", "consumer.ts"], { cwd: consumerDir, stdio: "inherit" });

  // The host supplies the optional polyfill; the tarball has no runtime dependency on it.
  const provider = join(
    packageRoot,
    "node_modules",
    "@js-temporal",
    "polyfill",
    "dist",
    "index.esm.js",
  );

  const providerSpecifier = JSON.stringify(pathToFileURL(provider).href);

  writeFileSync(
    join(consumerDir, "temporal-host.mjs"),
    `import { Temporal } from ${providerSpecifier}; globalThis.Temporal = Temporal;\n`,
  );

  execFileSync("node", ["--import", "./temporal-host.mjs", "emitted/consumer.js"], {
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
