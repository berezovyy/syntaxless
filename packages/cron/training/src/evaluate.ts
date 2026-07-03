/**
 * Candidate/baseline evaluation over a gold corpus.
 *
 * Every number this module reports comes from the *built* `packages/cron/dist/`
 * runtime, never from `src/`. The bundled `dist` is scored directly. When a
 * weight module path is supplied (a candidate artifact), the built tree is
 * copied to a scratch directory and only its `model/weights.gen.js` is
 * replaced, so the exact shipped engine runs on the candidate parameters in
 * this same process. The crate is what will ship, not a re-implementation.
 *
 * The corpus is gold data with rendered spans, an expected AST, and an expected
 * compiled expression. `tokenRoles` scores the built internal modules
 * (tokenizer + inference); `exactAst` and `compiledExpression` score the built
 * public entry (`parse` + `compile`). The three contracts are reported side by
 * side and never summed.
 *
 * Reports are write-only evidence: `results/<set>.json` is never read as an
 * input to another run. The calibration and held-out frame splits are
 * development data and may not gate promotion; every report says so in `scope`.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

import type { CompileResult } from "../../src/compiler/cron";
import type { RawToken } from "../../src/model/tokenizer";
import type { ParseResult } from "../../src/parser/parse";
import type { ScheduleIntent } from "../../src/types/intent";

/** Default corpus/report locations, relative to the repository root. */
export const DEFAULTS = {
  dist: "packages/cron/dist",
  corpus: "packages/cron/training/data/gold/reserved-carriers.jsonl",
  out: "packages/cron/training/results/reserved-carriers.json",
} as const;

/** Cap the failures carried into a report so one bad artifact cannot bloat it. */
export const MAX_FAILURES = 50;

/** Repository root: this file lives at `packages/cron/training/src/`. */
export function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

/** Resolve a CLI path against the repository root unless already absolute. */
export function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

/** SHA-256 as lowercase hex. */
export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** One gold record. `spans` are renderer-supplied labels, never parser output. */
export interface GoldRecord {
  id: string;
  family: string;
  text: string;
  spans: readonly {
    start: number;
    end: number;
    label: string;
    clauseStart: boolean;
  }[];
  intent: ScheduleIntent | null;
  expected: { supported: boolean; expression: string | null };
}

/** Running correct/total counts; the guard's `Counts` shape. */
export interface Counts {
  total: number;
  correct: number;
}

export type Metric = "tokenRoles" | "exactAst" | "compiledExpression";

export interface ContractFailure {
  id: string;
  family: string;
  detail: string;
  expected: unknown;
  actual: unknown;
}

/** One contract's report. Every report carries its own provenance fields. */
export interface ContractReport {
  metric: Metric;
  artifactSha256: string;
  corpus: string;
  corpusSha256: string;
  familyDistribution: Record<string, number>;
  scope: string;
  total: number;
  correct: number;
  accuracy: number;
  byFamily: Record<string, Counts>;
  failures: ContractFailure[];
}

export interface EvaluationResult {
  artifactSha256: string;
  corpus: string;
  corpusSha256: string;
  familyDistribution: Record<string, number>;
  scope: string;
  contracts: ContractReport[];
}

/**
 * Throw when reports disagree about the artifact that produced them. A report
 * set is only comparable when every member came from one decoded artifact.
 */
export function assertSameArtifact(reports: readonly { artifactSha256: string }[]): string {
  if (reports.length === 0) {
    throw new Error("assertSameArtifact: at least one report is required");
  }

  const expected = reports[0]!.artifactSha256;

  for (const report of reports) {
    if (report.artifactSha256 !== expected) {
      throw new Error(`artifact mismatch: ${report.artifactSha256} !== ${expected}`);
    }
  }

  return expected;
}

interface WeightsArtifact {
  q: string;
}

interface DistIndex {
  parse: (text: string) => Promise<ParseResult>;
  compile: (intent: ScheduleIntent) => CompileResult;
}

interface DistTokenizer {
  tokenize: (text: string) => RawToken[];
}

interface DistInference {
  inferTokens: (tokens: readonly RawToken[]) => { labels: Uint8Array };
}

interface DistLabels {
  LABELS: readonly string[];
}

/** A decoded runtime plus the provenance of the artifact it runs. */
export interface Engine {
  artifactSha256: string;
  parse: (text: string) => Promise<ParseResult>;
  compile: (intent: ScheduleIntent) => CompileResult;
  tokenize: (text: string) => RawToken[];
  inferTokens: (tokens: readonly RawToken[]) => { labels: Uint8Array };
  labelName: (id: number) => string;
  dispose: () => Promise<void>;
}

async function importModule<T>(path: string): Promise<T> {
  return (await import(pathToFileURL(path).href)) as T;
}

/**
 * Load the built runtime from `distDir`. With no `weightsModulePath` the built
 * module runs as shipped. With one, a scratch copy of `dist/` is written with
 * the supplied artifact so the built engine can be scored on it.
 */
export async function loadEngine(distDir: string, weightsModulePath?: string): Promise<Engine> {
  const built = resolveRepo(distDir);

  if (!existsSync(join(built, "index.js"))) {
    throw new Error(`built runtime missing: ${built}/index.js (run the build)`);
  }

  let root = built;
  let artifact: WeightsArtifact;
  let temp: string | null = null;

  if (weightsModulePath === undefined) {
    const bundled = await importModule<{ weights: WeightsArtifact }>(
      join(built, "model", "weights.gen.js"),
    );

    artifact = bundled.weights;
  } else {
    const modulePath = resolveRepo(weightsModulePath);

    if (!existsSync(modulePath)) {
      throw new Error(`weights module not found: ${modulePath}`);
    }

    const supplied = await importModule<{ weights: WeightsArtifact }>(modulePath);

    artifact = supplied.weights;
    temp = await mkdtemp(join(tmpdir(), "syntaxless-evaluate-"));
    root = join(temp, "dist");
    await cp(built, root, { recursive: true });

    await writeFile(
      join(root, "model", "weights.gen.js"),
      `export const weights = ${JSON.stringify(artifact)};\n`,
    );
  }

  const index = await importModule<DistIndex>(join(root, "index.js"));
  const tokenizer = await importModule<DistTokenizer>(join(root, "model", "tokenizer.js"));
  const inference = await importModule<DistInference>(join(root, "model", "inference.js"));
  const labels = await importModule<DistLabels>(join(root, "model", "labels.js"));

  const artifactSha256 = sha256Hex(artifact.q);
  const scratch = temp;

  return {
    artifactSha256,
    parse: index.parse.bind(index),
    compile: index.compile.bind(index),
    tokenize: tokenizer.tokenize.bind(tokenizer),
    inferTokens: inference.inferTokens.bind(inference),
    labelName: (id: number) => {
      const label = labels.LABELS[id];

      if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

      return label;
    },
    dispose: async () => {
      if (scratch !== null) await rm(scratch, { recursive: true, force: true });
    },
  };
}

interface Accumulator {
  total: number;
  correct: number;
  byFamily: Record<string, Counts>;
  failures: ContractFailure[];
}

function emptyAccumulator(): Accumulator {
  return { total: 0, correct: 0, byFamily: {}, failures: [] };
}

function record(
  accumulator: Accumulator,
  family: string,
  correct: boolean,
  failure?: ContractFailure,
): void {
  const counts = (accumulator.byFamily[family] ??= { total: 0, correct: 0 });

  counts.total += 1;
  accumulator.total += 1;

  if (correct) {
    counts.correct += 1;
    accumulator.correct += 1;
  } else if (failure !== undefined && accumulator.failures.length < MAX_FAILURES) {
    accumulator.failures.push(failure);
  }
}

function report(
  metric: Metric,
  accumulator: Accumulator,
  provenance: {
    artifactSha256: string;
    corpus: string;
    corpusSha256: string;
    familyDistribution: Record<string, number>;
    scope: string;
  },
): ContractReport {
  return {
    metric,
    ...provenance,
    total: accumulator.total,
    correct: accumulator.correct,
    accuracy: accumulator.total === 0 ? 0 : accumulator.correct / accumulator.total,
    byFamily: accumulator.byFamily,
    failures: accumulator.failures,
  };
}

/** Map each non-whitespace token onto the gold span that covers it. */
function goldLabelFor(
  token: RawToken,
  spans: GoldRecord["spans"],
  cursor: { index: number },
): string | null {
  while (cursor.index < spans.length && spans[cursor.index]!.end <= token.start) {
    cursor.index += 1;
  }

  const span = spans[cursor.index];

  if (span === undefined || token.start < span.start || token.end > span.end) {
    throw new Error(`unaligned token span`);
  }

  return span.label;
}

export function parseCorpus(contents: string): GoldRecord[] {
  return contents
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as GoldRecord);
}

/**
 * Score one decoded engine against a corpus. All three contracts share the
 * engine's `artifactSha256`; the caller may assert that with
 * {@link assertSameArtifact}.
 */
export async function evaluateCorpus(
  engine: Engine,
  corpusPath: string,
  scope: string,
): Promise<EvaluationResult> {
  const source = resolveRepo(corpusPath);
  const contents = await readFile(source, "utf8");
  const corpusSha256 = sha256Hex(contents);
  const records = parseCorpus(contents);

  const familyDistribution: Record<string, number> = {};

  for (const item of records) {
    familyDistribution[item.family] = (familyDistribution[item.family] ?? 0) + 1;
  }

  const tokenRoles = emptyAccumulator();
  const exactAst = emptyAccumulator();
  const compiledExpression = emptyAccumulator();

  for (const item of records) {
    const tokens = engine.tokenize(item.text);
    const predictions = engine.inferTokens(tokens);
    const cursor = { index: 0 };

    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index]!;

      if (token.kind === 3) continue;
      const gold = goldLabelFor(token, item.spans, cursor);
      const predicted = engine.labelName(predictions.labels[index]!);
      const correct = gold === predicted;

      record(
        tokenRoles,
        item.family,
        correct,
        correct
          ? undefined
          : {
              id: item.id,
              family: item.family,
              detail: `token ${JSON.stringify(token.text)}`,
              expected: gold,
              actual: predicted,
            },
      );
    }

    const parsed = await engine.parse(item.text);

    if (item.intent === null) {
      const rejected = !parsed.ok;

      record(exactAst, item.family, rejected, {
        id: item.id,
        family: item.family,
        detail: "expected no schedule",
        expected: null,
        actual: parsed.ok ? parsed.intent : { ok: false, code: parsed.code },
      });

      record(compiledExpression, item.family, rejected, {
        id: item.id,
        family: item.family,
        detail: "expected no expression",
        expected: item.expected,
        actual: parsed.ok ? "parsed" : { ok: false, code: parsed.code },
      });

      continue;
    }

    const astCorrect = parsed.ok && isDeepStrictEqual(parsed.intent, item.intent);

    record(exactAst, item.family, astCorrect, {
      id: item.id,
      family: item.family,
      detail: "intent mismatch",
      expected: item.intent,
      actual: parsed.ok ? parsed.intent : { ok: false, code: parsed.code },
    });

    const compiled = parsed.ok ? engine.compile(parsed.intent) : null;

    const expressionCorrect = item.expected.supported
      ? parsed.ok &&
        compiled !== null &&
        compiled.supported &&
        compiled.expression === item.expected.expression
      : parsed.ok && compiled !== null && !compiled.supported;

    record(compiledExpression, item.family, expressionCorrect, {
      id: item.id,
      family: item.family,
      detail: "compiled expression mismatch",
      expected: item.expected,
      actual: compiled,
    });
  }

  const provenance = {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
  };

  return {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
    contracts: [
      report("tokenRoles", tokenRoles, provenance),
      report("exactAst", exactAst, provenance),
      report("compiledExpression", compiledExpression, provenance),
    ],
  };
}

export interface EvaluateOptions {
  dist: string;
  corpus: string;
  out: string;
  weights?: string;
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

export function parseArgs(argv: readonly string[]): EvaluateOptions {
  let dist: string = DEFAULTS.dist;
  let corpus: string = DEFAULTS.corpus;
  let out: string = DEFAULTS.out;
  let weights: string | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;

    switch (arg) {
      case "--dist":
        dist = readOption(argv, index);
        index += 1;
        break;
      case "--corpus":
        corpus = readOption(argv, index);
        index += 1;
        break;
      case "--out":
        out = readOption(argv, index);
        index += 1;
        break;
      case "--weights":
        weights = readOption(argv, index);
        index += 1;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  return { dist, corpus, out, ...(weights === undefined ? {} : { weights }) };
}

export const DEFAULT_SCOPE =
  "Held-out reserved-carrier phrases. Calibration and held-out frame splits " +
  "are development data and may not gate promotion.";

if (import.meta.main) {
  const args = parseArgs(process.argv.slice(2));
  const engine = await loadEngine(args.dist, args.weights);

  try {
    const result = await evaluateCorpus(engine, args.corpus, DEFAULT_SCOPE);

    assertSameArtifact(result.contracts);
    const out = resolveRepo(args.out);

    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, `${JSON.stringify(result, null, 2)}\n`);

    for (const contract of result.contracts) {
      process.stdout.write(
        `${contract.metric}: ${contract.correct}/${contract.total} ` +
          `(${contract.accuracy.toFixed(3)})\n`,
      );
    }

    process.stdout.write(`artifact sha256: ${result.artifactSha256}\n`);
    process.stdout.write(`report -> ${out}\n`);
  } finally {
    await engine.dispose();
  }
}
