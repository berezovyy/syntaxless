/** Resumable, no-database coordinator for the complete all-unique corpus. */
import { spawnSync } from "node:child_process";
import {
  closeSync,
  copyFileSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import { SHARD_FORMAT } from "../../featurize";
import { featurizeKindsStreaming } from "../../featurize-kinds";
import { classifyPopulation } from "../../mix/quotas";
import { CanonicalGzipWriter } from "../jsonl-io";
import { type CorpusRecord, SPLITS } from "../merge";
import {
  COMPLETE_SOURCE_VERSION,
  completeSourceCandidates,
  type CompleteSourceName,
  type CompleteSourceOptions,
} from "./complete-sources";
import { CONSTRUCTIONS_PATH } from "./constructions";
import { CorpusUnionPlan, StreamingCorpusUnion } from "./global-union";
import { manualSourceManifestSchema } from "./manual-source";
import { loadProtectedText } from "./protected-text";
import { diskPreflight, hashArtifact } from "./publication";
import { sourceExpectations } from "./source-expectations";
import { corpusCodeHashes, fileHash } from "./source-hashes";

const training = resolve(fileURLToPath(new URL("../../../", import.meta.url)));
const repo = resolve(training, "../../..");
const INITIAL_REQUIRED_BYTES = 24 * 1024 ** 3;

export const PRODUCTION_CONFIG_PATH = new URL(
  "../../../data/production/production-complete-v1.json",
  import.meta.url,
);

export const DEFAULT_PRODUCTION_OUTPUT = join(training, "data/synth/production-complete-v1");

export const PRODUCTION_PHASES = [
  "plan",
  "emit",
  "expectations",
  "shards",
  "verify",
  "publish",
] as const;

export type ProductionPhase = (typeof PRODUCTION_PHASES)[number];

const configSchema = z
  .object({
    version: z.literal(1),
    corpusVersion: z.literal("production-complete-v1"),
    policy: z.literal("all-eligible-runtime-normalized-texts"),
    manual: z
      .object({
        records: z.number().int().positive(),
        seed: z.number().int(),
        kernelsPerKind: z.number().int().positive(),
      })
      .strict(),
    union: z.object({ partitions: z.number().int().min(1).max(256) }).strict(),
    inputs: z
      .object({
        manualCorpus: z.string().min(1),
        manualManifest: z.string().min(1),
        publicCorpus: z.string().min(1),
        publicManifest: z.string().min(1),
        publicManifestSha256: z.string().regex(/^[a-f0-9]{64}$/),
        verifiedLlm: z.string().min(1),
        acceptedRealDirectory: z.string().min(1),
      })
      .strict(),
  })
  .strict();

type Config = z.infer<typeof configSchema>;
type Artifact = { bytes: number; sha256: string };
type Expectations = Awaited<ReturnType<typeof sourceExpectations>>;
type Population = "R" | "H" | "E";

interface BuildState {
  version: 1;
  corpusVersion: string;
  output: string;
  config: Config;
  configSha256: string;
  inputs: Record<string, string>;
  implementation: Record<string, string>;
  toolchain: Record<string, string | undefined>;
  plan?: {
    summary: {
      candidates: number;
      uniqueTexts: number;
      duplicates: number;
      partitions: number;
      ingestSha256: string;
      selectedReferencesSha256: string;
    };
    candidatesBySource: Record<CompleteSourceName, number>;
  };
  emit?: {
    corpus: Artifact;
    emittedBySource: Record<CompleteSourceName, number>;
    splits: Record<(typeof SPLITS)[number], number>;
    populations: Record<Population, number>;
    authorship: Record<string, number>;
  };
  expectations?: { artifact: Artifact; expectedShardBytes: number };
  shards?: { artifacts: Record<string, Artifact> };
  verify?: { artifact: Artifact };
}

const SOURCE_NAMES: readonly CompleteSourceName[] = [
  "manual",
  "authored",
  "systematic",
  "public",
  "verified-llm",
  "accepted-human",
];

function emptySourceCounts(): Record<CompleteSourceName, number> {
  return Object.fromEntries(SOURCE_NAMES.map((source) => [source, 0])) as Record<
    CompleteSourceName,
    number
  >;
}

function resolveTrainingPath(path: string): string {
  const resolved = resolve(training, path);

  if (resolved !== training && !resolved.startsWith(`${training}/`)) {
    throw new Error(`Production input is outside the training package: ${path}`);
  }

  return resolved;
}

function configFilePath(path: string | URL): string {
  return typeof path === "string" ? resolve(path) : fileURLToPath(path);
}

function pathsFor(outputValue: string) {
  const output = resolve(outputValue);
  const parent = dirname(output);
  const name = basename(output);

  return {
    output,
    parent,
    work: join(parent, `.${name}.work`),
    state: join(parent, `.${name}.state.json`),
    lock: join(parent, `.${name}.build.lock`),
  };
}

function writeJsonExclusive(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
}

function writeState(path: string, state: BuildState): void {
  const temporary = `${path}.tmp-${process.pid}`;

  writeFileSync(temporary, `${JSON.stringify(state, null, 2)}\n`, { flag: "wx" });
  const fd = openSync(temporary, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }

  renameSync(temporary, path);
}

function readState(path: string): BuildState {
  return JSON.parse(readFileSync(path, "utf8")) as BuildState;
}

async function artifact(path: string): Promise<Artifact> {
  return { bytes: statSync(path).size, sha256: await hashArtifact(path) };
}

class CorpusGzipWriter {
  readonly #writer: CanonicalGzipWriter;
  #pending = "";
  #closed = false;

  constructor(path: string) {
    this.#writer = new CanonicalGzipWriter(path);
  }

  add(record: CorpusRecord): Promise<void> | undefined {
    this.#pending += `${JSON.stringify(record)}\n`;
    if (this.#pending.length >= 1024 * 1024) return this.flush();
  }

  private async flush(): Promise<void> {
    if (!this.#pending) return;
    await this.#writer.append(this.#pending);
    this.#pending = "";
  }

  async close(): Promise<void> {
    if (this.#closed) throw new Error("Corpus JSONL writer is already closed");
    await this.flush();
    await this.#writer.close();
    this.#closed = true;
  }

  async abort(): Promise<void> {
    if (this.#closed) return;
    await this.#writer.abort();
    this.#closed = true;
    this.#pending = "";
  }
}

function hashDirectoryFiles(directory: string): Record<string, string> {
  const result: Record<string, string> = {};

  if (!existsSync(directory)) return result;

  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name, "en"),
    )) {
      const path = join(current, entry.name);

      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) result[relative(repo, path)] = fileHash(path);
    }
  };

  walk(directory);

  return result;
}

function sourceOptions(
  config: Config,
  protectedText: ReturnType<typeof loadProtectedText>,
  progress: (phase: string, count?: number) => void,
): CompleteSourceOptions {
  return {
    manualRecords: config.manual.records,
    manualSeed: config.manual.seed,
    manualKernelsPerKind: config.manual.kernelsPerKind,
    manualCorpusPath: resolveTrainingPath(config.inputs.manualCorpus),
    manualManifestPath: resolveTrainingPath(config.inputs.manualManifest),
    publicCorpusPath: resolveTrainingPath(config.inputs.publicCorpus),
    verifiedLlmPath: resolveTrainingPath(config.inputs.verifiedLlm),
    acceptedRealDirectory: resolveTrainingPath(config.inputs.acceptedRealDirectory),
    protectedText,
    progress: (source, count) => progress(`source/${source}`, count),
  };
}

async function inputHashes(
  configPath: string | URL,
  config: Config,
  protectedText: ReturnType<typeof loadProtectedText>,
): Promise<Record<string, string>> {
  const manualCorpus = resolveTrainingPath(config.inputs.manualCorpus);
  const manualManifest = resolveTrainingPath(config.inputs.manualManifest);
  const manualManifestSha256 = fileHash(manualManifest);
  const manual = manualSourceManifestSchema.parse(JSON.parse(readFileSync(manualManifest, "utf8")));

  if (
    manual.records !== config.manual.records ||
    manual.seed !== config.manual.seed ||
    manual.kernelsPerKind !== config.manual.kernelsPerKind
  ) {
    throw new Error("Manual-source manifest does not match production config");
  }

  const actualManual = await artifact(manualCorpus);

  if (
    actualManual.bytes !== manual.compressed.bytes ||
    actualManual.sha256 !== manual.compressed.sha256
  ) {
    throw new Error("Compressed manual source differs from its manifest");
  }

  const publicCorpus = resolveTrainingPath(config.inputs.publicCorpus);
  const publicManifest = resolveTrainingPath(config.inputs.publicManifest);

  if (fileHash(publicManifest) !== config.inputs.publicManifestSha256) {
    throw new Error("Pinned public-source manifest checksum mismatch");
  }

  const publicData = JSON.parse(readFileSync(publicManifest, "utf8")) as {
    artifacts?: Record<string, Artifact>;
  };

  const expectedPublic = publicData.artifacts?.["corpus.jsonl"];

  if (!expectedPublic) throw new Error("Public-source manifest lacks corpus.jsonl");
  const actualPublic = await artifact(publicCorpus);

  if (
    actualPublic.bytes !== expectedPublic.bytes ||
    actualPublic.sha256 !== expectedPublic.sha256
  ) {
    throw new Error("Public corpus differs from its pinned manifest");
  }

  const verifiedLlm = resolveTrainingPath(config.inputs.verifiedLlm);
  const constructions = fileURLToPath(CONSTRUCTIONS_PATH);

  const hashes: Record<string, string> = {
    [relative(repo, configFilePath(configPath))]: fileHash(configPath),
    [relative(repo, manualManifest)]: manualManifestSha256,
    [relative(repo, manualCorpus)]: actualManual.sha256,
    [relative(repo, publicManifest)]: config.inputs.publicManifestSha256,
    [relative(repo, publicCorpus)]: actualPublic.sha256,
    [relative(repo, verifiedLlm)]: fileHash(verifiedLlm),
    [relative(repo, constructions)]: fileHash(constructions),
    ...hashDirectoryFiles(resolveTrainingPath(config.inputs.acceptedRealDirectory)),
  };

  for (const source of protectedText.sources) hashes[relative(repo, source.file)] = source.sha256;

  return Object.fromEntries(Object.entries(hashes).sort(([a], [b]) => a.localeCompare(b)));
}

function verifyPythonAvailable(python: string): Record<string, string> {
  const result = spawnSync(
    python,
    [
      "-c",
      "import json,sys,numpy,torch; print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'torch':torch.__version__}))",
    ],
    { encoding: "utf8" },
  );

  if (result.error || result.status !== 0) {
    throw new Error(`Python training environment unavailable: ${result.error ?? result.stderr}`);
  }

  return JSON.parse(result.stdout) as Record<string, string>;
}

function assertState(
  state: BuildState,
  output: string,
  config: Config,
  configPath: string | URL,
): void {
  if (
    state.version !== 1 ||
    state.output !== output ||
    state.corpusVersion !== config.corpusVersion ||
    state.configSha256 !== fileHash(configPath)
  ) {
    throw new Error("Production work state belongs to different inputs or output");
  }

  if (JSON.stringify(corpusCodeHashes()) !== JSON.stringify(state.implementation)) {
    throw new Error("Corpus implementation changed since the plan phase");
  }
}

function nextPhase(state: BuildState | undefined): ProductionPhase {
  if (!state?.plan) return "plan";
  if (!state.emit) return "emit";
  if (!state.expectations) return "expectations";
  if (!state.shards) return "shards";
  if (!state.verify) return "verify";

  return "publish";
}

async function planPhase(
  statePath: string,
  work: string,
  output: string,
  config: Config,
  configPath: string | URL,
  progress: (phase: string, count?: number) => void,
): Promise<BuildState> {
  if (existsSync(statePath)) throw new Error("Plan phase is already complete");
  // Without a committed state file, everything at this owned hidden path is
  // an interrupted plan attempt and is safe to rebuild.
  rmSync(work, { recursive: true, force: true });
  mkdirSync(work, { recursive: true });
  diskPreflight(work, INITIAL_REQUIRED_BYTES);
  const protectedText = loadProtectedText();
  const inputs = await inputHashes(configPath, config, protectedText);
  const implementation = corpusCodeHashes();
  const python = join(repo, "training/cron/.venv/bin/python");

  const toolchain = {
    bun: process.versions.bun,
    platform: process.platform,
    architecture: process.arch,
    ...verifyPythonAvailable(python),
  };

  const scratch = join(work, ".union-scratch");

  rmSync(scratch, { recursive: true, force: true });

  const union = new StreamingCorpusUnion({
    directory: scratch,
    partitions: config.union.partitions,
  });

  const candidatesBySource = emptySourceCounts();

  try {
    for await (const candidate of completeSourceCandidates(
      sourceOptions(config, protectedText, progress),
    )) {
      candidatesBySource[candidate.source]++;
      union.add(candidate.record, candidate.union);
    }

    const plan = union.finalize();
    const target = join(work, "union-plan");

    rmSync(target, { recursive: true, force: true });
    mkdirSync(target);
    copyFileSync(join(scratch, "manifest.json"), join(target, "manifest.json"));

    copyFileSync(
      join(scratch, "selected-references.bitmap"),
      join(target, "selected-references.bitmap"),
    );

    const state: BuildState = {
      version: 1,
      corpusVersion: config.corpusVersion,
      output,
      config,
      configSha256: fileHash(configPath),
      inputs,
      implementation,
      toolchain,
      plan: { summary: plan.summary, candidatesBySource },
    };

    writeState(statePath, state);

    return state;
  } finally {
    union.abort();
  }
}

async function emitPhase(
  state: BuildState,
  statePath: string,
  work: string,
  config: Config,
  progress: (phase: string, count?: number) => void,
): Promise<void> {
  if (!state.plan || state.emit) throw new Error("Emit phase is not pending");
  const protectedText = loadProtectedText();
  const plan = new CorpusUnionPlan(join(work, "union-plan"));
  const replay = plan.replay();
  const partial = join(work, "corpus.jsonl.partial.gz");
  const corpusPath = join(work, "corpus.jsonl.gz");

  rmSync(partial, { force: true });
  rmSync(corpusPath, { force: true });
  const writer = new CorpusGzipWriter(partial);
  const emittedBySource = emptySourceCounts();
  const splits = { train: 0, validation: 0, heldout: 0 };
  const populations: Record<Population, number> = { R: 0, H: 0, E: 0 };
  const authorship: Record<string, number> = {};

  try {
    for await (const candidate of completeSourceCandidates(
      sourceOptions(config, protectedText, progress),
    )) {
      const decision = replay.add(candidate.record, candidate.union);

      if (!decision.selected) continue;

      const record: CorpusRecord = {
        ...candidate.record,
        id: `${config.corpusVersion}/${decision.textHash}`,
      };

      const flush = writer.add(record);

      if (flush) await flush;
      emittedBySource[candidate.source]++;
      splits[record.split]++;

      const population = classifyPopulation({
        intent: record.intent,
        speechAct: record.speechAct ?? "none",
        spans: record.spans,
      });

      populations[population]++;
      const author = record.provenance?.authorship ?? "legacy-unspecified";

      authorship[author] = (authorship[author] ?? 0) + 1;
    }

    replay.finish();
    await writer.close();
    renameSync(partial, corpusPath);
    const corpus = await artifact(corpusPath);

    if (Object.values(splits).some((count) => count === 0)) {
      throw new Error(`Complete corpus has an empty split: ${JSON.stringify(splits)}`);
    }

    state.emit = { corpus, emittedBySource, splits, populations, authorship };
    writeState(statePath, state);
  } finally {
    await writer.abort();
  }
}

async function expectationsPhase(
  state: BuildState,
  statePath: string,
  work: string,
): Promise<void> {
  if (!state.plan || !state.emit || state.expectations) {
    throw new Error("Expectations phase is not pending");
  }

  const expectations = await sourceExpectations(join(work, "corpus.jsonl.gz"));

  if (
    expectations.corpus.records !== state.plan.summary.uniqueTexts ||
    expectations.corpus.sha256 !== state.emit.corpus.sha256
  ) {
    throw new Error("JSONL expectations differ from the union output");
  }

  const path = join(work, "source-expectations.json");

  rmSync(path, { force: true });
  writeJsonExclusive(path, expectations);

  const expectedShardBytes = Object.values(expectations.splits).reduce(
    (total, split) =>
      total + Object.values(split.fields).reduce((sum, field) => sum + field.bytes, 0),
    0,
  );

  state.expectations = { artifact: await artifact(path), expectedShardBytes };
  writeState(statePath, state);
}

function shardArtifacts(
  work: string,
  expectations: Expectations,
): Promise<Record<string, Artifact>> {
  return (async () => {
    const artifacts: Record<string, Artifact> = {};

    for (const split of SPLITS) {
      const prefix = `corpus-${split}`;

      artifacts[`${prefix}.json`] = await artifact(join(work, `${prefix}.json`));
      const expectedSplit = expectations.splits[split]!;

      for (const [field, format] of Object.entries(SHARD_FORMAT)) {
        const name = `${prefix}${format.suffix}`;
        const expected = expectedSplit.fields[field as keyof typeof expectedSplit.fields];

        if (!expected || statSync(join(work, name)).size !== expected.bytes) {
          throw new Error(`Shard artifact differs from expectations: ${name}`);
        }

        artifacts[name] = expected;
      }
    }

    return artifacts;
  })();
}

async function shardsPhase(state: BuildState, statePath: string, work: string): Promise<void> {
  if (!state.plan || !state.emit || !state.expectations || state.shards) {
    throw new Error("Shards phase is not pending");
  }

  const expectations = JSON.parse(
    readFileSync(join(work, "source-expectations.json"), "utf8"),
  ) as Expectations;

  diskPreflight(work, state.expectations.expectedShardBytes + 64 * 1024 ** 2, 512 * 1024 ** 2);

  const summary = await featurizeKindsStreaming(
    join(work, "corpus.jsonl.gz"),
    join(work, "corpus"),
  );

  if (
    summary.records !== state.plan.summary.uniqueTexts ||
    Object.values(summary.splits).some(
      (split) => split.skippedEmpty !== 0 || split.skippedLong !== 0,
    )
  ) {
    throw new Error("Streaming featurization skipped complete-corpus records");
  }

  state.shards = { artifacts: await shardArtifacts(work, expectations) };
  writeState(statePath, state);
}

async function verifyPhase(state: BuildState, statePath: string, work: string): Promise<void> {
  if (!state.plan || !state.shards || state.verify) throw new Error("Verify phase is not pending");
  const report = join(work, "python-verification.json");

  rmSync(report, { force: true });
  const python = join(repo, "training/cron/.venv/bin/python");

  const result = spawnSync(
    python,
    [
      join(repo, "training/cron/verify_production_shards.py"),
      join(work, "corpus.jsonl.gz"),
      join(work, "corpus"),
      join(work, "source-expectations.json"),
      report,
    ],
    { stdio: "inherit" },
  );

  if (result.error || result.status !== 0) {
    throw new Error(`Python shard verification failed: ${result.error ?? result.status}`);
  }

  const parsed = JSON.parse(readFileSync(report, "utf8")) as {
    passed?: boolean;
    records?: number;
  };

  if (!parsed.passed || parsed.records !== state.plan.summary.uniqueTexts) {
    throw new Error("Python did not consume the complete corpus");
  }

  state.verify = { artifact: await artifact(report) };
  writeState(statePath, state);
}

function syncDirectory(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

async function publishPhase(
  state: BuildState,
  statePath: string,
  work: string,
  output: string,
  configPath: string | URL,
): Promise<void> {
  if (!state.plan || !state.emit || !state.expectations || !state.shards || !state.verify) {
    throw new Error("Publish phase requires every prior phase");
  }

  const protectedText = loadProtectedText();
  const currentInputs = await inputHashes(configPath, state.config, protectedText);

  if (JSON.stringify(currentInputs) !== JSON.stringify(state.inputs)) {
    throw new Error("Production inputs changed since the plan phase");
  }

  for (const [name, hash] of Object.entries(
    hashDirectoryFiles(resolveTrainingPath(state.config.inputs.acceptedRealDirectory)),
  )) {
    if (state.inputs[name] !== hash) {
      throw new Error(`Production input appeared during generation: ${name}`);
    }
  }

  const artifacts: Record<string, Artifact> = {
    "corpus.jsonl.gz": state.emit.corpus,
    "source-expectations.json": state.expectations.artifact,
    "python-verification.json": state.verify.artifact,
    ...state.shards.artifacts,
  };

  const manifest = {
    version: 1,
    status: "complete-all-eligible-unique-corpus-and-verified-shards",
    corpusVersion: state.config.corpusVersion,
    policy: state.config.policy,
    config: state.config,
    configSha256: state.configSha256,
    sourceTraversalVersion: COMPLETE_SOURCE_VERSION,
    inputs: state.inputs,
    implementation: state.implementation,
    toolchain: state.toolchain,
    counts: {
      candidates: state.plan.summary.candidates,
      uniqueRecords: state.plan.summary.uniqueTexts,
      duplicatesRemoved: state.plan.summary.duplicates,
      candidatesBySource: state.plan.candidatesBySource,
      emittedBySource: state.emit.emittedBySource,
      splits: state.emit.splits,
      populations: state.emit.populations,
      authorship: state.emit.authorship,
    },
    union: {
      partitions: state.plan.summary.partitions,
      ingestSha256: state.plan.summary.ingestSha256,
      selectedReferencesSha256: state.plan.summary.selectedReferencesSha256,
      identity: "SHA-256 of runtime-normalized text",
      conflictPolicy: "same wording with different model supervision fails",
      winnerPolicy: "source priority, then stable provenance reference",
    },
    protectedSources: protectedText.sources.map((source) => ({
      ...source,
      file: relative(repo, source.file),
    })),
    trainingInputPrefix: "corpus",
    splitPrefixes: Object.fromEntries(SPLITS.map((split) => [split, `corpus-${split}`])),
    artifacts,
    limitations: [
      ...(state.emit.emittedBySource["accepted-human"] === 0
        ? ["No accepted real-user rows currently carry explicit consented human provenance"]
        : []),
      "AI-authored constructions are labeled as AI-authored",
      "Corpus verification proves data integrity and loader consumption, not model quality",
    ],
  };

  rmSync(join(work, "union-plan"), { recursive: true, force: true });
  rmSync(join(work, "manifest.json"), { force: true });
  writeJsonExclusive(join(work, "manifest.json"), manifest);
  const manifestFd = openSync(join(work, "manifest.json"), "r");

  try {
    fsyncSync(manifestFd);
  } finally {
    closeSync(manifestFd);
  }

  syncDirectory(work);
  if (existsSync(output)) throw new Error(`Refusing to overwrite production version: ${output}`);
  renameSync(work, output);
  syncDirectory(dirname(output));
  rmSync(statePath, { force: true });
}

export interface ProductionPhaseResult {
  completed: ProductionPhase | "already-published";
  next?: ProductionPhase;
  output: string;
  work: string;
}

function clearStaleLock(path: string): void {
  if (!existsSync(path)) return;
  let live = false;

  try {
    const value = JSON.parse(readFileSync(path, "utf8")) as { pid?: unknown };

    if (typeof value.pid === "number" && Number.isInteger(value.pid)) {
      try {
        process.kill(value.pid, 0);
        live = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EPERM") live = true;
      }
    }
  } catch {
    // A malformed lock cannot identify a live owner and is treated as stale.
  }

  if (live) throw new Error(`Production build is already running: ${path}`);
  rmSync(path, { force: true });
}

/** Run exactly one resumable phase. Reinvoke with `next` until published. */
export async function runProductionPhase(
  requested: ProductionPhase | "next" = "next",
  outputValue = DEFAULT_PRODUCTION_OUTPUT,
  configPath: string | URL = PRODUCTION_CONFIG_PATH,
  progress: (phase: string, count?: number) => void = () => {},
): Promise<ProductionPhaseResult> {
  const paths = pathsFor(outputValue);

  mkdirSync(paths.parent, { recursive: true });

  if (existsSync(paths.output)) {
    if (!existsSync(join(paths.output, "manifest.json"))) {
      throw new Error(`Output exists without a production manifest: ${paths.output}`);
    }

    rmSync(paths.state, { force: true });

    return { completed: "already-published", output: paths.output, work: paths.work };
  }

  const config = configSchema.parse(JSON.parse(readFileSync(configPath, "utf8")));

  clearStaleLock(paths.lock);
  const lockFd = openSync(paths.lock, "wx", 0o600);

  try {
    writeFileSync(lockFd, JSON.stringify({ pid: process.pid, output: paths.output }));
    fsyncSync(lockFd);
    let state = existsSync(paths.state) ? readState(paths.state) : undefined;

    if (state) assertState(state, paths.output, config, configPath);
    const phase = requested === "next" ? nextPhase(state) : requested;

    progress(`phase/${phase}`);

    if (phase === "plan") {
      state = await planPhase(paths.state, paths.work, paths.output, config, configPath, progress);
    } else {
      if (!state) throw new Error(`${phase} phase requires a completed plan phase`);
      if (phase === "emit") await emitPhase(state, paths.state, paths.work, config, progress);
      else if (phase === "expectations") await expectationsPhase(state, paths.state, paths.work);
      else if (phase === "shards") await shardsPhase(state, paths.state, paths.work);
      else if (phase === "verify") await verifyPhase(state, paths.state, paths.work);
      else await publishPhase(state, paths.state, paths.work, paths.output, configPath);
    }

    const updated = existsSync(paths.state) ? readState(paths.state) : undefined;

    return {
      completed: phase,
      ...(phase === "publish" ? {} : { next: nextPhase(updated) }),
      output: paths.output,
      work: paths.work,
    };
  } finally {
    closeSync(lockFd);
    rmSync(paths.lock, { force: true });
  }
}

if (import.meta.main) {
  const [first, second, ...extra] = process.argv.slice(2);

  if (extra.length) throw new Error("Usage: build.ts [PHASE] [NEW_OUTPUT_DIRECTORY]");
  const isPhase = first === "next" || PRODUCTION_PHASES.includes(first as ProductionPhase);
  const phase = isPhase ? (first as ProductionPhase | "next") : "next";

  const output = resolve(
    isPhase ? (second ?? DEFAULT_PRODUCTION_OUTPUT) : (first ?? DEFAULT_PRODUCTION_OUTPUT),
  );

  const result = await runProductionPhase(phase, output, PRODUCTION_CONFIG_PATH, (name, n) =>
    console.error(`${name}${n === undefined ? "" : `: ${n.toLocaleString()}`}`),
  );

  console.log(JSON.stringify(result, null, 2));
}
