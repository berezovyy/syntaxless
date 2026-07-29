/** A consumed acceptance set is evidence for one frozen candidate, never training. */
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { fromRepository, toRepository } from "../../../shared/training/paths.js";
import { runSizeGate } from "../../scripts/size-gate.js";
import { compile } from "../../src/compiler/index.js";
import type { JsonSchema } from "../../src/compiler/index.js";
import { parse } from "../../src/parser/parse.js";
import type { ParseResult } from "../../src/parser/parse.js";
import type { FieldType } from "../../src/types/intent.js";
import { evaluateRecords, readCorpus } from "./evaluate.js";
import { sha256 } from "./freeze.js";
import type { CorpusRecord } from "./generate.js";
import { verifyParity } from "./parity.js";

const root = resolve(import.meta.dir, "../../../..");

export const MAJOR_FAMILIES: readonly FieldType[] = [
  "string",
  "email",
  "url",
  "uuid",
  "number",
  "integer",
  "boolean",
];

export interface AcceptanceRow extends CorpusRecord {
  trainingEligible: false;
  critical: boolean;
  compileTarget?: JsonSchema;
}

function sourceIdentity(): string {
  const inventory: Record<string, string> = {};

  function visit(directory: string): void {
    for (const entry of readdirSync(resolve(root, directory), {
      withFileTypes: true,
    })) {
      const path = `${directory}/${entry.name}`;

      if (entry.isDirectory()) {
        visit(path);
      } else if (entry.name.endsWith(".ts") && entry.name !== "weights.gen.ts") {
        inventory[path] = sha256(readFileSync(resolve(root, path)));
      }
    }
  }

  visit("packages/validation/src");
  visit("packages/validation/scripts");
  visit("packages/validation/training/src");
  visit("packages/validation/training/tests");
  visit("packages/shared");

  inventory["scripts/declarations.ts"] = sha256(
    readFileSync(resolve(root, "scripts/declarations.ts")),
  );

  inventory["scripts/check-package.ts"] = sha256(
    readFileSync(resolve(root, "scripts/check-package.ts")),
  );

  for (const name of [
    "package.json",
    "tsconfig.json",
    "tsconfig.build.json",
    "training/package.json",
    "training/tsconfig.json",
  ]) {
    inventory[`packages/validation/${name}`] = sha256(
      readFileSync(resolve(root, "packages/validation", name)),
    );
  }

  for (const path of [
    "training/validation/export.py",
    "training/shared/train.py",
    "training/shared/artifact.py",
    "training/shared/calibrate.py",
    "training/shared/model.py",
    "training/shared/dataset.py",
  ]) {
    inventory[path] = sha256(readFileSync(resolve(root, path)));
  }

  for (const name of ["gate", "evaluate", "parity", "freeze"]) {
    inventory[`training/${name}`] = sha256(readFileSync(join(import.meta.dir, `${name}.ts`)));
  }

  // Scope is a release contract, not a threshold chosen after seeing acceptance.
  inventory["packages/validation/evaluation/README.md"] = sha256(
    readFileSync(resolve(root, "packages/validation/evaluation/README.md")),
  );

  return sha256(JSON.stringify(Object.entries(inventory).sort(([a], [b]) => a.localeCompare(b))));
}

function candidateIdentity(run: string) {
  const directory = join(run, "frozen");
  const manifest = JSON.parse(readFileSync(join(directory, "freeze.json"), "utf8"));

  if (
    sha256(readFileSync(join(directory, "weights.json"))) !== manifest.artifactFileSha256 ||
    sha256(readFileSync(join(directory, "weights.gen.ts"))) !== manifest.sourceFileSha256
  ) {
    throw new Error("frozen artifact or its generated module changed");
  }

  return {
    artifactFileSha256: manifest.artifactFileSha256 as string,
    calibrationCorpusSha256: manifest.corpusSha256 as string,
    exportReportSha256: sha256(readFileSync(join(run, "candidate/export-report.json"))),
    sourceIdentity: sourceIdentity(),
    families: MAJOR_FAMILIES,
  };
}

export function prepareGate(run: string): void {
  writeFileSync(
    join(run, "scope-freeze.json"),
    JSON.stringify(
      {
        ...candidateIdentity(run),
        createdAt: new Date().toISOString(),
        acceptanceUsed: false,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );
}

export function acceptanceCoverage(rows: readonly AcceptanceRow[]) {
  const texts = new Set<string>();
  const positiveFamilies: Record<string, number> = {};

  let positive = 0;
  let negative = 0;
  let critical = 0;

  for (const row of rows) {
    if (
      row.trainingEligible !== false ||
      row.split !== "heldout" ||
      !row.text.trim() ||
      typeof row.critical !== "boolean" ||
      texts.has(row.text.trim().toLowerCase())
    ) {
      throw new Error(
        "acceptance must contain unique, evaluation-only rows with explicit critical flags",
      );
    }

    texts.add(row.text.trim().toLowerCase());
    critical += Number(row.critical);

    if (row.intent) {
      if (
        !MAJOR_FAMILIES.includes(row.intent.type) ||
        row.template !== row.intent.type ||
        (!row.compileTarget && !row.compileFailure)
      ) {
        throw new Error(
          "acceptance positive lacks its declared family or independent compile target",
        );
      }

      positive++;
      positiveFamilies[row.intent.type] = (positiveFamilies[row.intent.type] ?? 0) + 1;
    } else {
      negative++;
    }
  }

  if (
    positive < 500 ||
    negative < 600 ||
    critical === 0 ||
    MAJOR_FAMILIES.some((family) => (positiveFamilies[family] ?? 0) < 50)
  ) {
    throw new Error("acceptance coverage is below the predeclared minimum");
  }

  return { positive, negative, critical, positiveFamilies };
}

function command(args: string[]): void {
  execFileSync("bun", args, { cwd: root, stdio: "inherit" });
}

export async function runGate(run: string, acceptance: string, promote = false) {
  const frozen = join(run, "frozen");
  const scopeBytes = readFileSync(join(run, "scope-freeze.json"));
  const scope = JSON.parse(scopeBytes.toString());
  const { createdAt: _, acceptanceUsed: __, ...identity } = scope;

  if (!isDeepStrictEqual(candidateIdentity(run), identity)) {
    throw new Error("candidate or runtime scope changed after freeze");
  }

  // Consumed records store repository-relative paths, so both must be inside it.
  const recordedAcceptance = toRepository(acceptance);
  const recordedRun = toRepository(run);
  const bytes = readFileSync(acceptance);
  const digest = sha256(bytes);
  const authored = JSON.parse(readFileSync(`${acceptance}.meta.json`, "utf8"));

  if (
    authored.sha256 !== digest ||
    authored.scopeFreezeSha256 !== sha256(scopeBytes) ||
    authored.candidateSha256 !== identity.artifactFileSha256 ||
    !authored.authorship ||
    !authored.independenceLimits ||
    authored.createdAt < scope.createdAt
  ) {
    throw new Error("reserved acceptance metadata does not match this frozen candidate");
  }

  const rows = readCorpus(acceptance) as AcceptanceRow[];
  const coverage = acceptanceCoverage(rows);
  const corpusPath = JSON.parse(readFileSync(join(run, "report.json"), "utf8")).config?.corpus;

  if (!corpusPath) throw new Error("training report lacks corpus provenance");
  const trainingBytes = readFileSync(fromRepository(corpusPath));

  if (sha256(trainingBytes) !== identity.calibrationCorpusSha256) {
    throw new Error("training/calibration corpus changed");
  }

  const trainingTexts = new Set(
    readCorpus(fromRepository(corpusPath)).map((row) => row.text.trim().toLowerCase()),
  );

  if (rows.some((row) => trainingTexts.has(row.text.trim().toLowerCase()))) {
    throw new Error("acceptance wording overlaps the training corpus");
  }

  const previous = readdirSync(resolve(root, "packages/validation/evaluation/reserved"))
    .filter((name) => name.endsWith(".jsonl"))
    .map((name) => {
      const path = resolve(root, "packages/validation/evaluation/reserved", name);

      return { name, path, sha256: sha256(readFileSync(path)) };
    })
    .filter((entry) =>
      existsSync(resolve(root, "packages/validation/evaluation/consumed", `${entry.sha256}.json`)),
    )
    .map((entry) => ({ ...entry, rows: readCorpus(entry.path) as AcceptanceRow[] }));

  const oldTexts = new Set(
    previous.flatMap((entry) => entry.rows.map((row) => row.text.trim().toLowerCase())),
  );

  if (rows.some((row) => oldTexts.has(row.text.trim().toLowerCase()))) {
    throw new Error("reserved wording overlaps previously consumed acceptance");
  }

  // Validate every paired gold target before consuming model acceptance.
  for (const row of rows.filter((row) => row.intent)) {
    const compiled = compile(row.intent!);

    if (
      row.compileFailure
        ? compiled.supported || compiled.code !== row.compileFailure
        : !compiled.supported || !isDeepStrictEqual(compiled.schema, row.compileTarget)
    ) {
      throw new Error(`invalid paired compile gold: ${row.id}`);
    }
  }

  command(["run", "--cwd", "packages/validation", "typecheck"]);
  command(["run", "--cwd", "packages/validation/training", "typecheck"]);
  command(["run", "--cwd", "packages/validation", "test"]);
  command(["run", "--cwd", "packages/validation/training", "test"]);

  for (const name of ["test_model.py", "test_export.py", "test_dataset.py"]) {
    execFileSync(
      resolve(root, "training/cron/.venv/bin/python"),
      [resolve(root, "training/shared", name)],
      { cwd: root, stdio: "inherit" },
    );
  }

  const exported = JSON.parse(readFileSync(join(run, "candidate/export-report.json"), "utf8"));

  for (const entry of Object.values(exported.parity.files) as {
    path: string;
    sha256: string;
  }[]) {
    const path = join(frozen, "parity", entry.path.split("/").at(-1)!);

    if (sha256(readFileSync(path)) !== entry.sha256) {
      throw new Error("parity reference digest mismatch");
    }
  }

  const parity = await verifyParity(frozen);
  const weights = readFileSync(join(frozen, "weights.json"));

  const synthetic = await evaluateRecords(
    readCorpus(fromRepository(corpusPath)).filter((row) => row.split === "heldout"),
    async (row) => parse(row.text, { source: "model", weights }),
  );

  if (!synthetic.syntheticGatePassed) {
    throw new Error("synthetic gate failed; acceptance remains unconsumed");
  }

  const regressions = [];

  for (const entry of previous) {
    const critical = entry.rows.filter((row) => row.critical);
    const failures = [];

    for (const row of critical) {
      const result = await parse(row.text, { source: "model", weights });

      if (row.intent ? !result.ok || !isDeepStrictEqual(result.intent, row.intent) : result.ok) {
        failures.push({ id: row.id, text: row.text, actual: result });
      }
    }

    regressions.push({
      sha256: entry.sha256,
      name: entry.name,
      critical: critical.length,
      failures,
    });
  }

  writeFileSync(
    join(run, "critical-regressions.json"),
    JSON.stringify(regressions, null, 2) + "\n",
  );

  if (regressions.some((entry) => entry.failures.length)) {
    throw new Error("previous critical regression failed; acceptance remains unconsumed");
  }

  // O_EXCL prevents two candidates or concurrent invocations consuming one set.
  const consumed = resolve(root, "packages/validation/evaluation/consumed");

  mkdirSync(consumed, { recursive: true });

  writeFileSync(
    join(consumed, `${digest}.json`),
    JSON.stringify(
      {
        sha256: digest,
        acceptance: recordedAcceptance,
        candidateSha256: identity.artifactFileSha256,
        run: recordedRun,
        consumedAt: new Date().toISOString(),
        authorship: authored.authorship,
        independenceLimits: authored.independenceLimits,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  const predictions: ParseResult[] = [];

  for (const row of rows) predictions.push(await parse(row.text, { source: "model", weights }));
  const metrics = await evaluateRecords(rows, async (_, i) => predictions[i]!);

  const criticalFailures = rows
    .filter(
      (row, i) =>
        row.critical &&
        (row.intent
          ? !predictions[i]!.ok ||
            !isDeepStrictEqual((predictions[i] as { intent?: unknown }).intent, row.intent)
          : predictions[i]!.ok),
    )
    .map((row) => row.id);

  const passed =
    metrics.positiveExact >= 0.9 &&
    metrics.wrongAll <= 0.005 &&
    metrics.wrongAccepted <= 0.005 &&
    metrics.negativeFalseAcceptance <= 0.01 &&
    MAJOR_FAMILIES.every((family) => {
      const count = metrics.families[`positive/${family}`]!;

      return count.exact / count.total >= 0.85;
    }) &&
    criticalFailures.length === 0;

  const report = {
    candidateSha256: identity.artifactFileSha256,
    acceptanceSha256: digest,
    scope,
    coverage,
    parity,
    synthetic,
    regressions,
    metrics,
    criticalFailures,
    authorship: authored.authorship,
    independenceLimits: authored.independenceLimits,
    languagePassed: passed,
    promoted: false,
    packagePassed: false,
  };

  const reportPath = join(run, "gate-report.json");

  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  if (!passed) return report;

  const staged = join(run, "gate-package");

  mkdirSync(staged);

  try {
    for (const name of ["package.json", "README.md", "LICENSE"]) {
      copyFileSync(resolve(root, "packages/validation", name), join(staged, name));
    }

    command([
      "run",
      "packages/validation/scripts/build.ts",
      join(frozen, "weights.gen.ts"),
      join(staged, "dist"),
    ]);

    command(["run", "packages/validation/scripts/check-package.ts", staged]);
    command(["run", "packages/validation/scripts/check-browser.ts", join(staged, "dist")]);
    const size = runSizeGate(join(staged, "dist"));

    Object.assign(report, { packagePassed: true, size });

    if (promote) {
      // Calibration travels inside the artifact. The previous artifact is
      // retained with its actual readiness status, never called known-good.
      const active = resolve(root, "packages/validation/src/model/weights.gen.ts");
      const rollback = join(run, "previous-weights.gen.ts");

      copyFileSync(active, rollback);
      const pending = `${active}.pending`;

      writeFileSync(pending, readFileSync(join(frozen, "weights.gen.ts")), {
        flag: "wx",
      });

      renameSync(pending, active);

      try {
        command(["run", "--cwd", "packages/validation", "check:package"]);
        report.promoted = true;
      } catch (error) {
        writeFileSync(pending, readFileSync(rollback), { flag: "wx" });
        renameSync(pending, active);
        throw error;
      }
    }

    writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  } finally {
    rmSync(staged, { recursive: true, force: true });
  }

  return report;
}

if (import.meta.main) {
  const [mode, run, acceptance] = process.argv.slice(2);

  if (!run || !["prepare", "check", "promote"].includes(mode!)) {
    throw new Error("usage: gate <prepare|check|promote> <run-directory> [reserved.jsonl]");
  }

  if (mode === "prepare") {
    prepareGate(resolve(run));
  } else {
    if (!acceptance) throw new Error("reserved acceptance is required");
    const report = await runGate(resolve(run), resolve(acceptance), mode === "promote");

    console.log(
      JSON.stringify({
        languagePassed: report.languagePassed,
        packagePassed: report.packagePassed,
        promoted: report.promoted,
      }),
    );

    if (!report.languagePassed || !report.packagePassed) process.exitCode = 1;
  }
}
