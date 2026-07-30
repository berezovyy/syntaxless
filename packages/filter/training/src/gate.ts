import { Database } from "bun:sqlite";
import { execFileSync } from "node:child_process";
/** Frozen, one-shot language acceptance plus actual SQLite and package verification. */
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
import type { Acceptance } from "../../evaluation/author.js";
import { WITNESSES } from "../../evaluation/fixtures.js";
import { runSizeGate } from "../../scripts/size-gate.js";
import { compile } from "../../src/compile.js";
import { evaluateTruth } from "../../src/evaluate.js";
import { parse } from "../../src/parse.js";
import type { ParseResult } from "../../src/parse.js";
import { hash, readRows } from "./corpus.js";
import { evaluate, FAMILIES } from "./evaluate.js";
import { verifyParity } from "./parity.js";

const root = resolve(import.meta.dir, "../../../..");
const base = resolve(root, "packages/filter");
const consumed = join(base, "evaluation/consumed");

function identity(run: string) {
  const frozen = join(run, "frozen");
  const manifest = JSON.parse(readFileSync(join(frozen, "freeze.json"), "utf8"));

  if (
    hash(readFileSync(join(frozen, "weights.json"))) !== manifest.artifactFileSha256 ||
    hash(readFileSync(join(frozen, "weights.gen.ts"))) !== manifest.sourceFileSha256
  ) {
    throw new Error("frozen artifact changed");
  }

  const sources: Record<string, string> = {};

  function visit(path: string) {
    for (const entry of readdirSync(path, { withFileTypes: true })) {
      const file = join(path, entry.name);

      if (entry.isDirectory()) {
        visit(file);
      } else if (entry.name.endsWith(".ts") && entry.name !== "weights.gen.ts") {
        sources[file.slice(root.length + 1)] = hash(readFileSync(file));
      }
    }
  }

  visit(join(base, "src"));
  visit(join(base, "scripts"));
  visit(join(base, "training/src"));
  visit(join(root, "packages/shared"));
  sources["scripts/declarations.ts"] = hash(readFileSync(join(root, "scripts/declarations.ts")));
  sources["scripts/check-package.ts"] = hash(readFileSync(join(root, "scripts/check-package.ts")));

  for (const path of [
    "training/shared/train.py",
    "training/shared/export.py",
    "training/shared/artifact.py",
    "training/shared/model.py",
    "training/shared/dataset.py",
  ]) {
    sources[path] = hash(readFileSync(join(root, path)));
  }

  sources["packages/filter/evaluation/README.md"] = hash(
    readFileSync(join(base, "evaluation/README.md")),
  );

  sources["filter-witnesses"] = hash(readFileSync(join(base, "evaluation/fixtures.ts")));

  return {
    artifactFileSha256: manifest.artifactFileSha256,
    corpusSha256: manifest.corpusSha256,
    exportReportSha256: hash(readFileSync(join(run, "candidate/export-report.json"))),
    sourceSha256: hash(JSON.stringify(Object.entries(sources).sort())),
    families: FAMILIES,
  };
}

const command = (args: string[]) => execFileSync("bun", args, { cwd: root, stdio: "inherit" });

function paired(rows: readonly Acceptance[]): void {
  const db = new Database(":memory:");

  try {
    db.exec("CREATE TABLE records (id INTEGER, amount ANY, label ANY, flag ANY) STRICT");
    const insert = db.prepare("INSERT INTO records VALUES (?, ?, ?, ?)");

    for (const [i, row] of WITNESSES.entries()) {
      insert.run(
        i,
        row.amount as number | null,
        row.label as string | null,
        row.flag === null ? null : Number(row.flag),
      );
    }

    for (const row of rows) {
      if (!row.intent) continue;
      const result = compile(row.intent, row.context);

      if (!result.supported) throw new Error("positive gold does not compile");

      const actual = db
        .prepare(`SELECT ${result.sql} AS truth FROM records ORDER BY id`)
        .all(...result.parameters) as { truth: number | null }[];

      const sql = actual.map((value) => (value.truth === null ? null : Boolean(value.truth)));

      const js = WITNESSES.map((w) =>
        evaluateTruth(
          row.intent!,
          Object.fromEntries(
            row.context.fields.map((field) => [field.id, w[field.column as string]]),
          ),
          row.context,
        ),
      );

      if (!isDeepStrictEqual(sql, row.compileTarget) || !isDeepStrictEqual(js, row.compileTarget)) {
        throw new Error(`invalid paired gold: ${row.text}`);
      }
    }
  } finally {
    db.close();
  }
}

export async function gate(run: string, acceptance: string) {
  const scopeBytes = readFileSync(join(run, "scope-freeze.json"));
  const scope = JSON.parse(scopeBytes.toString());

  const { createdAt, ...frozenIdentity } = scope;

  if (!isDeepStrictEqual(identity(run), frozenIdentity)) {
    throw new Error("scope, source or artifact changed after freeze");
  }

  const bytes = readFileSync(acceptance);
  const digest = hash(bytes);
  const meta = JSON.parse(readFileSync(acceptance + ".meta.json", "utf8"));

  if (
    meta.sha256 !== digest ||
    meta.candidateSha256 !== scope.artifactFileSha256 ||
    meta.scopeFreezeSha256 !== hash(scopeBytes) ||
    meta.createdAt < createdAt ||
    !meta.authorship ||
    !meta.independenceLimits ||
    meta.trainingEligible !== false
  ) {
    throw new Error("acceptance provenance mismatch");
  }

  const rows = readRows(acceptance) as Acceptance[];
  const positive = rows.filter((r) => r.intent);
  const negative = rows.filter((r) => !r.intent);
  const seen = new Set<string>();

  for (const row of rows) {
    const key = row.text.trim().toLowerCase();

    if (
      row.trainingEligible !== false ||
      row.split !== "heldout" ||
      typeof row.critical !== "boolean" ||
      !key ||
      seen.has(key)
    ) {
      throw new Error("acceptance must be unique and evaluation-only");
    }

    seen.add(key);
  }

  if (
    positive.length < 500 ||
    negative.length < 600 ||
    !rows.some((r) => r.critical) ||
    FAMILIES.some((f) => positive.filter((r) => r.family === f).length < 50)
  ) {
    throw new Error("acceptance coverage below declared minimum");
  }

  const corpus = JSON.parse(readFileSync(join(run, "report.json"), "utf8")).config.corpus;

  if (hash(readFileSync(corpus)) !== scope.corpusSha256) throw new Error("corpus changed");

  const training = readRows(corpus);
  const trainingTexts = new Set(training.map((r) => r.text.trim().toLowerCase()));

  if (rows.some((r) => trainingTexts.has(r.text.trim().toLowerCase()))) {
    throw new Error("acceptance overlaps training wording");
  }

  const prior = existsSync(consumed)
    ? readdirSync(consumed).map((name) => JSON.parse(readFileSync(join(consumed, name), "utf8")))
    : [];

  const previous = prior.map((entry) => {
    const path = fromRepository(entry.acceptance);

    if (hash(readFileSync(path)) !== entry.sha256) throw new Error("previous acceptance changed");

    return readRows(path) as Acceptance[];
  });

  const oldTexts = new Set(previous.flatMap((rs) => rs.map((r) => r.text.trim().toLowerCase())));

  if (rows.some((r) => oldTexts.has(r.text.trim().toLowerCase()))) {
    throw new Error("fresh wording overlaps consumed acceptance");
  }

  paired(rows);
  command(["run", "--cwd", "packages/filter", "typecheck"]);
  command(["run", "--cwd", "packages/filter/training", "typecheck"]);
  command(["run", "--cwd", "packages/filter", "test"]);
  command(["run", "--cwd", "packages/filter/training", "test"]);
  const exported = JSON.parse(readFileSync(join(run, "candidate/export-report.json"), "utf8"));

  for (const file of Object.values(exported.parity.files) as {
    path: string;
    sha256: string;
  }[]) {
    if (
      hash(readFileSync(join(run, "frozen/parity", file.path.split("/").at(-1)!))) !== file.sha256
    ) {
      throw new Error("parity reference changed");
    }
  }

  const parity = await verifyParity(join(run, "frozen"));
  const weights = readFileSync(join(run, "frozen/weights.json"));

  const synthetic = await evaluate(
    training.filter((r) => r.split === "heldout"),
    async (r) => parse(r.text, { ...r.context, weights }),
  );

  if (!synthetic.syntheticGatePassed) {
    throw new Error("synthetic gate failed; reserved remains unconsumed");
  }

  for (const row of previous.flatMap((rs) => rs.filter((r) => r.critical))) {
    const result = await parse(row.text, { ...row.context, weights });

    if (row.intent ? !result.ok || !isDeepStrictEqual(result.intent, row.intent) : result.ok) {
      throw new Error("prior critical regression failed; reserved remains unconsumed");
    }
  }

  // Verify infrastructure and public reference before consuming reserved evidence.
  const staged = join(run, "gate-package");

  mkdirSync(staged);

  try {
    for (const name of ["package.json", "README.md", "LICENSE"]) {
      copyFileSync(join(base, name), join(staged, name));
    }

    command([
      "run",
      "packages/filter/scripts/build.ts",
      join(run, "frozen/weights.gen.ts"),
      join(staged, "dist"),
    ]);

    command(["run", "packages/filter/scripts/check-package.ts", staged]);
    command(["run", "packages/filter/scripts/check-browser.ts", join(staged, "dist")]);
    const size = runSizeGate(join(staged, "dist"));

    mkdirSync(consumed, { recursive: true });

    writeFileSync(
      join(consumed, digest + ".json"),
      JSON.stringify(
        {
          sha256: digest,
          acceptance: toRepository(acceptance),
          candidateSha256: scope.artifactFileSha256,
          consumedAt: new Date().toISOString(),
          authorship: meta.authorship,
          independenceLimits: meta.independenceLimits,
        },
        null,
        2,
      ) + "\n",
      { flag: "wx" },
    );

    const predictions: ParseResult[] = [];

    for (const row of rows) predictions.push(await parse(row.text, { ...row.context, weights }));
    const metrics = await evaluate(rows, async (_, i) => predictions[i]!);

    const criticalFailures = rows.flatMap((row, i) =>
      row.critical &&
      (row.intent
        ? !predictions[i]!.ok ||
          !isDeepStrictEqual((predictions[i] as { intent?: unknown }).intent, row.intent)
        : predictions[i]!.ok)
        ? [row.text]
        : [],
    );

    const languagePassed =
      metrics.positiveExact >= 0.9 &&
      metrics.wrongAll <= 0.005 &&
      metrics.wrongAccepted <= 0.005 &&
      metrics.negativeFalseAcceptance <= 0.01 &&
      FAMILIES.every((f) => {
        const c = metrics.families["positive/" + f]!;

        return c.exact / c.total >= 0.85;
      }) &&
      !criticalFailures.length;

    const report = {
      scope,
      acceptanceSha256: digest,
      authorship: meta.authorship,
      independenceLimits: meta.independenceLimits,
      synthetic,
      parity,
      size,
      metrics,
      criticalFailures,
      languagePassed,
      packagePassed: true,
      browserPassed: true,
      promoted: false,
    };

    const reportPath = join(run, "gate-report.json");

    writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

    if (languagePassed) {
      const active = join(base, "src/model/weights.gen.ts");
      const pending = active + ".pending";
      const rollback = join(run, "previous-weights.gen.ts");

      copyFileSync(active, rollback);

      writeFileSync(pending, readFileSync(join(run, "frozen/weights.gen.ts")), {
        flag: "wx",
      });

      renameSync(pending, active);

      try {
        command(["run", "--cwd", "packages/filter", "check:package"]);
        report.promoted = true;
      } catch (error) {
        writeFileSync(pending, readFileSync(rollback), { flag: "wx" });
        renameSync(pending, active);
        throw error;
      }

      writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
    }

    return report;
  } finally {
    rmSync(staged, { recursive: true, force: true });
  }
}

if (import.meta.main) {
  const [mode, run, acceptance] = process.argv.slice(2);

  if (!run) throw new Error("usage: gate <prepare|promote> <run> [acceptance]");

  if (mode === "prepare") {
    writeFileSync(
      join(run, "scope-freeze.json"),
      JSON.stringify({ ...identity(run), createdAt: new Date().toISOString() }, null, 2) + "\n",
      { flag: "wx" },
    );
  } else if (mode === "promote" && acceptance) {
    const result = await gate(resolve(run), resolve(acceptance));

    console.log({ languagePassed: result.languagePassed, promoted: result.promoted });
    if (!result.promoted) process.exitCode = 1;
  } else {
    throw new Error("usage: gate <prepare|promote> <run> [acceptance]");
  }
}
