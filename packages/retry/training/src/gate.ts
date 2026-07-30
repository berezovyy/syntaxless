import { execFileSync } from "node:child_process";
/** Frozen, one-shot language acceptance with mathematical schedule targets and package checks. */
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
import { runSizeGate } from "../../scripts/size-gate.js";
import { compile } from "../../src/compile.js";
import { parse } from "../../src/parse.js";
import type { ParseResult } from "../../src/parse.js";
import { hash, readRows } from "./corpus.js";
import { evaluate, FAMILIES } from "./evaluate.js";
import { verifyParity } from "./parity.js";

const root = resolve(import.meta.dir, "../../../..");
const base = resolve(root, "packages/retry");
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

  for (const name of [
    "package.json",
    "tsconfig.json",
    "tsconfig.build.json",
    "training/tsconfig.json",
    "training/package.json",
  ]) {
    sources["packages/retry/" + name] = hash(readFileSync(join(base, name)));
  }

  visit(join(root, "packages/shared"));
  sources["scripts/declarations.ts"] = hash(readFileSync(join(root, "scripts/declarations.ts")));

  for (const path of [
    "training/shared/train.py",
    "training/shared/export.py",
    "training/shared/artifact.py",
    "training/shared/model.py",
    "training/shared/dataset.py",
  ]) {
    sources[path] = hash(readFileSync(join(root, path)));
  }

  sources["scripts/check-package.ts"] = hash(readFileSync(join(root, "scripts/check-package.ts")));

  sources["packages/retry/evaluation/README.md"] = hash(
    readFileSync(join(base, "evaluation/README.md")),
  );

  sources["retry-witnesses"] = hash(readFileSync(join(base, "evaluation/fixtures.ts")));

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
  for (const row of rows) {
    if (!row.intent) continue;
    const result = compile(row.intent, row.compileOptions);
    const normalized = result.supported ? result : { supported: false, code: result.code };

    if (!isDeepStrictEqual(normalized, row.compileTarget)) {
      throw new Error(`invalid paired schedule target: ${row.text}`);
    }
  }
}

function summarize(rows: readonly Acceptance[], sha256: string) {
  return {
    sha256,
    rows: rows.length,
    critical: rows.filter((r) => r.critical),
    textHashes: [...new Set(rows.map((r) => hash(r.text.trim().toLowerCase())))].sort(),
  };
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
    const key = JSON.stringify([row.text.trim().toLowerCase(), row.context]);

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
    ? readdirSync(consumed)
        .filter((name) => /^[a-f0-9]{64}\.json$/u.test(name))
        .map((name) => {
          const record = JSON.parse(readFileSync(join(consumed, name), "utf8"));

          return { ...record, acceptance: fromRepository(record.acceptance) };
        })
    : [];

  const previous = prior.map((entry) => {
    if (entry.archiveSha256) {
      const archived = readFileSync(join(consumed, entry.sha256 + ".archive.json"));

      if (hash(archived) !== entry.archiveSha256) {
        throw new Error("previous acceptance archive changed");
      }

      const summary = JSON.parse(archived.toString()) as {
        sha256: string;
        critical: Acceptance[];
        textHashes: string[];
      };

      if (summary.sha256 !== entry.sha256) throw new Error("acceptance archive provenance changed");

      if (existsSync(entry.acceptance) && hash(readFileSync(entry.acceptance)) !== entry.sha256) {
        throw new Error("previous acceptance changed");
      }

      return summary;
    }

    const priorBytes = readFileSync(entry.acceptance);

    if (hash(priorBytes) !== entry.sha256) throw new Error("previous acceptance changed");

    return summarize(readRows(entry.acceptance) as Acceptance[], entry.sha256);
  });

  const oldTexts = new Set(previous.flatMap((summary) => summary.textHashes));

  if (rows.some((r) => oldTexts.has(hash(r.text.trim().toLowerCase())))) {
    throw new Error("fresh wording overlaps consumed acceptance");
  }

  paired(rows);
  command(["run", "--cwd", "packages/retry", "typecheck"]);
  command(["run", "--cwd", "packages/retry/training", "typecheck"]);
  command(["run", "--cwd", "packages/retry/training", "test"]);
  command(["run", "--cwd", "packages/retry", "test"]);
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

  const regression = { protectedPositives: 0, protectedNegatives: 0 };

  for (const entry of prior) {
    if (!existsSync(entry.acceptance)) continue;

    for (const row of readRows(entry.acceptance)) {
      const baseline = await parse(row.text, row.context);
      const candidate = await parse(row.text, { ...row.context, weights });

      if (row.intent) {
        if (baseline.ok && isDeepStrictEqual(baseline.intent, row.intent)) {
          regression.protectedPositives++;

          if (!candidate.ok || !isDeepStrictEqual(candidate.intent, row.intent)) {
            throw Error("prior positive regression failed; reserved remains unconsumed");
          }
        }
      } else if (!baseline.ok) {
        regression.protectedNegatives++;

        if (candidate.ok) {
          throw Error("prior negative regression failed; reserved remains unconsumed");
        }
      }
    }
  }

  for (const row of previous.flatMap((summary) => summary.critical)) {
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
      "packages/retry/scripts/build.ts",
      join(run, "frozen/weights.gen.ts"),
      join(staged, "dist"),
    ]);

    command(["run", "packages/retry/scripts/check-package.ts", staged]);
    command(["run", "packages/retry/scripts/check-browser.ts", join(staged, "dist")]);
    const size = runSizeGate(join(staged, "dist"));

    mkdirSync(consumed, { recursive: true });
    const archive = JSON.stringify(summarize(rows, digest)) + "\n";

    writeFileSync(join(consumed, digest + ".archive.json"), archive, { flag: "wx" });

    writeFileSync(
      join(consumed, digest + ".json"),
      JSON.stringify(
        {
          sha256: digest,
          acceptance: toRepository(acceptance),
          candidateSha256: scope.artifactFileSha256,
          consumedAt: new Date().toISOString(),
          archiveSha256: hash(archive),
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
      regression,
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
        command(["run", "--cwd", "packages/retry", "check:package"]);
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
