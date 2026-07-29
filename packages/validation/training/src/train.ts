/** One validation training entry point: prepare, train, export, and evaluate. */
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statfsSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { LABELS } from "../../src/model/labels.js";
import { tokenize } from "../../src/model/tokenizer.js";
import { resultFromLabeledTokens } from "../../src/parser/model.js";
import { writeCompact } from "./compact.js";
import { readCorpus } from "./evaluate.js";
import { featurizeFiles, featurizeRecord } from "./featurize.js";
import { freezeCandidate } from "./freeze.js";
import type { CorpusRecord } from "./generate.js";
import { verifyParity } from "./parity.js";

const root = resolve(import.meta.dir, "../../../..");
const python = resolve(root, "training/cron/.venv/bin/python");
const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const prefix = `packages/validation/training/data/synth/compact-v12-25k-${timestamp}`;
const data = prefix + ".jsonl";
const run = `curve25k64-${timestamp}`;
const runDir = resolve(root, "training/validation/runs", run);

function checkSupervision(rows: readonly CorpusRecord[]): void {
  const roles = new Map<string, Set<string>>();

  for (const row of rows) {
    const tokens = tokenize(row.text);
    const features = featurizeRecord(row, tokens);

    if (row.intent !== null) {
      const present = roles.get(row.split) ?? new Set<string>();

      for (const span of row.spans) present.add(span.label);
      roles.set(row.split, present);
    }

    const labeled = tokens.map((token, i) => ({
      token,
      role: token.kind === 3 ? 0 : features.labels[i]!,
      score: 1,
      boundaryScore: 1,
      boundary: features.boundaries[i] === 1,
    }));

    const decoded = resultFromLabeledTokens(labeled, row.text);

    if (
      row.intent === null
        ? decoded.ok
        : !decoded.ok || !isDeepStrictEqual(decoded.intent, row.intent)
    ) {
      throw new Error(`invalid supervision: ${row.text}: ${JSON.stringify(decoded)}`);
    }
  }

  for (const split of ["train", "validation", "heldout"]) {
    for (const label of LABELS.filter((label) => label !== "O" && label !== "UNSUPPORTED")) {
      if (!roles.get(split)?.has(label)) {
        throw new Error(`missing supported role ${label} in ${split}`);
      }
    }
  }
}

function runPython(args: string[]): void {
  const child = spawnSync(python, args, { cwd: root, stdio: "inherit" });

  if (child.status !== 0) throw new Error(`Python phase failed (${child.status}): ${args[0]}`);
}

function storage(): Record<string, number> {
  const paths = [
    ".git",
    "packages/validation/src",
    "packages/validation/training/data/synth",
    "training/validation/runs",
    "packages/validation/dist",
    "node_modules",
    "training/cron/.venv",
  ];

  const resource = process.resourceUsage();
  const sizes: Record<string, number> = {};

  for (const path of paths) {
    sizes[path] = existsSync(resolve(root, path))
      ? Number(
          execFileSync("du", ["-s", "-B1", path], {
            cwd: root,
            encoding: "utf8",
          }).split(/\s/)[0],
        )
      : 0;
  }

  const disk = statfsSync(root);

  sizes.free = disk.bavail * disk.bsize;
  sizes.preparationPeakRssKiB = resource.maxRSS;
  const historyPath = resolve(runDir, "storage.json");
  const history = existsSync(historyPath) ? JSON.parse(readFileSync(historyPath, "utf8")) : [];

  history.push({ timestamp: new Date().toISOString(), sizes });
  writeFileSync(historyPath, JSON.stringify(history, null, 2) + "\n");

  const generated =
    (sizes["packages/validation/training/data/synth"] ?? 0) +
    (sizes["training/validation/runs"] ?? 0);

  if (generated > 512 * 1024 * 1024) {
    throw new Error("active-domain generated storage budget exceeded");
  }

  console.log("storage", sizes);

  return sizes;
}

if (!existsSync(python)) {
  throw new Error(
    "compatible cron Python environment is required; no environment is copied automatically",
  );
}

mkdirSync(runDir, { recursive: true });

if (storage().free! < 512 * 1024 * 1024) {
  throw new Error("insufficient free disk for the active-domain budget");
}

writeCompact(resolve(root, data), 25_000);
checkSupervision(readCorpus(resolve(root, data)));

for (const split of ["train", "validation", "heldout"] as const) {
  const shard = featurizeFiles({
    input: resolve(root, data),
    out: resolve(root, `${prefix}-${split}`),
    split,
  });

  if (shard.skippedLong || shard.skippedEmpty) {
    throw new Error("corpus preparation dropped supervised rows");
  }
}

storage();

runPython([
  "training/shared/train.py",
  "--package",
  "validation",
  "--data-prefix",
  prefix,
  "--corpus",
  data,
  "--run",
  run,
  "--hidden",
  "64",
]);

runPython([
  "training/validation/export.py",
  "--checkpoint",
  `${runDir}/best.pt`,
  "--out",
  `${runDir}/candidate`,
]);

console.log(
  "freeze",
  await freezeCandidate(resolve(root, data), `${runDir}/candidate`, `${runDir}/frozen`),
);

console.log("parity", await verifyParity(`${runDir}/frozen`));

for (const split of ["validation", "heldout"]) {
  execFileSync(
    "bun",
    [
      "run",
      "packages/validation/training/src/evaluate.ts",
      data,
      split,
      `${runDir}/frozen/weights.json`,
      `${runDir}/${split}-runtime.json`,
    ],
    { cwd: root, stdio: "inherit" },
  );
}

const heldout = JSON.parse(readFileSync(resolve(runDir, "heldout-runtime.json"), "utf8"));

storage();

console.log(
  JSON.stringify({
    runDir,
    syntheticGatePassed: heldout.syntheticGatePassed,
    promoted: false,
    remaining: "Reserved acceptance, package checks, and gate-controlled promotion.",
  }),
);
