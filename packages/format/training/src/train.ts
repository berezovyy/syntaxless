/** One bounded coordinator; exports are inactive until the full release gate passes. */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, statfsSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { writeCorpus } from "./corpus.js";
import { FAMILIES } from "./evaluate.js";
import { featurize } from "./featurize.js";
import { freeze } from "./freeze.js";
import { verifyParity } from "./parity.js";

const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const root = resolve(import.meta.dir, "../../../..");
const prefix = resolve(root, "packages/format/training/data/synth", "compact-v6-8k-" + timestamp);
const corpus = prefix + ".jsonl";

const run = "groups32-" + timestamp;
const directory = resolve(root, "training/format/runs", run);

const python = resolve(root, "training/cron/.venv/bin/python");

mkdirSync(directory, { recursive: true });

function storage() {
  const sizes = Object.fromEntries(
    ["packages/format", "training/format/runs"].map((path) => [
      path,
      Number(
        execFileSync("du", ["-s", "-B1", path], { cwd: root, encoding: "utf8" }).split(/\s/)[0],
      ),
    ]),
  );

  if (Object.values(sizes).reduce((a, b) => a + b, 0) > 512 * 1024 * 1024) {
    throw new Error("format storage budget exceeded");
  }

  const fs = statfsSync(root);

  return {
    allocated: sizes,
    free: fs.bavail * fs.bsize,
    preparationPeakRssKiB: process.resourceUsage().maxRSS,
  };
}

const history = [{ timestamp: new Date().toISOString(), ...storage() }];
const rows = writeCorpus(corpus, 8000);

for (const split of ["train", "validation", "heldout"] as const) {
  for (const family of FAMILIES) {
    if (!rows.some((row) => row.split === split && row.intent && row.family === family)) {
      throw new Error(`missing positive ${family} coverage in ${split}`);
    }
  }

  featurize(
    rows.filter((row) => row.split === split),
    prefix + "-" + split,
  );
}

history.push({ timestamp: new Date().toISOString(), ...storage() });
writeFileSync(resolve(directory, "storage.json"), JSON.stringify(history, null, 2) + "\n");

execFileSync(
  python,
  [
    "training/shared/train.py",
    "--package",
    "format",
    "--data-prefix",
    prefix,
    "--corpus",
    corpus,
    "--run",
    run,
    "--hidden",
    "32",
  ],
  { cwd: root, stdio: "inherit" },
);

execFileSync(
  python,
  [
    "training/shared/export.py",
    "--package",
    "format",
    "--checkpoint",
    resolve(directory, "best.pt"),
    "--out",
    resolve(directory, "candidate"),
  ],
  { cwd: root, stdio: "inherit" },
);

console.log(
  "freeze",
  (await freeze(corpus, resolve(directory, "candidate"), resolve(directory, "frozen"))).calibration
    .confidenceFloor,
);

console.log("parity", await verifyParity(resolve(directory, "frozen")));

for (const split of ["validation", "heldout"]) {
  execFileSync(
    "bun",
    [
      "run",
      "packages/format/training/src/evaluate.ts",
      corpus,
      split,
      resolve(directory, "frozen/weights.json"),
      resolve(directory, split + "-runtime.json"),
    ],
    { cwd: root, stdio: "inherit" },
  );
}

history.push({ timestamp: new Date().toISOString(), ...storage() });
writeFileSync(resolve(directory, "storage.json"), JSON.stringify(history, null, 2) + "\n");

console.log({
  run: directory,
  syntheticGatePassed: JSON.parse(readFileSync(resolve(directory, "heldout-runtime.json"), "utf8"))
    .syntheticGatePassed,
  promoted: false,
});
