/** Replay every consumed acceptance set against the bundled model (regression evidence only). */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { fromRepository } from "../../../shared/training/paths.js";
import type { Acceptance } from "../../evaluation/author.js";
import { weights } from "../../src/model/weights.gen.js";
import { parse } from "../../src/parse.js";
import { hash, readRows } from "./corpus.js";

const consumed = resolve(import.meta.dir, "../../evaluation/consumed");

console.log({ bundledArtifactSha256: hash(JSON.stringify(weights) + "\n") });

for (const name of readdirSync(consumed).filter((n) => /^[a-f0-9]{64}\.json$/u.test(n))) {
  const record = JSON.parse(readFileSync(join(consumed, name), "utf8"));
  const full = fromRepository(record.acceptance);
  const available = existsSync(full);

  const rows = (
    available
      ? readRows(full)
      : JSON.parse(readFileSync(join(consumed, record.sha256 + ".archive.json"), "utf8")).critical
  ) as Acceptance[];

  const tally = {
    positives: 0,
    exact: 0,
    negatives: 0,
    accepted: 0,
    critical: 0,
    criticalFailed: 0,
  };

  for (const row of rows) {
    const result = await parse(row.text, row.context);

    const correct = row.intent
      ? result.ok && isDeepStrictEqual(result.intent, row.intent)
      : !result.ok;

    if (row.intent) {
      tally.positives++;
      if (correct) tally.exact++;
    } else {
      tally.negatives++;
      if (result.ok) tally.accepted++;
    }

    if (row.critical) {
      tally.critical++;
      if (!correct) tally.criticalFailed++;
    }
  }

  console.log({
    acceptance: record.acceptance,
    sha256: record.sha256,
    rows: available ? "full" : "archived critical rows",
    ...tally,
  });
}
