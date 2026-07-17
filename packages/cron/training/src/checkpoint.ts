/** Score an epoch with the actual calibrated TypeScript parser, one row at a time. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { readJsonl } from "../../../shared/training/jsonl.js";
import { canonicalizeIntent } from "./canonical-intent.js";
import { loadCompactProtection } from "./compact-protection.js";
import { type CorpusRecord, corpusRecordSchema } from "./corpus/merge.js";
import { replaceJson } from "./corpus/production/publication.js";
import { corpusCodeHashes, fileHash } from "./corpus/production/source-hashes.js";
import { type Engine, loadEngine } from "./evaluate.js";

interface Counts {
  total: number;
  correct: number;
  positiveTotal: number;
  positiveExact: number;
  positiveAccepted: number;
  positiveWrongAccepted: number;
  negativeTotal: number;
  negativeFalseAccepted: number;
}

function counts(): Counts {
  return {
    total: 0,
    correct: 0,
    positiveTotal: 0,
    positiveExact: 0,
    positiveAccepted: 0,
    positiveWrongAccepted: 0,
    negativeTotal: 0,
    negativeFalseAccepted: 0,
  };
}

const rate = (correct: number, total: number): number => (total ? correct / total : 0);

export function scoreCheckpointRecords(
  engine: Engine,
  records: Iterable<CorpusRecord>,
  split: "validation" | "heldout",
) {
  const total = counts();
  const byFamily: Record<string, Counts> = {};
  const byKind: Record<string, Counts> = {};
  const bySpeechAct: Record<string, Counts> = {};

  const failures: { id: string; family: string; category: string }[] = [];
  const sampled = new Map<string, number>();
  const digest = createHash("sha256");

  let rawPositiveExact = 0;

  for (const row of records) {
    if (row.split !== split) continue;
    corpusRecordSchema.parse(row);

    if (total.total === 10000) {
      throw new Error("Checkpoint scoring exceeds the compact corpus limit");
    }

    digest.update(JSON.stringify(row) + "\n");

    const prediction = engine.predict(row.text);
    const parsed = prediction.parse({ overlay: "off" });

    const positive = row.intent !== null;

    const exact =
      positive &&
      parsed.ok &&
      isDeepStrictEqual(canonicalizeIntent(parsed.intent), canonicalizeIntent(row.intent!));

    try {
      const raw = prediction.decodeRaw("off");

      if (
        positive &&
        isDeepStrictEqual(canonicalizeIntent(raw.intent), canonicalizeIntent(row.intent!))
      ) {
        rawPositiveExact++;
      }
    } catch {
      /* Raw refusal is diagnostic, never a successful positive. */
    }

    const correct = positive ? exact : !parsed.ok;
    const family = (byFamily[row.family] ??= counts());
    const kind = (byKind[row.template?.split("/")[0] ?? row.family] ??= counts());
    const speech = (bySpeechAct[row.speechAct ?? (positive ? "request" : "none")] ??= counts());

    for (const count of [total, family, kind, speech]) {
      count.total++;
      if (correct) count.correct++;

      if (positive) {
        count.positiveTotal++;
        if (parsed.ok) count.positiveAccepted++;
        if (exact) count.positiveExact++;
        if (parsed.ok && !exact) count.positiveWrongAccepted++;
      } else {
        count.negativeTotal++;
        if (parsed.ok) count.negativeFalseAccepted++;
      }
    }

    if (!correct) {
      const category = positive
        ? parsed.ok
          ? "wrong-accepted-meaning"
          : "positive-refusal"
        : "negative-false-accept";

      const key = row.family + ":" + category;
      const count = sampled.get(key) ?? 0;

      if (count < 3) failures.push({ id: row.id, family: row.family, category });
      sampled.set(key, count + 1);
    }
  }

  if (!total.positiveTotal || !total.negativeTotal) {
    throw new Error("Checkpoint scoring requires positive and negative support");
  }

  const positiveFamilies = Object.values(byFamily).filter((count) => count.positiveTotal);
  const positiveExactRate = rate(total.positiveExact, total.positiveTotal);

  const macroPositiveExact =
    positiveFamilies.reduce(
      (sum, count) => sum + rate(count.positiveExact, count.positiveTotal),
      0,
    ) / positiveFamilies.length;

  const wrongAcceptedRateAllPositive = rate(total.positiveWrongAccepted, total.positiveTotal);

  const wrongAcceptedRateAcceptedPositive = total.positiveAccepted
    ? rate(total.positiveWrongAccepted, total.positiveAccepted)
    : null;

  const negativeFalseAcceptRate = rate(total.negativeFalseAccepted, total.negativeTotal);

  return {
    version: 1,
    split,
    scope: "synthetic development, not fresh acceptance",
    promotionEvidence: false,
    artifactSha256: engine.artifactSha256,
    selectedRecordsSha256: digest.digest("hex"),
    ...total,
    byFamily,
    byKind,
    bySpeechAct,
    positiveExactRate,
    macroPositiveExact,
    wrongAcceptedRateAllPositive,
    wrongAcceptedRateAcceptedPositive,
    negativeFalseAcceptRate,
    rawPositiveExact,
    rawPositiveExactRate: rate(rawPositiveExact, total.positiveTotal),
    failures,
    syntheticGatePassed:
      positiveExactRate >= 0.95 &&
      positiveFamilies.every((count) => rate(count.positiveExact, count.positiveTotal) >= 0.9) &&
      wrongAcceptedRateAllPositive <= 0.005 &&
      wrongAcceptedRateAcceptedPositive !== null &&
      wrongAcceptedRateAcceptedPositive <= 0.005 &&
      negativeFalseAcceptRate <= 0.01 &&
      Object.values(byFamily).every(
        (count) =>
          !count.negativeTotal || rate(count.negativeFalseAccepted, count.negativeTotal) <= 0.01,
      ),
    peakRssKiB: process.resourceUsage().maxRSS,
  };
}

export async function scoreCheckpoint(
  corpus: string,
  weights: string,
  split: "validation" | "heldout",
) {
  const engine = await loadEngine(resolve(import.meta.dir, "../../dist"), weights);
  const result = scoreCheckpointRecords(engine, readJsonl(corpus) as Iterable<CorpusRecord>, split);

  return {
    ...result,
    corpus: resolve(corpus),
    corpusSha256: fileHash(corpus),
    sourceHashes: corpusCodeHashes(import.meta.url),
  };
}

if (import.meta.main) {
  const [corpus, weights, split, output] = process.argv.slice(2);

  if (
    !corpus ||
    !weights ||
    !output ||
    !["validation", "heldout"].includes(split ?? "") ||
    process.argv.length !== 6
  ) {
    throw new Error("Checkpoint scoring is internal to bun run train");
  }

  // A full training invocation uses only certified compact inputs. Unit tests
  // can exercise the pure scorer without authorizing a training corpus.
  const proof = JSON.parse(
    readFileSync(join(dirname(resolve(corpus)), "preparation.json"), "utf8"),
  );

  if (
    proof.trainingEligible !== true ||
    proof.sha256 !== fileHash(corpus) ||
    !Number.isInteger(proof.rows) ||
    proof.rows < 1 ||
    proof.rows > 10000
  ) {
    throw new Error("Checkpoint scoring requires verified compact preparation");
  }

  const protection = loadCompactProtection();

  if (proof.consumption?.ledgerSha256 !== protection.consumption.ledgerSha256) {
    throw new Error("Checkpoint scoring consumption identity changed");
  }

  const result = await scoreCheckpoint(corpus, weights, split as "validation" | "heldout");

  replaceJson(output, result);
}
