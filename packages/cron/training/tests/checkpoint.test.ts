import { expect, test } from "bun:test";

import { labelId } from "../../src/model/labels.js";
import { scoreCheckpointRecords } from "../src/checkpoint.js";
import type { CorpusRecord } from "../src/corpus/merge.js";
import { loadTestEngine } from "./helpers/candidate-engine.js";
import { checkpointRecords } from "./helpers/checkpoint-records.js";

const rows = checkpointRecords;

test(
  "checkpoint scoring counts actual wrong meanings and negative accepts, " +
    "and doesn't inspect another split",
  async () => {
    const engine = await loadTestEngine(labelId.WEEKDAY_MON, true);
    const untouched = JSON.stringify(rows);

    const report = scoreCheckpointRecords(
      engine,
      [...rows, { split: "train" } as CorpusRecord, { split: "heldout" } as CorpusRecord],
      "validation",
    );

    expect(report.total).toBe(3);
    expect(report.positiveExact).toBe(1);
    expect(report.positiveAccepted).toBe(2);
    expect(report.positiveWrongAccepted).toBe(1);
    expect(report.negativeFalseAccepted).toBe(1);
    expect(report.positiveExactRate).toBe(0.5);
    expect(report.macroPositiveExact).toBe(0.5);
    expect(report.wrongAcceptedRateAllPositive).toBe(0.5);
    expect(report.wrongAcceptedRateAcceptedPositive).toBe(0.5);
    expect(report.negativeFalseAcceptRate).toBe(1);
    expect(report.rawPositiveExact).toBe(1);
    expect(report.bySpeechAct.request!.positiveExact).toBe(1);
    expect(report.byKind["single-weekday"]!.positiveWrongAccepted).toBe(1);

    expect(report.failures.map((row) => row.category)).toEqual([
      "wrong-accepted-meaning",
      "negative-false-accept",
    ]);

    expect(report.syntheticGatePassed).toBe(false);
    expect(report.promotionEvidence).toBe(false);
    expect(JSON.stringify(rows)).toBe(untouched);
  },
);

test(
  "an always-refusing artifact earns no positive " + "accuracy and cannot pass the synthetic gate",
  async () => {
    const engine = await loadTestEngine(labelId.O, true);
    const report = scoreCheckpointRecords(engine, rows, "validation");

    expect(report.positiveExactRate).toBe(0);
    expect(report.macroPositiveExact).toBe(0);
    expect(report.positiveAccepted).toBe(0);
    expect(report.wrongAcceptedRateAcceptedPositive).toBeNull();
    expect(report.negativeFalseAcceptRate).toBe(0);
    expect(report.correct).toBe(1);
    expect(report.syntheticGatePassed).toBe(false);

    expect(() =>
      scoreCheckpointRecords(
        engine,
        rows.filter((row) => row.intent),
        "validation",
      ),
    ).toThrow("positive and negative support");
  },
);
