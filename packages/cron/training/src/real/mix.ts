/**
 * Mix Track C accepted rows into a generator train JSONL at a target RT.
 * RT = n_real / (n_real + n_gen). Held-out / validation stay generator-only.
 */
import { readFileSync, writeFileSync } from "node:fs";

import type { SpeechAct } from "../../../src/model/speech-acts";
import { FAMILIES } from "../corpus/merge";
import type { CorpusRecord } from "../corpus/merge";
import { sha256Hex } from "../fingerprint";
import type { Family } from "../sampler";
import type { AcceptedRealRecord } from "./accepted";

const FAMILY_SET = new Set<string>(FAMILIES);

export function realCountForRt(nGen: number, rt: number): number {
  if (nGen < 1) throw new RangeError("mix: generator train split is empty");
  if (rt <= 0) return 0;
  if (rt >= 1) throw new RangeError("mix: RT must be in (0, 1)");

  return Math.max(1, Math.round((rt / (1 - rt)) * nGen));
}

export function mixRt(nReal: number, nGen: number): number {
  const total = nReal + nGen;

  if (total <= 0) throw new RangeError("mix: empty train split");

  return nReal / total;
}

export function toCorpusRecord(record: AcceptedRealRecord, index: number): CorpusRecord {
  const family = (
    record.family !== undefined && FAMILY_SET.has(record.family) ? record.family : "interval"
  ) as Family;

  const speechAct: SpeechAct =
    record.speechAct === "none" || record.intent === null ? "none" : "request";

  return {
    id: record.id || `real-${String(index).padStart(6, "0")}`,
    split: "train",
    family,
    fingerprint: record.fingerprint ?? sha256Hex(record.text),
    text: record.text,
    intent: record.intent,
    speechAct,
    spans: record.spans.map((span) => ({ ...span })),
    source: "llm",
    template: "real/accepted",
  };
}

export interface MixResult {
  nReal: number;
  nGen: number;
  rt: number;
  fraction: string;
}

export function mixAcceptedIntoTrain(
  trainPath: string,
  accepted: readonly AcceptedRealRecord[],
  rt: number,
): MixResult {
  const lines = readFileSync(trainPath, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const nGen = lines.length;
  const nReal = realCountForRt(nGen, rt);

  if (accepted.length < nReal) {
    throw new RangeError(
      `mix: need ${nReal} accepted real rows for RT=${rt} with ${nGen} generator rows; have ${accepted.length}`,
    );
  }

  const mixed = [
    ...lines,
    ...accepted
      .slice(0, nReal)
      .map((record, index) => JSON.stringify(toCorpusRecord(record, index))),
  ];

  writeFileSync(trainPath, `${mixed.join("\n")}\n`);
  const ratio = mixRt(nReal, nGen);

  return {
    nReal,
    nGen,
    rt: ratio,
    fraction: `${nReal}/${nReal + nGen}`,
  };
}
