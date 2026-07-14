/**
 * Load Track C accepted rows: agree:3, exec:true, roundtrip:true.
 * `rejected.jsonl` and yield/manifest are not training-eligible.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type { Label } from "../../../src/model/labels";
import type { ScheduleIntent } from "../../../src/types/intent";
import type { CorpusProvenance } from "../corpus/provenance";

/** Explicit consent metadata required before a real row may enter training. */
export interface ModelTrainingConsent {
  granted: true;
  scope: "model-training";
  recordedBy: string;
}

export interface AcceptedRealRecord {
  id: string;
  text: string;
  intent: ScheduleIntent | null;
  spans: readonly {
    start: number;
    end: number;
    label: Label;
    clauseStart: boolean;
    actionable?: boolean;
  }[];
  agree: 3;
  exec: true;
  roundtrip: true;
  family?: string;
  fingerprint?: string;
  speechAct?: string;
  /** Optional at collection time; complete-corpus integration requires it. */
  trainingConsent?: ModelTrainingConsent;
  /** Optional at collection time; integration requires human-authored provenance. */
  provenance?: CorpusProvenance;
}

const SKIP = new Set(["rejected.jsonl", "yield.json", "manifest.json"]);

export function isAcceptedReal(value: unknown): value is AcceptedRealRecord {
  if (value === null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;

  return (
    record.agree === 3 &&
    record.exec === true &&
    record.roundtrip === true &&
    typeof record.text === "string" &&
    Array.isArray(record.spans)
  );
}

export function loadAcceptedReal(dir: string): AcceptedRealRecord[] {
  if (!existsSync(dir)) return [];

  const names = readdirSync(dir)
    .filter((name) => name.endsWith(".jsonl") && !SKIP.has(name))
    .sort();

  const records: AcceptedRealRecord[] = [];

  for (const name of names) {
    const text = readFileSync(join(dir, name), "utf8");

    for (const line of text.split("\n")) {
      if (line.trim().length === 0) continue;
      const value = JSON.parse(line) as unknown;

      if (isAcceptedReal(value)) records.push(value);
    }
  }

  return records;
}
