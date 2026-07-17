/** Fixed semantic fixtures, not training inputs or language performance evidence. */
import type { CorpusRecord } from "../../src/corpus/merge.js";

export const checkpointRecords: CorpusRecord[] = [
  {
    id: "monday",
    split: "validation",
    family: "terse",
    fingerprint: "weekday",
    text: "monday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
    speechAct: "request",
    spans: [{ start: 0, end: 6, label: "WEEKDAY_MON", clauseStart: true, actionable: true }],
    template: "single-weekday/bare",
    source: "rule",
  },
  {
    id: "tuesday",
    split: "validation",
    family: "terse",
    fingerprint: "weekday",
    text: "tuesday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["tuesday"] },
    speechAct: "request",
    spans: [{ start: 0, end: 7, label: "WEEKDAY_TUE", clauseStart: true, actionable: true }],
    template: "single-weekday/bare",
    source: "rule",
  },
  {
    id: "negative",
    split: "validation",
    family: "negative",
    fingerprint: "ordinary",
    text: "read",
    intent: null,
    speechAct: "none",
    spans: [{ start: 0, end: 4, label: "O", clauseStart: false, actionable: false }],
    template: "negative/ordinary",
    source: "rule",
  },
];
