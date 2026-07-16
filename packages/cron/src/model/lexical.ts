/**
 * Closed-class channel for the factorized trunk.
 * The id is a feature, not a decode override: the tagger still predicts the label.
 */
import { LEXICON } from "../parser/lexicon.js";

export const LEXICON_CLASSES = 12;

const CLASS_BY_LABEL: readonly [string, number][] = [
  ["WEEKDAY_", 1],
  ["DAYGROUP_", 2],
  ["UNIT_", 3],
  ["ORDINAL_", 4],
  ["BOUND_", 5],
  ["EXCEPT", 6],
  ["MERIDIEM_", 7],
  ["RECUR", 8],
  ["GLUE", 9],
  ["COUNT", 10],
  ["DOM_LIT", 11],
];

/** 0 when the folded token is not in the runtime lexicon. */
export function lexiconClassId(text: string): number {
  const entry = LEXICON[text.toLowerCase()];

  // Plain-object lookup also sees Object.prototype keys ("constructor").
  if (entry === undefined || typeof entry !== "object" || typeof entry.label !== "string") return 0;
  const label = entry.label;

  for (const [prefix, id] of CLASS_BY_LABEL) {
    if (label === prefix || label.startsWith(prefix)) return id;
  }

  return 0;
}
