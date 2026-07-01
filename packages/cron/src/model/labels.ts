/**
 * The one label vocabulary for cron token tagging.
 *
 * Labels describe the render structure only. They are chosen by the `add()`
 * call that inserted a fragment, never recovered from text and never read from
 * the runtime parser. There are no reserved slots: every id is in
 * `[0, LABEL_COUNT)`. An out-of-range predicted id is a hard error, never a
 * guess. Never reorder `LABELS`.
 */
export const LABELS = [
  "O",
  "RECUR",
  "INTERVAL",
  "WEEKDAY",
  "DAYGROUP",
  "DOM",
  "ORDINAL",
  "HOUR",
  "MINUTE",
  "MERIDIEM",
  "BOUND",
  "EXCEPT",
  "COUNT",
  "GLUE",
  "EDGE",
] as const;

export type Label = (typeof LABELS)[number];

export const LABEL_COUNT = LABELS.length;

export const labelId = Object.fromEntries(LABELS.map((name, id) => [name, id])) as Record<
  Label,
  number
>;

/** A predicted role is just a numeric label id. */
export type Role = number;

/** Literal filler: words that only shape syntax, kept literal in fingerprints. */
export const STRUCTURAL_LABELS = ["RECUR", "GLUE", "BOUND", "EXCEPT", "EDGE"] as const;

export type StructuralLabel = (typeof STRUCTURAL_LABELS)[number];

/** Value slots: the semantic payload the model must predict. */
export const CONTENT_LABELS = [
  "INTERVAL",
  "WEEKDAY",
  "DAYGROUP",
  "DOM",
  "ORDINAL",
  "HOUR",
  "MINUTE",
  "MERIDIEM",
  "COUNT",
] as const;

export type ContentLabel = (typeof CONTENT_LABELS)[number];

const STRUCTURAL_SET: ReadonlySet<string> = new Set(STRUCTURAL_LABELS);

/** True when `label` is literal filler rather than a value slot. */
export function isStructural(label: Label): boolean {
  return STRUCTURAL_SET.has(label);
}

/**
 * Map a predicted role id back to a label. Out-of-range ids throw; the caller
 * must never clamp or substitute `"O"` for a corrupt prediction.
 */
export function labelFromId(id: number): Label {
  const label = LABELS[id];

  if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

  return label;
}
