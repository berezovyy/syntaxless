/**
 * Utterance-level speech-act classes predicted alongside token roles.
 * Never reorder. Out-of-range ids are a hard error, never `"none"`.
 *
 * This is not schedule shape; see SCHEDULE_KINDS. Schedule-kind remains the
 * shape head (weekly vs monthly). Speech-act is the request / not-a-request
 * head. Do not union them.
 */
export const SPEECH_ACTS = [
  "none",
  "request",
  "descriptive",
  "question",
  "quoted",
  "correction",
  "one-off",
  "cancel",
  "incomplete",
] as const;

export type SpeechAct = (typeof SPEECH_ACTS)[number];

export const SPEECH_ACT_COUNT = SPEECH_ACTS.length;

/** Decision head. `other` refuses. Corpus speech acts collapse into these three. */
export const DECISION_ACTS = ["request", "correction", "other"] as const;

export function isDecisionVocabulary(labels: readonly string[]): boolean {
  return (
    labels.length === DECISION_ACTS.length && DECISION_ACTS.every((name, id) => labels[id] === name)
  );
}

export const speechActId = Object.fromEntries(SPEECH_ACTS.map((name, id) => [name, id])) as Record<
  SpeechAct,
  number
>;

/**
 * Map a predicted speech-act id back to a class name. Out-of-range ids throw;
 * the caller must never clamp or substitute `"none"` for a corrupt prediction.
 */
export function speechActFromId(id: number): SpeechAct {
  const act = SPEECH_ACTS[id];

  if (act === undefined) throw new RangeError(`speech-act id out of range: ${id}`);

  return act;
}
