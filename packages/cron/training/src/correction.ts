/**
 * Two-intent correction wraps. Gold intent is B; only B's tokens are actionable.
 * Inner role labels stay real on both clauses. Glue comes from WORDS.correction.
 */
import { WEEKDAY_SEQUENCE, WORDS } from "../../src/parser/lexicon.js";
import { normalize } from "../../src/parser/normalize";
import type { ScheduleIntent, Weekday } from "../../src/types/intent";
import { copySentence } from "./discourse";
import type { Family } from "./family";
import { assertValid } from "./family";
import { Sentence } from "./sentence";

export type CorrectionSlot = "A" | "B";

export interface CorrectionFrame {
  id: string;
  split: "train" | "heldout";
  /** Glue and slot order. Gold is always B. */
  pieces: ReadonlyArray<CorrectionSlot | { glue: string }>;
}

const C = WORDS.correction;
const COMMA = WORDS.glue.comma;

export const CORRECTION_FRAMES: readonly CorrectionFrame[] = [
  {
    id: "i-meant",
    split: "train",
    pieces: [{ glue: C.iMeant }, "B", { glue: C.ratherThan }, "A"],
  },
  {
    id: "use-instead",
    split: "train",
    pieces: [{ glue: C.use }, "B", { glue: C.insteadOf }, "A"],
  },
  {
    id: "sorry-not",
    split: "train",
    pieces: [{ glue: C.sorry }, { glue: COMMA }, "B", { glue: COMMA }, { glue: C.not }, "A"],
  },
  {
    id: "not-use",
    split: "train",
    pieces: [{ glue: C.not }, "A", { glue: C.dash }, { glue: C.use }, "B"],
  },
  {
    id: "make-that",
    split: "heldout",
    pieces: [{ glue: C.makeThat }, "B", { glue: COMMA }, { glue: C.not }, "A"],
  },
  {
    id: "correction-rather",
    split: "heldout",
    pieces: [{ glue: C.correction }, "B", { glue: C.ratherThan }, "A"],
  },
  {
    id: "instead-of",
    split: "heldout",
    pieces: [{ glue: C.insteadOf }, "A", { glue: COMMA }, "B"],
  },
];

export interface CorrectionPair {
  a: ScheduleIntent;
  b: ScheduleIntent;
  family: Family;
}

function weekly(weekday: Weekday): ScheduleIntent {
  return { frequency: { unit: "week", interval: 1 }, weekdays: [weekday] };
}

/** Finite, seedless A/B pairs. Same family; selector or time differs. */
export function correctionPairs(): readonly CorrectionPair[] {
  const pairs: CorrectionPair[] = [];

  for (let index = 0; index < WEEKDAY_SEQUENCE.length; index++) {
    const b = WEEKDAY_SEQUENCE[index]!;
    const a = WEEKDAY_SEQUENCE[(index + WEEKDAY_SEQUENCE.length - 1) % WEEKDAY_SEQUENCE.length]!;

    if (a === b) continue;
    pairs.push({ a: weekly(a), b: weekly(b), family: "terse" });
  }

  pairs.push({
    a: { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 0 } },
    b: { frequency: { unit: "day", interval: 1 }, time: { hour: 10, minute: 0 } },
    family: "time-of-day",
  });

  pairs.push({
    a: { frequency: { unit: "minute", interval: 20 } },
    b: { frequency: { unit: "minute", interval: 30 } },
    family: "interval",
  });

  for (const pair of pairs) {
    assertValid(pair.a);
    assertValid(pair.b);
  }

  return pairs;
}

function uniqueFragments(a: Sentence, b: Sentence): boolean {
  if (a.text.length === 0 || b.text.length === 0) return false;
  if (normalize(a.text) === normalize(b.text)) return false;

  return true;
}

/**
 * Wrap A and B. Returns null when fragments collide or a slot is missing.
 * Insertion ranges are recorded at add() time, never recovered from indexOf.
 */
export function wrapCorrection(a: Sentence, b: Sentence, frame: CorrectionFrame): Sentence | null {
  if (!uniqueFragments(a, b)) return null;
  if (!frame.pieces.includes("A") || !frame.pieces.includes("B")) return null;
  const wrapped = new Sentence();
  let rangeA: { start: number; end: number } | undefined;
  let rangeB: { start: number; end: number } | undefined;

  for (const piece of frame.pieces) {
    if (piece === "A" || piece === "B") {
      const source = piece === "A" ? a : b;
      const start = wrapped.text.length;

      copySentence(source, wrapped, piece === "B");
      const end = wrapped.text.length;

      if (start === end) return null;
      if (piece === "A") rangeA = { start, end };
      else rangeB = { start, end };
    } else {
      const tight = piece.glue === COMMA;

      wrapped.add(piece.glue, "O", wrapped.text.length === 0 || tight ? "" : " ", false);
    }
  }

  if (rangeA === undefined || rangeB === undefined) return null;
  if (rangeA.start < rangeB.end && rangeB.start < rangeA.end) return null;

  return wrapped;
}
