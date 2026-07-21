import { readNumber } from "../comparison/number.js";
import { validateField } from "../compiler/validate.js";
import { type Label, LABEL_COUNT, labelFromId, TYPE_BY_LABEL } from "../model/labels.js";
import type { RawToken } from "../model/tokenizer.js";
/** Model roles own type/operator meaning; TypeScript resolves exact literals and slots. */
import type { Bound, ComparisonIntent, FieldIntent, FieldType } from "../types/intent.js";
import type { ComparisonMeaning, Unit } from "./lexicon.js";
import type { ParseFailureCode } from "./parse.js";

export interface LabeledToken {
  token: RawToken;
  role: number;
  boundary: boolean;
  score: number;
  boundaryScore: number;
}

export interface DecodedIntent {
  intent: FieldIntent;
  confidence: number;
}

export class UnknownRoleError extends RangeError {
  readonly role: number;
  constructor(role: number) {
    super(`unknown role id: ${role}`);
    this.name = "UnknownRoleError";
    this.role = role;
  }
}

export class InputError extends RangeError {
  constructor(
    message: string,
    readonly code: ParseFailureCode = "unrecognized",
  ) {
    super(message);
    this.name = "InputError";
  }
}

export class UnsupportedInputError extends InputError {
  constructor() {
    super("unsupported or non-actionable input");
    this.name = "UnsupportedInputError";
  }
}

const STRING_LIKE = new Set<FieldType>(["string", "email", "url", "uuid"]);
const NUMBER_LIKE = new Set<FieldType>(["number", "integer"]);
const OPERATOR_LABELS = new Set<Label>(["GTE", "GT", "LTE", "LT", "BETWEEN", "EXACT"]);

interface Run {
  label: Label;
  text: string;
  tokens: LabeledToken[];
}

function contentTokens(labeled: readonly LabeledToken[]): LabeledToken[] {
  return labeled.filter((item) => item.token.kind !== 3);
}

function joinRuns(labeled: readonly LabeledToken[], source: string): Run[] {
  const runs: Run[] = [];

  for (const item of contentTokens(labeled)) {
    const label = labelFromId(item.role);
    const last = runs[runs.length - 1];

    if (last !== undefined && last.label === label) {
      last.tokens.push(item);
      last.text = source.slice(last.tokens[0]!.token.start, item.token.end);
      continue;
    }

    runs.push({
      label,
      text: source.slice(item.token.start, item.token.end),
      tokens: [item],
    });
  }

  return runs;
}

function tokenizeRun(text: string): string[] {
  return text.split(/[\s,]+/).filter((token) => token.length > 0);
}

const OPERATOR_MEANINGS: Partial<Record<Label, ComparisonMeaning>> = {
  GTE: { side: "min", inclusive: true },
  GT: { side: "min", inclusive: false },
  LTE: { side: "max", inclusive: true },
  LT: { side: "max", inclusive: false },
  BETWEEN: { side: "between" },
  EXACT: { side: "exact" },
};

function inclusiveLength(bound: Bound, side: "min" | "max"): number {
  if (bound.inclusive) return bound.value;

  return side === "min" ? bound.value + 1 : bound.value - 1;
}

function placeComparison(
  type: FieldType,
  comparison: ComparisonIntent,
  unit: Unit | undefined,
): Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> {
  if (STRING_LIKE.has(type)) {
    if (unit !== "character") {
      throw new InputError("string-like comparisons require a character unit", "unrecognized-slot");
    }

    const fields: Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> = {};

    if (comparison.min !== undefined) fields.minLength = inclusiveLength(comparison.min, "min");
    if (comparison.max !== undefined) fields.maxLength = inclusiveLength(comparison.max, "max");

    return fields;
  }

  if (NUMBER_LIKE.has(type)) {
    if (unit !== undefined) {
      throw new InputError("number-like comparisons do not take a unit", "unrecognized-slot");
    }

    const fields: Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> = {};

    if (comparison.min !== undefined) fields.minimum = comparison.min;
    if (comparison.max !== undefined) fields.maximum = comparison.max;

    return fields;
  }

  throw new InputError("this type does not take a comparison", "unrecognized-slot");
}

function mergeComparison(into: ComparisonIntent, next: ComparisonIntent): void {
  if (next.min !== undefined) {
    if (into.min !== undefined) {
      throw new InputError("conflicting min bounds", "conflicting-bounds");
    }

    into.min = next.min;
  }

  if (next.max !== undefined) {
    if (into.max !== undefined) {
      throw new InputError("conflicting max bounds", "conflicting-bounds");
    }

    into.max = next.max;
  }
}

function clauseFromOperator(
  op: Exclude<ComparisonMeaning, { side: "between" }>,
  value: number,
): ComparisonIntent {
  if (op.side === "exact") {
    return {
      min: { value, inclusive: true },
      max: { value, inclusive: true },
    };
  }

  if (op.side === "min") return { min: { value, inclusive: op.inclusive } };

  return { max: { value, inclusive: op.inclusive } };
}

function takeFollowingNumber(
  runs: readonly Run[],
  index: number,
): { value: number; consumed: number } | null {
  let offset = 1;

  while (index + offset < runs.length && runs[index + offset]!.label === "GLUE") offset += 1;
  const next = runs[index + offset];

  if (next === undefined) return null;

  if (next.label === "NUMBER") {
    const parsed = readNumber(tokenizeRun(next.text), 0);

    if (parsed && parsed.tokenCount === tokenizeRun(next.text).length) {
      return { value: parsed.value, consumed: offset };
    }
  }

  return null;
}

function decodeComparison(runs: readonly Run[]): {
  comparison: ComparisonIntent;
  unit: Unit | undefined;
} {
  const comparison: ComparisonIntent = {};
  let unit: Unit | undefined;
  let saw = false;

  for (let index = 0; index < runs.length; index++) {
    const run = runs[index]!;

    if (run.label === "UNIT_CHARACTER") {
      unit = "character";
      continue;
    }

    const meaning = OPERATOR_MEANINGS[run.label];

    if (meaning === undefined) {
      if (run.label === "NUMBER") throw new InputError("number has no comparison operator");
      continue;
    }

    if (meaning.side === "between") {
      const start = takeFollowingNumber(runs, index);

      if (!start) throw new InputError("incomplete between range");
      index += start.consumed;
      const glue = runs[index + 1];

      if (glue?.label === "GLUE") index += 1;
      const end = takeFollowingNumber(runs, index);

      if (!end) throw new InputError("incomplete between range");
      index += end.consumed;
      if (start.value > end.value) throw new InputError("inverted range", "inverted-range");

      mergeComparison(comparison, {
        min: { value: start.value, inclusive: true },
        max: { value: end.value, inclusive: true },
      });

      saw = true;
      continue;
    }

    const number = takeFollowingNumber(runs, index);

    if (!number) throw new InputError("comparison is missing a number");
    index += number.consumed;
    mergeComparison(comparison, clauseFromOperator(meaning, number.value));
    saw = true;
  }

  if (!saw) throw new InputError("no comparison clause");

  return { comparison, unit };
}

function minScore(labeled: readonly LabeledToken[]): number {
  let min = 1;
  let saw = false;

  for (const item of contentTokens(labeled)) {
    if (item.role < 0 || item.role >= LABEL_COUNT) throw new UnknownRoleError(item.role);

    if (!Number.isFinite(item.score) || item.score < 0 || item.score > 1) {
      throw new RangeError("token confidence must be finite and in [0, 1]");
    }

    const label = labelFromId(item.role);

    if (label === "O") continue;
    saw = true;
    if (item.score < min) min = item.score;
  }

  return saw ? min : 0;
}

/**
 * Resolve labeled tokens into a field intent. Throws {@link UnknownRoleError}
 * on an out-of-range id and `RangeError` when the labels do not name a field.
 */
export function decodeIntent(labeled: readonly LabeledToken[], source: string): DecodedIntent {
  for (const item of labeled) {
    if (!Number.isInteger(item.role) || item.role < 0 || item.role >= LABEL_COUNT) {
      throw new UnknownRoleError(item.role);
    }
  }

  const runs = joinRuns(labeled, source);
  const types: FieldType[] = [];
  const presence: boolean[] = [];
  let nullable = false;
  const comparisonRuns: Run[] = [];

  for (const run of runs) {
    if (run.label === "UNSUPPORTED") throw new UnsupportedInputError();
    if (run.label === "O" || run.label === "GLUE") continue;
    const type = TYPE_BY_LABEL[run.label];

    if (type !== undefined) {
      types.push(type);
      continue;
    }

    if (run.label === "REQUIRED" || run.label === "OPTIONAL") {
      presence.push(run.label === "REQUIRED");
      continue;
    }

    if (run.label === "NULLABLE") {
      nullable = true;
      continue;
    }

    comparisonRuns.push(run);
  }

  if (types.length === 0) throw new InputError("missing field type", "missing-type");
  if (types.length > 1) throw new InputError("multiple field types");

  if (presence.length === 0) {
    throw new InputError("missing required or optional", "missing-presence");
  }

  if (presence.length > 1) throw new InputError("conflicting presence", "conflicting-presence");

  const intent: FieldIntent = { type: types[0]!, required: presence[0]! };

  if (nullable) intent.nullable = true;

  const hasComparison = comparisonRuns.some((run) => OPERATOR_LABELS.has(run.label));

  if (!hasComparison && comparisonRuns.length > 0) throw new InputError("incomplete comparison");

  if (hasComparison) {
    const placed = decodeComparison(runs);
    const fields = placeComparison(intent.type, placed.comparison, placed.unit);

    if (fields.minLength !== undefined) intent.minLength = fields.minLength;
    if (fields.maxLength !== undefined) intent.maxLength = fields.maxLength;
    if (fields.minimum !== undefined) intent.minimum = fields.minimum;
    if (fields.maximum !== undefined) intent.maximum = fields.maximum;
  }

  const invalid = validateField(intent);

  if (invalid) {
    throw new InputError(
      invalid.reason,
      invalid.code === "conflicting-bounds"
        ? "conflicting-bounds"
        : invalid.code === "wrong-slot"
          ? "unrecognized-slot"
          : "unrecognized",
    );
  }

  return { intent, confidence: minScore(labeled) };
}
