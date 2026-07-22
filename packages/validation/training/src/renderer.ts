/**
 * Deterministic renderer: {@link FieldIntent} -> surface text + spans.
 *
 * Every fragment is inserted through {@link Sentence.add}, which is the single
 * source of span labels. The renderer never parses text and never derives a
 * label from a rendered string: the label chosen at insertion is final.
 *
 * All constraint surface words come from the runtime lexicon
 * (`../../src/parser/lexicon.ts`). Templates are lexicon surfaces, not
 * ad-hoc strings that cannot round-trip through the rule parser.
 */
import { type Label, TYPE_LABELS } from "../../src/model/labels";
import {
  COMPARISON_ALIASES,
  type ComparisonMeaning,
  NULLABLE_ALIASES,
  NUMBER_WORDS,
  PRESENCE_ALIASES,
  TYPE_ALIASES,
  UNIT_ALIASES,
} from "../../src/parser/lexicon";
import type { Bound, FieldIntent, FieldType } from "../../src/types/intent";
import type { Rng } from "./rng";
import { Sentence, type Span } from "./sentence";

export interface RenderedExample {
  text: string;
  spans: Span[];
  /** `${kind}/${template}`; recorded as the example's template name. */
  template: string;
  style: string;
}

export type NumberStyle = "digits" | "words";

export type RenderOrder =
  | "presence-type"
  | "type-presence"
  | "presence-glue-type"
  | "type-glue-presence"
  | "presence-type-glue-comparison"
  | "presence-type-comparison"
  | "type-presence-comparison"
  | "comparison-presence-type"
  | "comparison-type-presence"
  | "type-comparison-presence"
  | "type-copula-presence-comparison";

export type ComparisonOp = Exclude<ComparisonMeaning, { side: "between" } | { side: "exact" }>;

/**
 * Spoken comparison as the renderer should emit it. Length exclusive values
 * are the pre-widen numbers (`more than 10 characters` → intent minLength 11).
 */
export interface SpokenComparison {
  min?: { value: number; inclusive: boolean };
  max?: { value: number; inclusive: boolean };
  between?: { min: number; max: number };
  exact?: number;
  unit?: boolean;
}

export interface RenderOptions {
  template?: string;
  style?: string;
  order?: RenderOrder;
  typeSurface?: string;
  presenceSurface?: string;
  nullableSurface?: string;
  minSurface?: string;
  maxSurface?: string;
  betweenSurface?: string;
  exactSurface?: string;
  unitSurface?: string;
  glueSurface?: string;
  numberStyle?: NumberStyle;
  spoken?: SpokenComparison;
  nullable?: boolean;
}

const GLUE_SURFACES = ["with"] as const;

const NUMBER_WORD_BY_VALUE: Readonly<Record<number, string>> = (() => {
  const preferred: Record<number, string> = {};

  for (const [word, value] of Object.entries(NUMBER_WORDS)) {
    const current = preferred[value];

    if (current === undefined) {
      preferred[value] = word;
      continue;
    }

    const hyphenated = word.includes("-") && !current.includes("-");
    const oneHundred = word === "one hundred" && current === "hundred";

    if (hyphenated || oneHundred) preferred[value] = word;
  }

  return preferred;
})();

function sameMeaning(left: ComparisonMeaning, right: ComparisonMeaning): boolean {
  if (left.side !== right.side) return false;
  if (left.side === "between" || left.side === "exact") return true;

  return left.inclusive === (right as ComparisonOp).inclusive;
}

export function surfacesForType(type: FieldType): string[] {
  return Object.entries(TYPE_ALIASES)
    .filter(([, value]) => value === type)
    .map(([surface]) => surface);
}

export function surfacesForPresence(required: boolean): string[] {
  return Object.entries(PRESENCE_ALIASES)
    .filter(([, value]) => value === required)
    .map(([surface]) => surface);
}

export function surfacesForUnit(): string[] {
  return Object.keys(UNIT_ALIASES);
}

export function surfacesForNullable(): string[] {
  return Object.keys(NULLABLE_ALIASES);
}

export function surfacesForComparison(meaning: ComparisonMeaning): string[] {
  return Object.entries(COMPARISON_ALIASES)
    .filter(([, value]) => sameMeaning(value, meaning))
    .map(([surface]) => surface);
}

export function comparisonLabel(meaning: ComparisonMeaning): Label {
  if (meaning.side === "between") return "BETWEEN";
  if (meaning.side === "exact") return "EXACT";
  if (meaning.side === "min") return meaning.inclusive ? "GTE" : "GT";

  return meaning.inclusive ? "LTE" : "LT";
}

export function formatNumber(value: number, style: NumberStyle): string {
  if (style === "digits" || !Number.isInteger(value) || value < 0 || value > 100) {
    return String(value);
  }

  return NUMBER_WORD_BY_VALUE[value] ?? String(value);
}

function pick<T>(items: readonly T[], rng: Rng | undefined, fallback: T): T {
  if (items.length === 0) return fallback;
  if (rng === undefined) return items[0]!;

  return rng.pick(items);
}

function requireSurface(surface: string, table: Record<string, unknown>, kind: string): void {
  if (!(surface in table)) {
    throw new RangeError(`renderer ${kind} surface is not in the lexicon: ${surface}`);
  }
}

function addNumber(sentence: Sentence, value: number, style: NumberStyle): void {
  sentence.add(formatNumber(value, style), "NUMBER");
}

function addUnit(sentence: Sentence, surface: string): void {
  requireSurface(surface, UNIT_ALIASES, "unit");
  sentence.add(surface, "UNIT_CHARACTER");
}

function addOp(sentence: Sentence, meaning: ComparisonMeaning, surface: string): void {
  requireSurface(surface, COMPARISON_ALIASES, "comparison");

  if (!sameMeaning(COMPARISON_ALIASES[surface]!, meaning)) {
    throw new RangeError(`comparison surface ${surface} does not match ${JSON.stringify(meaning)}`);
  }

  sentence.add(surface, comparisonLabel(meaning));
}

function addBound(
  sentence: Sentence,
  meaning: ComparisonOp,
  surface: string,
  value: number,
  numberStyle: NumberStyle,
  unitSurface: string | undefined,
): void {
  addOp(sentence, meaning, surface);
  addNumber(sentence, value, numberStyle);
  if (unitSurface !== undefined) addUnit(sentence, unitSurface);
}

function addBetween(
  sentence: Sentence,
  surface: string,
  min: number,
  max: number,
  numberStyle: NumberStyle,
  unitSurface: string | undefined,
): void {
  addOp(sentence, { side: "between" }, surface);
  addNumber(sentence, min, numberStyle);
  sentence.add("and", "GLUE");
  addNumber(sentence, max, numberStyle);
  if (unitSurface !== undefined) addUnit(sentence, unitSurface);
}

function addExact(
  sentence: Sentence,
  surface: string,
  value: number,
  numberStyle: NumberStyle,
  unitSurface: string | undefined,
): void {
  addOp(sentence, { side: "exact" }, surface);
  addNumber(sentence, value, numberStyle);
  if (unitSurface !== undefined) addUnit(sentence, unitSurface);
}

function addComparison(
  sentence: Sentence,
  spoken: SpokenComparison,
  options: {
    minSurface: string;
    maxSurface: string;
    betweenSurface: string;
    exactSurface: string;
    unitSurface: string | undefined;
    numberStyle: NumberStyle;
  },
): void {
  sentence.clause();

  if (spoken.between !== undefined) {
    addBetween(
      sentence,
      options.betweenSurface,
      spoken.between.min,
      spoken.between.max,
      options.numberStyle,
      options.unitSurface,
    );

    return;
  }

  if (spoken.exact !== undefined) {
    addExact(
      sentence,
      options.exactSurface,
      spoken.exact,
      options.numberStyle,
      options.unitSurface,
    );

    return;
  }

  if (spoken.min !== undefined) {
    addBound(
      sentence,
      { side: "min", inclusive: spoken.min.inclusive },
      options.minSurface,
      spoken.min.value,
      options.numberStyle,
      spoken.max === undefined ? options.unitSurface : undefined,
    );
  }

  if (spoken.min !== undefined && spoken.max !== undefined) {
    sentence.add("and", "GLUE");
  }

  if (spoken.max !== undefined) {
    addBound(
      sentence,
      { side: "max", inclusive: spoken.max.inclusive },
      options.maxSurface,
      spoken.max.value,
      options.numberStyle,
      options.unitSurface,
    );
  }
}

function addPresence(sentence: Sentence, required: boolean, surface: string): void {
  requireSurface(surface, PRESENCE_ALIASES, "presence");

  if (PRESENCE_ALIASES[surface] !== required) {
    throw new RangeError(`presence surface ${surface} does not match required=${required}`);
  }

  sentence.add(surface, required ? "REQUIRED" : "OPTIONAL");
}

function addType(sentence: Sentence, type: FieldType, surface: string): void {
  requireSurface(surface, TYPE_ALIASES, "type");

  if (TYPE_ALIASES[surface] !== type) {
    throw new RangeError(`type surface ${surface} does not match ${type}`);
  }

  sentence.add(surface, TYPE_LABELS[type]);
}

function addNullable(sentence: Sentence, surface: string): void {
  requireSurface(surface, NULLABLE_ALIASES, "nullable");
  sentence.add(surface, "NULLABLE");
}

function spokenFromIntent(intent: FieldIntent): SpokenComparison | undefined {
  if (intent.minLength !== undefined || intent.maxLength !== undefined) {
    const spoken: SpokenComparison = { unit: true };

    if (intent.minLength !== undefined) {
      spoken.min = { value: intent.minLength, inclusive: true };
    }

    if (intent.maxLength !== undefined) {
      spoken.max = { value: intent.maxLength, inclusive: true };
    }

    if (
      intent.minLength !== undefined &&
      intent.maxLength !== undefined &&
      intent.minLength === intent.maxLength
    ) {
      return { exact: intent.minLength, unit: true };
    }

    return spoken;
  }

  if (intent.minimum !== undefined || intent.maximum !== undefined) {
    const spoken: SpokenComparison = { unit: false };

    if (intent.minimum !== undefined) spoken.min = { ...intent.minimum };
    if (intent.maximum !== undefined) spoken.max = { ...intent.maximum };

    if (
      intent.minimum !== undefined &&
      intent.maximum !== undefined &&
      intent.minimum.value === intent.maximum.value &&
      intent.minimum.inclusive &&
      intent.maximum.inclusive
    ) {
      return { exact: intent.minimum.value, unit: false };
    }

    return spoken;
  }

  return undefined;
}

function defaultOrder(spoken: SpokenComparison | undefined): RenderOrder {
  return spoken === undefined ? "presence-type" : "presence-type-glue-comparison";
}

/**
 * Render one field constraint as lexicon English. Surfaces default to the
 * first matching lexicon row; pass `rng` to sample among aliases.
 */
export function renderField(
  intent: FieldIntent,
  rng?: Rng,
  options: RenderOptions = {},
): RenderedExample {
  const typeSurface = options.typeSurface ?? pick(surfacesForType(intent.type), rng, intent.type);

  const presenceSurface =
    options.presenceSurface ??
    pick(surfacesForPresence(intent.required), rng, intent.required ? "required" : "optional");

  const nullable = options.nullable ?? intent.nullable === true;

  const nullableSurface = nullable
    ? (options.nullableSurface ?? pick(surfacesForNullable(), rng, "nullable"))
    : undefined;

  const spoken = options.spoken ?? spokenFromIntent(intent);

  const numberStyle =
    options.numberStyle ?? (rng !== undefined && rng.bool(0.35) ? "words" : "digits");

  const glueSurface = options.glueSurface ?? pick(GLUE_SURFACES, rng, "with");

  const unitSurface =
    spoken?.unit === true
      ? (options.unitSurface ?? pick(surfacesForUnit(), rng, "characters"))
      : undefined;

  const minSurface =
    options.minSurface ??
    pick(
      surfacesForComparison({
        side: "min",
        inclusive: spoken?.min?.inclusive ?? true,
      }),
      rng,
      spoken?.min?.inclusive === false ? "more than" : "at least",
    );

  const maxSurface =
    options.maxSurface ??
    pick(
      surfacesForComparison({
        side: "max",
        inclusive: spoken?.max?.inclusive ?? true,
      }),
      rng,
      spoken?.max?.inclusive === false ? "less than" : "max",
    );

  const betweenSurface =
    options.betweenSurface ?? pick(surfacesForComparison({ side: "between" }), rng, "between");

  const exactSurface =
    options.exactSurface ?? pick(surfacesForComparison({ side: "exact" }), rng, "exactly");

  const order = options.order ?? defaultOrder(spoken);

  const sentence = new Sentence();

  const comparison = {
    minSurface,
    maxSurface,
    betweenSurface,
    exactSurface,
    unitSurface,
    numberStyle,
  };

  const head = (): void => {
    if (nullableSurface !== undefined) addNullable(sentence, nullableSurface);
  };

  switch (order) {
    case "presence-type":
      addPresence(sentence, intent.required, presenceSurface);
      addType(sentence, intent.type, typeSurface);
      head();
      break;
    case "type-presence":
      addType(sentence, intent.type, typeSurface);
      addPresence(sentence, intent.required, presenceSurface);
      head();
      break;
    case "presence-glue-type":
      addPresence(sentence, intent.required, presenceSurface);
      sentence.add(glueSurface, "GLUE");
      addType(sentence, intent.type, typeSurface);
      head();
      break;
    case "type-glue-presence":
      addType(sentence, intent.type, typeSurface);
      sentence.add(glueSurface, "GLUE");
      addPresence(sentence, intent.required, presenceSurface);
      head();
      break;
    case "presence-type-glue-comparison":
      addPresence(sentence, intent.required, presenceSurface);
      addType(sentence, intent.type, typeSurface);
      head();

      if (spoken !== undefined) {
        sentence.add(glueSurface, "GLUE");
        addComparison(sentence, spoken, comparison);
      }

      break;
    case "presence-type-comparison":
      addPresence(sentence, intent.required, presenceSurface);
      addType(sentence, intent.type, typeSurface);
      head();
      if (spoken !== undefined) addComparison(sentence, spoken, comparison);
      break;
    case "type-presence-comparison":
      addType(sentence, intent.type, typeSurface);
      addPresence(sentence, intent.required, presenceSurface);
      head();
      if (spoken !== undefined) addComparison(sentence, spoken, comparison);
      break;
    case "type-copula-presence-comparison":
      addType(sentence, intent.type, typeSurface);
      sentence.add("is", "GLUE");
      addPresence(sentence, intent.required, presenceSurface);
      head();

      if (spoken !== undefined) {
        sentence.add(glueSurface, "GLUE");
        addComparison(sentence, spoken, comparison);
      }

      break;
    case "comparison-presence-type":
      if (spoken !== undefined) addComparison(sentence, spoken, comparison);
      addPresence(sentence, intent.required, presenceSurface);
      addType(sentence, intent.type, typeSurface);
      head();
      break;
    case "comparison-type-presence":
      if (spoken !== undefined) addComparison(sentence, spoken, comparison);
      addType(sentence, intent.type, typeSurface);
      addPresence(sentence, intent.required, presenceSurface);
      head();
      break;
    case "type-comparison-presence":
      addType(sentence, intent.type, typeSurface);
      if (spoken !== undefined) addComparison(sentence, spoken, comparison);
      addPresence(sentence, intent.required, presenceSurface);
      head();
      break;

    default: {
      const exhaustive: never = order;

      throw new RangeError(`unknown render order: ${String(exhaustive)}`);
    }
  }

  return {
    text: sentence.text,
    spans: sentence.spans,
    template: options.template ?? `${intent.type}/${order}`,
    style: options.style ?? order,
  };
}

/** Render a negative example as a single `O` span. No gold intent. */
export function renderNegative(text: string, template: string): RenderedExample {
  const sentence = new Sentence();

  sentence.add(text, "O");

  return {
    text: sentence.text,
    spans: sentence.spans,
    template,
    style: "negative",
  };
}

export function bound(value: number, inclusive: boolean): Bound {
  return { value, inclusive };
}
