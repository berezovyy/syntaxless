/**
 * Sentence-kind catalog for the backward corpus generator.
 *
 * Coverage is kinds, not paraphrases of one frame. A kind is one way a field
 * constraint is said (required email + max length, integer min exclusive, …)
 * together with the intents that kind can name. Templates are alternate
 * wordings of that kind; held-out templates use a different frame so train
 * and held-out fingerprints cannot collide.
 *
 * Surfaces come only from the runtime lexicon. No LLM is consulted here.
 */
import type { FieldIntent, FieldType } from "../../src/types/intent.js";
import type { Family } from "./generate.js";
import {
  type NumberStyle,
  renderField,
  renderNegative,
  type RenderOptions,
  type RenderOrder,
  type SpokenComparison,
  surfacesForComparison,
  surfacesForPresence,
  surfacesForType,
  surfacesForUnit,
} from "./renderer.js";
import type { RenderedExample } from "./renderer.js";

export type KindId =
  | "email-required-maxlength"
  | "email-optional-maxlength"
  | "optional-url"
  | "required-url"
  | "url-required-minlength"
  | "required-uuid"
  | "optional-uuid"
  | "uuid-exact-length"
  | "required-string"
  | "optional-string"
  | "string-min-length"
  | "string-max-length"
  | "string-min-length-exclusive"
  | "string-max-length-exclusive"
  | "string-between-length"
  | "string-exact-length"
  | "boolean-required"
  | "boolean-optional"
  | "required-number"
  | "optional-number"
  | "number-min-inclusive"
  | "number-min-exclusive"
  | "number-max-inclusive"
  | "number-max-exclusive"
  | "number-between"
  | "number-exact"
  | "integer-min-inclusive"
  | "integer-min-exclusive"
  | "integer-max-inclusive"
  | "integer-max-exclusive"
  | "integer-between"
  | "integer-exact"
  | "required-nullable-email"
  | "negative-descriptive"
  | "negative-question"
  | "negative-quoted"
  | "negative-approximate"
  | "negative-wrong-domain"
  | "negative-incomplete";

export type KindSplit = "train" | "heldout";
export type ExampleSplit = "train" | "validation" | "heldout";

export interface KindExample {
  kind: KindId;
  family: Family;
  template: string;
  split: ExampleSplit;
  intent: FieldIntent | null;
  example: RenderedExample;
  splitExample?: RenderedExample;
}

interface LengthValue {
  min?: number;
  max?: number;
  exclusiveMin?: number;
  exclusiveMax?: number;
  exact?: number;
}

interface NumberValue {
  min?: { value: number; inclusive: boolean };
  max?: { value: number; inclusive: boolean };
  exact?: number;
  between?: { min: number; max: number };
}

interface TemplateSpec {
  id: string;
  order: RenderOrder;
  typeSurface?: string;
  presenceSurface?: string;
  minSurface?: string;
  maxSurface?: string;
  betweenSurface?: string;
  exactSurface?: string;
  unitSurface?: string;
  glueSurface?: string;
  numberStyle: NumberStyle;
  nullable?: boolean;
}

const LENGTH_ORDERS: readonly RenderOrder[] = [
  "presence-type-glue-comparison",
  "type-presence-comparison",
  "comparison-presence-type",
  "presence-type-comparison",
  "type-comparison-presence",
];

const BARE_ORDERS: readonly RenderOrder[] = [
  "presence-type",
  "type-presence",
  "presence-glue-type",
  "type-glue-presence",
];

function typeSurfaces(type: FieldType, take: number): string[] {
  return surfacesForType(type).slice(0, take);
}

function unitCycle(): string[] {
  const units = surfacesForUnit();

  return units.length >= 3 ? units.slice(0, 4) : ["characters", "chars", "character"];
}

function presenceCycle(required: boolean): string[] {
  const rows = surfacesForPresence(required);

  return rows.length >= 2 ? rows : required ? ["required"] : ["optional", "not required"];
}

function minSurfaces(inclusive: boolean): string[] {
  return surfacesForComparison({ side: "min", inclusive }).slice(0, 4);
}

function maxSurfaces(inclusive: boolean): string[] {
  return surfacesForComparison({ side: "max", inclusive }).slice(0, 4);
}

function betweenSurfaces(): string[] {
  return surfacesForComparison({ side: "between" });
}

function exactSurfaces(): string[] {
  return surfacesForComparison({ side: "exact" });
}

function lengthIntent(type: FieldType, required: boolean, value: LengthValue): FieldIntent {
  const intent: FieldIntent = { type, required };

  if (value.exact !== undefined) {
    intent.minLength = value.exact;
    intent.maxLength = value.exact;

    return intent;
  }

  if (value.min !== undefined) intent.minLength = value.min;
  if (value.max !== undefined) intent.maxLength = value.max;
  if (value.exclusiveMin !== undefined) intent.minLength = value.exclusiveMin + 1;
  if (value.exclusiveMax !== undefined) intent.maxLength = value.exclusiveMax - 1;

  return intent;
}

function spokenLength(value: LengthValue): SpokenComparison {
  if (value.exact !== undefined) return { exact: value.exact, unit: true };

  if (value.min !== undefined && value.max !== undefined) {
    return {
      min: { value: value.min, inclusive: true },
      max: { value: value.max, inclusive: true },
      unit: true,
    };
  }

  const spoken: SpokenComparison = { unit: true };

  if (value.exclusiveMin !== undefined) {
    spoken.min = { value: value.exclusiveMin, inclusive: false };
  } else if (value.min !== undefined) {
    spoken.min = { value: value.min, inclusive: true };
  }

  if (value.exclusiveMax !== undefined) {
    spoken.max = { value: value.exclusiveMax, inclusive: false };
  } else if (value.max !== undefined) {
    spoken.max = { value: value.max, inclusive: true };
  }

  return spoken;
}

function numberIntent(type: FieldType, required: boolean, value: NumberValue): FieldIntent {
  const intent: FieldIntent = { type, required };

  if (value.exact !== undefined) {
    intent.minimum = { value: value.exact, inclusive: true };
    intent.maximum = { value: value.exact, inclusive: true };

    return intent;
  }

  if (value.between !== undefined) {
    intent.minimum = { value: value.between.min, inclusive: true };
    intent.maximum = { value: value.between.max, inclusive: true };

    return intent;
  }

  if (value.min !== undefined) intent.minimum = value.min;
  if (value.max !== undefined) intent.maximum = value.max;

  return intent;
}

function spokenNumber(value: NumberValue): SpokenComparison {
  if (value.exact !== undefined) return { exact: value.exact, unit: false };
  if (value.between !== undefined) return { between: value.between, unit: false };
  const spoken: SpokenComparison = { unit: false };

  if (value.min !== undefined) spoken.min = value.min;
  if (value.max !== undefined) spoken.max = value.max;

  return spoken;
}

function lengthTemplates(
  type: FieldType,
  required: boolean,
  kind: "min" | "max" | "between" | "exact",
  inclusive: boolean,
): TemplateSpec[] {
  const types = typeSurfaces(type, 3);
  const presence = presenceCycle(required);
  const units = unitCycle();
  const styles: NumberStyle[] = ["digits", "words"];
  const templates: TemplateSpec[] = [];
  const orders = LENGTH_ORDERS;

  for (let index = 0; index < Math.max(orders.length, types.length, units.length); index++) {
    const spec: TemplateSpec = {
      id: `t${index}`,
      order: orders[index % orders.length]!,
      typeSurface: types[index % types.length]!,
      presenceSurface: presence[index % presence.length]!,
      unitSurface: units[index % units.length]!,
      glueSurface: "with",
      numberStyle: styles[index % styles.length]!,
    };

    if (kind === "min") {
      spec.minSurface = minSurfaces(inclusive)[index % minSurfaces(inclusive).length]!;
    }

    if (kind === "max") {
      spec.maxSurface = maxSurfaces(inclusive)[index % maxSurfaces(inclusive).length]!;
    }

    if (kind === "between") spec.betweenSurface = betweenSurfaces()[0]!;
    if (kind === "exact") spec.exactSurface = exactSurfaces()[index % exactSurfaces().length]!;
    templates.push(spec);
  }

  return templates;
}

function numberTemplates(
  type: FieldType,
  required: boolean,
  kind: "min" | "max" | "between" | "exact",
  inclusive: boolean,
): TemplateSpec[] {
  const types = typeSurfaces(type, 2);
  const presence = presenceCycle(required);
  const styles: NumberStyle[] = ["digits", "words"];
  const templates: TemplateSpec[] = [];

  for (let index = 0; index < LENGTH_ORDERS.length; index++) {
    const spec: TemplateSpec = {
      id: `t${index}`,
      order: LENGTH_ORDERS[index]!,
      typeSurface: types[index % types.length]!,
      presenceSurface: presence[index % presence.length]!,
      glueSurface: "with",
      numberStyle: styles[index % styles.length]!,
    };

    if (kind === "min") {
      spec.minSurface = minSurfaces(inclusive)[index % minSurfaces(inclusive).length]!;
    }

    if (kind === "max") {
      spec.maxSurface = maxSurfaces(inclusive)[index % maxSurfaces(inclusive).length]!;
    }

    if (kind === "between") spec.betweenSurface = betweenSurfaces()[0]!;
    if (kind === "exact") spec.exactSurface = exactSurfaces()[index % exactSurfaces().length]!;
    templates.push(spec);
  }

  return templates;
}

function bareTemplates(type: FieldType, required: boolean, nullable = false): TemplateSpec[] {
  const types = typeSurfaces(type, 3);
  const presence = presenceCycle(required);
  const templates: TemplateSpec[] = [];
  let index = 0;

  for (const order of BARE_ORDERS) {
    for (const typeSurface of types) {
      for (const presenceSurface of presence) {
        templates.push({
          id: `t${index++}`,
          order,
          typeSurface,
          presenceSurface,
          glueSurface: "with",
          numberStyle: "digits",
          nullable,
        });
      }
    }
  }

  return templates;
}

function optionsFrom(
  spec: TemplateSpec,
  spoken: SpokenComparison | undefined,
  kind: KindId,
): RenderOptions {
  const options: RenderOptions = {
    template: `${kind}/${spec.id}`,
    style: spec.order,
    order: spec.order,
    numberStyle: spec.numberStyle,
  };

  if (spec.typeSurface !== undefined) options.typeSurface = spec.typeSurface;
  if (spec.presenceSurface !== undefined) options.presenceSurface = spec.presenceSurface;
  if (spec.minSurface !== undefined) options.minSurface = spec.minSurface;
  if (spec.maxSurface !== undefined) options.maxSurface = spec.maxSurface;
  if (spec.betweenSurface !== undefined) options.betweenSurface = spec.betweenSurface;
  if (spec.exactSurface !== undefined) options.exactSurface = spec.exactSurface;
  if (spec.unitSurface !== undefined) options.unitSurface = spec.unitSurface;
  if (spec.glueSurface !== undefined) options.glueSurface = spec.glueSurface;
  if (spoken !== undefined) options.spoken = spoken;
  if (spec.nullable !== undefined) options.nullable = spec.nullable;

  return options;
}

function* expandLength(
  kind: KindId,
  type: FieldType,
  required: boolean,
  values: readonly LengthValue[],
  templates: readonly TemplateSpec[],
): Generator<KindExample> {
  for (const value of values) {
    const intent = lengthIntent(type, required, value);
    const spoken = spokenLength(value);

    for (const spec of templates) {
      const example = renderField(intent, undefined, optionsFrom(spec, spoken, kind));

      yield {
        kind,
        family: "constraint",
        template: example.template,
        split: "train",
        intent,
        example,
      };
    }
  }
}

function* expandNumber(
  kind: KindId,
  type: FieldType,
  required: boolean,
  values: readonly NumberValue[],
  templates: readonly TemplateSpec[],
): Generator<KindExample> {
  for (const value of values) {
    const intent = numberIntent(type, required, value);
    const spoken = spokenNumber(value);

    for (const spec of templates) {
      const example = renderField(intent, undefined, optionsFrom(spec, spoken, kind));

      yield {
        kind,
        family: "constraint",
        template: example.template,
        split: "train",
        intent,
        example,
      };
    }
  }
}

function* expandBare(
  kind: KindId,
  type: FieldType,
  required: boolean,
  templates: readonly TemplateSpec[],
  nullable = false,
): Generator<KindExample> {
  const intent: FieldIntent = { type, required };

  if (nullable) intent.nullable = true;

  for (const spec of templates) {
    const example = renderField(intent, undefined, optionsFrom(spec, undefined, kind));

    yield {
      kind,
      family: "constraint",
      template: example.template,
      split: "train",
      intent,
      example,
    };
  }
}

const LENGTHS_MAX: readonly LengthValue[] = [{ max: 8 }, { max: 20 }, { max: 50 }, { max: 100 }];
const LENGTHS_MIN: readonly LengthValue[] = [{ min: 1 }, { min: 5 }, { min: 10 }, { min: 24 }];

const LENGTHS_EX_MIN: readonly LengthValue[] = [
  { exclusiveMin: 3 },
  { exclusiveMin: 10 },
  { exclusiveMin: 20 },
];

const LENGTHS_EX_MAX: readonly LengthValue[] = [
  { exclusiveMax: 12 },
  { exclusiveMax: 20 },
  { exclusiveMax: 40 },
];

const LENGTHS_BETWEEN: readonly LengthValue[] = [
  { min: 5, max: 20 },
  { min: 1, max: 10 },
  { min: 8, max: 64 },
];

const LENGTHS_EXACT: readonly LengthValue[] = [{ exact: 8 }, { exact: 16 }, { exact: 36 }];

const NUM_MIN_INC: readonly NumberValue[] = [
  { min: { value: 0, inclusive: true } },
  { min: { value: 10, inclusive: true } },
  { min: { value: 18, inclusive: true } },
];

const NUM_MIN_EXC: readonly NumberValue[] = [
  { min: { value: 0, inclusive: false } },
  { min: { value: 10, inclusive: false } },
  { min: { value: 18, inclusive: false } },
];

const NUM_MAX_INC: readonly NumberValue[] = [
  { max: { value: 10, inclusive: true } },
  { max: { value: 100, inclusive: true } },
  { max: { value: 64, inclusive: true } },
];

const NUM_MAX_EXC: readonly NumberValue[] = [
  { max: { value: 10, inclusive: false } },
  { max: { value: 100, inclusive: false } },
  { max: { value: 1, inclusive: false } },
];

const NUM_BETWEEN: readonly NumberValue[] = [
  { between: { min: 5, max: 20 } },
  { between: { min: 1, max: 10 } },
  { between: { min: 0, max: 100 } },
];

const NUM_EXACT: readonly NumberValue[] = [{ exact: 0 }, { exact: 10 }, { exact: 42 }];

/**
 * Negative families. Wording is original to this catalog and each line must
 * fail the rule parser.
 */
const NEGATIVES: Record<
  Extract<
    KindId,
    | "negative-descriptive"
    | "negative-question"
    | "negative-quoted"
    | "negative-approximate"
    | "negative-wrong-domain"
    | "negative-incomplete"
  >,
  readonly string[]
> = {
  "negative-descriptive": [
    "Emails are usually required.",
    "Prices are usually at least 10.",
    "User names are generally optional.",
    "Most profiles need an email address.",
    "A typical password is fairly long.",
    "Age fields tend to be integers.",
    "Optional urls show up in bios.",
    "Boolean flags are often required in forms.",
  ],
  "negative-question": [
    "What's the max length for email?",
    "Is the integer field required?",
    "How many characters may a string have?",
    "Should url be optional?",
    "Do we want a uuid here?",
    "What is the minimum for age?",
    "Can boolean be nullable?",
    "Which type should this field use?",
  ],
  "negative-quoted": [
    'The docs say "required email max 100".',
    'Copy the phrase "optional url" into the ticket.',
    'The sample reads "required integer at least 18".',
    'A tooltip shows "between 5 and 20 characters".',
    'She quoted "required boolean" in the review.',
    'Search for "max 100 characters" in the guide.',
    'The heading is labeled "required uuid".',
    'Translate "optional string" for the locale file.',
  ],
  "negative-approximate": [
    "around 10 characters",
    "about 18 years",
    "roughly 100 long",
    "required string around 10 characters",
    "integer more or less 10",
    "email maybe max 100 characters",
    "something like 5 to 20 characters",
    "required number around ten",
  ],
  "negative-wrong-domain": [
    "Send a reminder every weekday at 9.",
    "Run this job every hour.",
    "Remind me each Monday morning.",
    "Schedule the backup at noon.",
    "Ping the channel every 15 minutes.",
    "Fire the alert on weekdays.",
    "Repeat the digest every month.",
    "Wake the worker at midnight.",
  ],
  "negative-incomplete": [
    "between 5",
    "required integer between 5",
    "at least",
    "required email max",
    "optional string more than",
    "required number from",
    "max characters required email",
    "required string between and 20 characters",
  ],
};

function* expandNegatives(): Generator<KindExample> {
  for (const [kind, texts] of Object.entries(NEGATIVES) as Array<
    [keyof typeof NEGATIVES, readonly string[]]
  >) {
    for (let index = 0; index < texts.length; index++) {
      const template = `${kind}/${index}`;
      const example = renderNegative(texts[index]!, template);

      yield {
        kind,
        family: "negative",
        template,
        split: "train",
        intent: null,
        example,
      };
    }
  }
}

export function* iterateKinds(wanted?: ReadonlySet<KindId>): Generator<KindExample> {
  const include = (id: KindId): boolean => wanted === undefined || wanted.has(id);

  if (include("email-required-maxlength")) {
    yield* expandLength(
      "email-required-maxlength",
      "email",
      true,
      LENGTHS_MAX,
      lengthTemplates("email", true, "max", true),
    );
  }

  if (include("email-optional-maxlength")) {
    yield* expandLength(
      "email-optional-maxlength",
      "email",
      false,
      LENGTHS_MAX,
      lengthTemplates("email", false, "max", true),
    );
  }

  if (include("optional-url")) {
    yield* expandBare("optional-url", "url", false, bareTemplates("url", false));
  }

  if (include("required-url")) {
    yield* expandBare("required-url", "url", true, bareTemplates("url", true));
  }

  if (include("url-required-minlength")) {
    yield* expandLength(
      "url-required-minlength",
      "url",
      true,
      LENGTHS_MIN,
      lengthTemplates("url", true, "min", true),
    );
  }

  if (include("required-uuid")) {
    yield* expandBare("required-uuid", "uuid", true, bareTemplates("uuid", true));
  }

  if (include("optional-uuid")) {
    yield* expandBare("optional-uuid", "uuid", false, bareTemplates("uuid", false));
  }

  if (include("uuid-exact-length")) {
    yield* expandLength(
      "uuid-exact-length",
      "uuid",
      true,
      LENGTHS_EXACT,
      lengthTemplates("uuid", true, "exact", true),
    );
  }

  if (include("required-string")) {
    yield* expandBare("required-string", "string", true, bareTemplates("string", true));
  }

  if (include("optional-string")) {
    yield* expandBare("optional-string", "string", false, bareTemplates("string", false));
  }

  if (include("string-min-length")) {
    yield* expandLength(
      "string-min-length",
      "string",
      true,
      LENGTHS_MIN,
      lengthTemplates("string", true, "min", true),
    );
  }

  if (include("string-max-length")) {
    yield* expandLength(
      "string-max-length",
      "string",
      true,
      LENGTHS_MAX,
      lengthTemplates("string", true, "max", true),
    );
  }

  if (include("string-min-length-exclusive")) {
    yield* expandLength(
      "string-min-length-exclusive",
      "string",
      true,
      LENGTHS_EX_MIN,
      lengthTemplates("string", true, "min", false),
    );
  }

  if (include("string-max-length-exclusive")) {
    yield* expandLength(
      "string-max-length-exclusive",
      "string",
      true,
      LENGTHS_EX_MAX,
      lengthTemplates("string", true, "max", false),
    );
  }

  if (include("string-between-length")) {
    yield* expandLength(
      "string-between-length",
      "string",
      true,
      LENGTHS_BETWEEN,
      lengthTemplates("string", true, "between", true),
    );
  }

  if (include("string-exact-length")) {
    yield* expandLength(
      "string-exact-length",
      "string",
      true,
      LENGTHS_EXACT,
      lengthTemplates("string", true, "exact", true),
    );
  }

  if (include("boolean-required")) {
    yield* expandBare("boolean-required", "boolean", true, bareTemplates("boolean", true));
  }

  if (include("boolean-optional")) {
    yield* expandBare("boolean-optional", "boolean", false, bareTemplates("boolean", false));
  }

  if (include("required-number")) {
    yield* expandBare("required-number", "number", true, bareTemplates("number", true));
  }

  if (include("optional-number")) {
    yield* expandBare("optional-number", "number", false, bareTemplates("number", false));
  }

  if (include("number-min-inclusive")) {
    yield* expandNumber(
      "number-min-inclusive",
      "number",
      true,
      NUM_MIN_INC,
      numberTemplates("number", true, "min", true),
    );
  }

  if (include("number-min-exclusive")) {
    yield* expandNumber(
      "number-min-exclusive",
      "number",
      true,
      NUM_MIN_EXC,
      numberTemplates("number", true, "min", false),
    );
  }

  if (include("number-max-inclusive")) {
    yield* expandNumber(
      "number-max-inclusive",
      "number",
      true,
      NUM_MAX_INC,
      numberTemplates("number", true, "max", true),
    );
  }

  if (include("number-max-exclusive")) {
    yield* expandNumber(
      "number-max-exclusive",
      "number",
      true,
      NUM_MAX_EXC,
      numberTemplates("number", true, "max", false),
    );
  }

  if (include("number-between")) {
    yield* expandNumber(
      "number-between",
      "number",
      true,
      NUM_BETWEEN,
      numberTemplates("number", true, "between", true),
    );
  }

  if (include("number-exact")) {
    yield* expandNumber(
      "number-exact",
      "number",
      true,
      NUM_EXACT,
      numberTemplates("number", true, "exact", true),
    );
  }

  if (include("integer-min-inclusive")) {
    yield* expandNumber(
      "integer-min-inclusive",
      "integer",
      true,
      NUM_MIN_INC,
      numberTemplates("integer", true, "min", true),
    );
  }

  if (include("integer-min-exclusive")) {
    yield* expandNumber(
      "integer-min-exclusive",
      "integer",
      true,
      NUM_MIN_EXC,
      numberTemplates("integer", true, "min", false),
    );
  }

  if (include("integer-max-inclusive")) {
    yield* expandNumber(
      "integer-max-inclusive",
      "integer",
      true,
      NUM_MAX_INC,
      numberTemplates("integer", true, "max", true),
    );
  }

  if (include("integer-max-exclusive")) {
    yield* expandNumber(
      "integer-max-exclusive",
      "integer",
      true,
      NUM_MAX_EXC,
      numberTemplates("integer", true, "max", false),
    );
  }

  if (include("integer-between")) {
    yield* expandNumber(
      "integer-between",
      "integer",
      true,
      NUM_BETWEEN,
      numberTemplates("integer", true, "between", true),
    );
  }

  if (include("integer-exact")) {
    yield* expandNumber(
      "integer-exact",
      "integer",
      true,
      NUM_EXACT,
      numberTemplates("integer", true, "exact", true),
    );
  }

  if (include("required-nullable-email")) {
    yield* expandBare(
      "required-nullable-email",
      "email",
      true,
      bareTemplates("email", true, true),
      true,
    );
  }

  const negativeWanted =
    wanted === undefined || [...wanted].some((id) => id.startsWith("negative-"));

  if (negativeWanted) {
    for (const example of expandNegatives()) {
      if (include(example.kind)) yield example;
    }
  }
}

export function expandKinds(wanted?: ReadonlySet<KindId>): KindExample[] {
  return [...iterateKinds(wanted)];
}

export function kindIds(): KindId[] {
  return [
    "email-required-maxlength",
    "email-optional-maxlength",
    "optional-url",
    "required-url",
    "url-required-minlength",
    "required-uuid",
    "optional-uuid",
    "uuid-exact-length",
    "required-string",
    "optional-string",
    "string-min-length",
    "string-max-length",
    "string-min-length-exclusive",
    "string-max-length-exclusive",
    "string-between-length",
    "string-exact-length",
    "boolean-required",
    "boolean-optional",
    "required-number",
    "optional-number",
    "number-min-inclusive",
    "number-min-exclusive",
    "number-max-inclusive",
    "number-max-exclusive",
    "number-between",
    "number-exact",
    "integer-min-inclusive",
    "integer-min-exclusive",
    "integer-max-inclusive",
    "integer-max-exclusive",
    "integer-between",
    "integer-exact",
    "required-nullable-email",
    "negative-descriptive",
    "negative-question",
    "negative-quoted",
    "negative-approximate",
    "negative-wrong-domain",
    "negative-incomplete",
  ];
}
