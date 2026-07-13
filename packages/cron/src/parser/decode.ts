import { compile } from "../compiler/cron.js";
import {
  validateDayOfMonth,
  validateInterval,
  validateOrdinalWeekday,
  validateTime,
  validateWeekdays,
} from "../compiler/validate.js";
/**
 * Deterministic decoder: predicted token roles -> validated {@link ScheduleIntent}.
 *
 * This is the exact inverse of the Phase 07 renderer. The model predicts roles;
 * every number word, clock conversion, weekday number, and interval multiplier
 * is resolved here in TypeScript. The decoder never guesses a role: an
 * out-of-range label id is a hard {@link UnknownRoleError}.
 *
 * Data only: importing this module pulls in the label vocabulary, the
 * tokenizer's `RawToken` type, the parser lexicon, and the Phase 03 bounds
 * checks. It never imports the inference engine or the training renderer.
 */
import { isStructural, LABEL_COUNT, labelFromId } from "../model/labels.js";
import type { Label } from "../model/labels.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import type { RawToken } from "../model/tokenizer.js";
import type {
  DaySpec,
  FrequencyUnit,
  Ordinal,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../types/intent.js";
import {
  ADVERB_FREQUENCIES,
  BOUND_ALIASES,
  DAY_ORDINALS,
  DAY_RULES,
  EXCEPTION_PREFIXES,
  FREQUENCY_IDIOMS,
  MINUTE_IDIOMS,
  MONTH_ALIASES,
  NAMED_HOURS,
  NUMBER_WORDS,
  ORDINAL_ALIASES,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WEEKDAY_DESCRIPTIONS,
  WEEKDAY_GROUPS,
  WEEKDAY_PLURAL_ALIASES,
  WEEKDAY_SEQUENCE,
  WORDS,
} from "./lexicon.js";

/** One token with its predicted role, clause boundary, and margin score. */
export interface LabeledToken {
  token: RawToken;
  role: number;
  boundary: boolean;
  score: number;
  /** Optional for oracle callers; runtime inference always supplies it. */
  boundaryScore?: number;
}

/** A validated intent plus the weakest margin among its content tokens. */
export interface DecodedIntent {
  intent: ScheduleIntent;
  confidence: number;
  /** Inspectable sequence-level inputs used to derive `confidence`. */
  evidence: SequenceConfidenceEvidence;
}

export interface SequenceConfidenceEvidence {
  tokenMargin: number;
  boundaryMargin: number;
  scheduleKind?: ScheduleKind;
  scheduleKindMargin?: number;
  scheduleKindAgreement: boolean;
  explicitScheduleEvidence: boolean;
  /** Present only when deterministic language evidence was reconciled. */
  lexiconAgreement?: boolean;
  unconsumedScheduleLikeValues: number;
  semanticConstraintViolations: number;
  compilerConsistent: boolean;
}

export interface ScheduleKindEvidence {
  kind: ScheduleKind;
  score: number;
}

/** An out-of-range predicted role id. Never mapped to `O`. */
export class UnknownRoleError extends Error {
  readonly role: number;
  constructor(role: number) {
    super(`unknown role id: ${role}`);
    this.name = "UnknownRoleError";
    this.role = role;
  }
}

export type UnsafeScheduleContextReason =
  | "informational-question"
  | "uncertain-alternative"
  | "competing-correction"
  | "quoted-schedule"
  | "historical-description"
  | "multiple-schedules"
  | "descriptive-statement"
  | "non-scheduling-language";

/** A syntactically schedule-like phrase used in a non-command discourse role. */
export class UnsafeScheduleContextError extends RangeError {
  readonly contextReason: UnsafeScheduleContextReason;
  constructor(contextReason: UnsafeScheduleContextReason) {
    super(`decodeIntent: unsafe schedule context: ${contextReason}`);
    this.name = "UnsafeScheduleContextError";
    this.contextReason = contextReason;
  }
}

interface Entry {
  token: RawToken;
  label: Label;
  boundary: boolean;
  score: number;
  boundaryScore: number;
}

interface Group {
  label: Label;
  tokens: RawToken[];
}

interface ClauseFacts {
  interval?: number;
  unit?: FrequencyUnit;
  other: boolean;
  weekdays: Weekday[];
  ordinal?: Ordinal;
  dayOfMonth?: number;
  bounds: RecurrenceBounds;
  hour?: number;
  minute?: number;
  meridiem?: "am" | "pm";
  minuteRelation?: "past" | "to";
  day?: DaySpec;
  exceptions: ScheduleExceptions;
}

function exactEvidence(): SequenceConfidenceEvidence {
  return {
    tokenMargin: 1,
    boundaryMargin: 1,
    scheduleKindAgreement: true,
    explicitScheduleEvidence: true,
    lexiconAgreement: true,
    unconsumedScheduleLikeValues: 0,
    semanticConstraintViolations: 0,
    compilerConsistent: true,
  };
}

function countUnconsumedScheduleLikeValues(text: string, intent: ScheduleIntent): number {
  const representedWeekdays = new Set<Weekday>([
    ...(intent.weekdays ?? []),
    ...(intent.ordinalWeekday === undefined ? [] : [intent.ordinalWeekday.weekday]),
    ...(intent.exceptions?.weekdays ?? []),
  ]);

  let unconsumed = 0;

  const weekdaySurfaces = new Map<string, Weekday>([
    ...Object.entries(WEEKDAY_ALIASES),
    ...Object.entries(WEEKDAY_PLURAL_ALIASES),
  ] as Array<[string, Weekday]>);

  for (const match of text.matchAll(/\b[\p{L}_]+\b/gu)) {
    const weekday = weekdaySurfaces.get(match[0]);

    if (weekday !== undefined && !representedWeekdays.has(weekday)) unconsumed++;
  }

  const representedDates = new Set<string>([
    ...(intent.bounds?.start === undefined ? [] : [intent.bounds.start]),
    ...(intent.bounds?.end === undefined ? [] : [intent.bounds.end]),
    ...(intent.exceptions?.dates ?? []),
  ]);

  for (const match of text.matchAll(/\b\d{4}-\d{2}-\d{2}\b/g)) {
    if (!representedDates.has(match[0])) unconsumed++;
  }

  const clockMentions = [
    ...text.matchAll(
      /\b(?:at|around)\s+(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight|half\s+past\s+[a-z]+|quarter\s+(?:past|to)\s+[a-z]+)\b/g,
    ),
  ].length;

  if (clockMentions > 1) unconsumed += clockMentions - 1;

  return unconsumed;
}

function compilerConsistent(intent: ScheduleIntent): boolean {
  const result = compile(intent);

  if (result.supported) return true;

  return ["non-unit-interval", "ordinal-weekday", "day-spec", "bounds", "exceptions"].includes(
    result.code,
  );
}

function unsafeScheduleContext(text: string): UnsafeScheduleContextReason | null {
  // The ordinary parser normalization deliberately removes sentence-final
  // punctuation. Keep a separate, lightweight normalization here because a
  // question mark is strong discourse evidence and must not disappear before
  // safety classification.
  text = text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\s+/g, " ")
    .trim();

  const scheduleCue =
    /\b(?:at\s+(?:\d{1,2}|noon|midnight)|\d{1,2}\s*(?:am|pm)|half\s+past|quarter\s+(?:past|to)|daily|each|every|everyday|hourly|minutes?|hours?|days?|weeks?|months?|monthly|morning|nightly|once\s+(?:a|per)|weekly|weekdays?|weekends?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b/;

  const requestQuestionPrefix =
    /\b(?:can|could|would|will)\s+you\b/.test(text) ||
    /\b(?:can|could|would)\s+(?:this|it)\s+(?:be\s+)?(?:arranged|happen|recur|repeat|run|scheduled)\b/.test(
      text,
    );

  const metaQuestion =
    /\b(?:can|could|would|will)\s+you\b.*\b(?:describe|explain|know|remember|say|tell)\b/.test(
      text,
    ) || /\b(?:can|could|would|will)\s+you\b.*\b(?:check|confirm)\s+(?:if|whether)\b/.test(text);

  const politeRequestQuestion =
    /\b(?:can|could|may)\s+i\s+(?:ask|book|get|have|request|set)\b/.test(text) ||
    /\b(?:could|would)\s+(?:it be possible to|you mind)\b/.test(text) ||
    /\b(?:do you think you could|is it possible (?:for you )?to|would it be (?:okay|ok) (?:for you )?to)\b/.test(
      text,
    );

  const requestQuestion =
    (requestQuestionPrefix || politeRequestQuestion) && scheduleCue.test(text) && !metaQuestion;

  if (/\?$/.test(text) && !requestQuestion) return "informational-question";

  const weekdayMentions = [
    ...text.matchAll(
      /\b(?:mon(?:day)?s?|tue(?:s|sday)?s?|wed(?:nesday)?s?|thu(?:r|rs|rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)\b/g,
    ),
  ].length;

  const cadenceMentions = [...text.matchAll(/\b(?:daily|hourly|monthly|weekly)\b/g)].length;

  const clockMentions = [
    ...text.matchAll(/\b(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)\b/g),
  ].length;

  const hasCompetingScheduleValues =
    weekdayMentions >= 2 ||
    cadenceMentions >= 2 ||
    clockMentions >= 2 ||
    [...text.matchAll(/\b\d{1,2}(?:st|nd|rd|th)\b/g)].length >= 2;

  const uncertainAlternative =
    !/\b(?:except|excluding|other than)\b/.test(text) &&
    ((/\b(?:maybe|might|perhaps|possibly|probably|unsure|not sure|undecided)\b/.test(text) &&
      scheduleCue.test(text)) ||
      /\beither\b.*\bor\b/.test(text) ||
      /\b(?:whether|between)\b.*\bor\b/.test(text) ||
      (/\bor\b/.test(text) && hasCompetingScheduleValues));

  if (uncertainAlternative) return "uncertain-alternative";

  const competingCorrection =
    /\b(?:not|instead of)\b.*\b(?:but|rather|instead)\b/.test(text) ||
    /\b(?:change|move|switch)\b.*\bfrom\b.*\bto\b/.test(text) ||
    (scheduleCue.test(text) &&
      (/^(?:no|wait)[,:]?\s+/.test(text) ||
        /\b(?:cancel|correcting|correction|disregard|forget|ignore|mistake|scratch|withdraw|wrong)\b/.test(
          text,
        ) ||
        /\b(?:i (?:meant|said)|not what i (?:asked|meant|said))\b/.test(text))) ||
    (hasCompetingScheduleValues &&
      /\b(?:actually|correction|i meant|instead|make that|rather than|wait)\b/.test(text)) ||
    /\b(?:disregard|forget|ignore|scratch|withdraw)\b.*\b(?:date|day|hour|reminder|schedule|time|weekday|weekend)\b/.test(
      text,
    );

  if (competingCorrection) return "competing-correction";

  const quotedSchedule =
    /["'‘’“”][^"'‘’“”]*\b(?:at|daily|each|every|hourly|monthly|weekly)\b[^"'‘’“”]*["'‘’“”]/.test(
      text,
    );

  const metaQuotation =
    quotedSchedule &&
    /\b(?:article|documentation|documenting|docs?|example|guide|handbook|manual|message|parsed|parser|phrase|sample|sentence|text|tooltip|tutorial|quote|quoted|wrote|contains?|mentions?|reads?|says?|shows?|uses?)\b/.test(
      text,
    );

  if (metaQuotation) return "quoted-schedule";

  if (
    scheduleCue.test(text) &&
    !hasDirectScheduleRequest(text) &&
    /\b(?:example|label|tutorial)\b.*\b(?:reads?|says?)\b/.test(text)
  ) {
    return "quoted-schedule";
  }

  if (
    /\b(?:copy|explain|translate)\b.*\b(?:phrase|sentence|string|words?)\b.*["'‘’“”]/.test(text)
  ) {
    return "quoted-schedule";
  }

  if (/^(?:copy|remove|translate)\b.*["'‘’“”].*["'‘’“”]/.test(text)) return "quoted-schedule";

  if (/\b(?:typed|wrote)\b.*["'‘’“”].*\b(?:every|monthly|weekly|daily|hourly)\b/.test(text)) {
    return "quoted-schedule";
  }

  // Explicit schedule-shaped language is common in prose that merely reports
  // a habit, timetable, label, recipe, or third-party event. These surfaces
  // are high-confidence model inputs but do not authorize creating a schedule.
  const reportedOrHabitual =
    scheduleCue.test(text) &&
    !hasDirectScheduleRequest(text) &&
    (/\bi usually\b/.test(text) ||
      /\baccording to\b/.test(text) ||
      /\b(?:card|description|documentation|event|manual|recipe|sign|timetable)\b.*\b(?:contains?|reads?|says?|shows?|written)\b/.test(
        text,
      ) ||
      /\b(?:contains?|reads?|says?|shows?|written)\b.*\b(?:card|description|documentation|event|manual|recipe|sign|timetable)\b/.test(
        text,
      ) ||
      /^(?:her|his|our|the|their)\b.*\b(?:arrives?|chimes?|is open|works?)\b/.test(text) ||
      /^(?:he|she|they)\s+(?:arrives?|leaves?|meets?|visits?|works?)\b/.test(text));

  if (reportedOrHabitual) return "descriptive-statement";

  if (
    scheduleCue.test(text) &&
    !hasDirectScheduleRequest(text) &&
    /^(?:i|we|you)\s+(?:eat|have|meet|work)\b/.test(text)
  ) {
    return "descriptive-statement";
  }

  if (
    scheduleCue.test(text) &&
    !hasDirectScheduleRequest(text) &&
    /^my\s+\w+\s+(?:calls?|goes?|runs?|visits?|walks?|works?)\b/.test(text)
  ) {
    return "descriptive-statement";
  }

  if (/^there\s+(?:are|is)\b/.test(text) && scheduleCue.test(text)) return "descriptive-statement";

  if (
    scheduleCue.test(text) &&
    !hasDirectScheduleRequest(text) &&
    /^(?:he|she|they|[a-z]+)\s+(?:explained|mentioned|reported|said|told)\b/.test(text)
  ) {
    return "descriptive-statement";
  }

  if (
    !hasDirectScheduleRequest(text) &&
    (/^(?:buses?|trains?)\b.*\barrive\b/.test(text) ||
      /^(?:his|her|their)\b.*\b(?:is due|performs?)\b/.test(text) ||
      /^my\b.*\bwakes?\b/.test(text) ||
      /^the\b.*\b(?:has|opens?)\b.*\b(?:printed|every|each|daily|weekly|monthly|hourly)\b/.test(
        text,
      ) ||
      /^(?:flights?|planes?)\b.*\bdepart\b/.test(text) ||
      /^the\s+manual\b.*\brecommends?\b/.test(text) ||
      /^this\b.*\b(?:costs?|charges?)\b.*\bper\s+(?:day|week|month)\b/.test(text) ||
      /^the\s+clock\b.*\bstrikes?\b/.test(text))
  ) {
    return "descriptive-statement";
  }

  if (
    !hasDirectScheduleRequest(text) &&
    (/^my\b.*\bleaves?\b/.test(text) ||
      /^(?:her|his|their)\b.*\brenews?\b/.test(text) ||
      /^the\s+(?:guide|handbook|manual)\b.*\b(?:recommends?|suggests?)\b/.test(text) ||
      /^the\s+(?:alarm|bell|clock)\b.*\b(?:rings?|strikes?)\b/.test(text))
  ) {
    return "descriptive-statement";
  }

  // A duration describes how long something continued; it is not a cadence.
  // Keep this independent of model labels so a token such as "hours" cannot
  // turn an ordinary past or present-tense observation into "every hour".
  const descriptiveDuration =
    /\b(?:last(?:s|ed)?|ran|runs?|took|takes?|work(?:s|ed)?)\s+(?:for\s+)?(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:minutes?|hours?|days?|weeks?|months?)\b/.test(
      text,
    );

  if (descriptiveDuration) return "historical-description";

  const historical =
    scheduleCue.test(text) &&
    (/\b(?:arrived|fired|formerly|had|happened|met|occurred|previous|previously|ran|rang|used to|was scheduled|were scheduled|worked|yesterday)\b/.test(
      text,
    ) ||
      /\b(?:already|earlier today|last (?:month|week|year)|years? ago)\b/.test(text) ||
      /\bbefore\b.*\b(?:migration|merger|outage|retired|retirement|shutdown)\b/.test(text));

  if (/^before\s+the\s+\w+\b/.test(text) && scheduleCue.test(text)) return "historical-description";
  if (historical) return "historical-description";

  if (/\b(?:back then|old rota|old schedule|in those days)\b/.test(text)) {
    return "historical-description";
  }

  if (
    (/^in\s+\d{4}\b/.test(text) || /\bfinished\b.*\bmonthly\b/.test(text)) &&
    scheduleCue.test(text)
  ) {
    return "historical-description";
  }

  const hypothetical =
    scheduleCue.test(text) &&
    (/^(?:imagine|suppose|supposing)\b/.test(text) ||
      (/^if\b/.test(text) && /\b(?:could|might|would|will)\b/.test(text)) ||
      /\bwould be\b.*\bif\b/.test(text) ||
      /\b(?:no decision has been made|we will tell you)\b/.test(text));

  if (hypothetical) return "non-scheduling-language";

  if (
    scheduleCue.test(text) &&
    (/\bunless\b.*\b(?:make more sense|might be better|would be better)\b/.test(text) ||
      /\bbetween\b.*\band\b.*\b(?:feels? right|seems? right|undecided)\b/.test(text) ||
      /\b(?:hold off|do not choose yet|set nothing yet|still deciding|not decided)\b/.test(text) ||
      (hasCompetingScheduleValues && /\bboth\b.*\bproposals?\b/.test(text)))
  ) {
    return "uncertain-alternative";
  }

  if (/\b(?:and|or|from|through)\s*[?.,;:]?$/.test(text) && scheduleCue.test(text)) {
    return "non-scheduling-language";
  }

  if (/^at\s+(?:half|quarter|[a-z]+|\d{1,2})\s+(?:past|after|to|before)\s*[?.,;:]?$/.test(text)) {
    return "non-scheduling-language";
  }

  if (
    /^(?:on\s+)?(?:mon(?:day)?s?|tue(?:sday)?s?|wed(?:nesday)?s?|thu(?:rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)\s+at\s*[?.,;:]?$/.test(
      text,
    )
  ) {
    return "non-scheduling-language";
  }

  if (
    /^(?:never mind about|do not schedule|don't schedule|withdraw my request|ignore the earlier)\b/.test(
      text,
    )
  ) {
    return "non-scheduling-language";
  }

  if (/^(?:cancel|delete|disable|remove)\b/.test(text)) return "non-scheduling-language";

  if (/^stop\b/.test(text) && !/\bbefore\s+\d{4}-\d{2}-\d{2}\b/.test(text)) {
    return "non-scheduling-language";
  }

  if (/\bevery\s+(?:a\s+)?few\s+(?:minutes?|hours?|days?|weeks?|months?)\b/.test(text)) {
    return "non-scheduling-language";
  }

  if (/\b(?:a|some)\s+few\s+times?\s+per\s+(?:day|week|month)\b/.test(text)) {
    return "non-scheduling-language";
  }

  if (/\bdo not\s+(?:add|arrange|create|make|run|schedule|send|set|start)\b/.test(text)) {
    return "non-scheduling-language";
  }

  if (
    /\b(?:near|around)\s+the\s+(?:beginning|middle|end)\s+of\s+(?:each|every|the)\s+month\b/.test(
      text,
    )
  ) {
    return "non-scheduling-language";
  }

  if (/\b(?:maybe|perhaps)\b/.test(text) && weekdayMentions >= 2) return "uncertain-alternative";

  if (/^at\s+(?:[a-z]+|\d{1,2})\s+minutes?\s+(?:past|after|to|before)\s*[?.,;:]?$/.test(text)) {
    return "non-scheduling-language";
  }

  if (/^between\s+\d{4}-\d{2}-\d{2}\s+and\s*[?.,;:]?$/.test(text)) return "non-scheduling-language";

  const recurrenceCues = [...text.matchAll(/\b(?:daily|each|every|hourly|monthly|weekly)\b/g)]
    .length;

  if (
    recurrenceCues > 1 &&
    /\b(?:another|both|candidate|compare|option|or|proposal|versus)\b/.test(text)
  ) {
    return "multiple-schedules";
  }

  return null;
}

/** Identify prose that mentions a possible schedule value without asking for
 * a schedule. This guard is deliberately applied only when the decoder found
 * no deterministic schedule evidence; explicit recurrence, clock, bound, and
 * exception syntax remains eligible even inside a declarative carrier. */
function unrequestedDeclarativeContext(text: string): boolean {
  text = text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[.!?]+$/, "")
    .replace(/\s+/g, " ")
    .trim();

  if (hasDirectScheduleRequest(text)) return false;

  const bareSchedule =
    /^(?:at\b|daily\b|each\b|every\b|hourly\b|monthly\b|once\b|on\b|starting\b|the\s+(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|last)\b|until\b|weekly\b|weekdays?\b|weekends?\b|mon(?:day)?s?\b|tue(?:sday)?s?\b|wed(?:nesday)?s?\b|thu(?:rsday)?s?\b|fri(?:day)?s?\b|sat(?:urday)?s?\b|sun(?:day)?s?\b|\d{1,2}(?:st|nd|rd|th)\b)/.test(
      text,
    );

  if (bareSchedule) return false;

  return /\b(?:am|are|arrives?|calls?|closes?|ends?|happens?|is|leaves?|meets?|occurs?|opens?|runs?|starts?|takes?|was|were|works?)\b/.test(
    text,
  );
}

/** Command/request evidence used only to decide whether an otherwise explicit
 * schedule surface belongs to the user request. This does not create any
 * schedule values by itself. */
function hasDirectScheduleRequest(text: string): boolean {
  return (
    /(?:^|,\s*)(?:actually[,:]?\s+|for now,?\s+|from now on[,:]?\s+|kindly\s+|please\s+)?(?:just\s+)?(?:add|arrange|begin|book|charge|check|close|configure|continue|create|do|enable|give|have|issue|keep|let|make|meet|notify|ping|poll|put|refresh|remind|repeat|run|schedule|send|set|start|take|trigger|use)\b/.test(
      text,
    ) ||
    /\bplease\s+(?:add|arrange|book|configure|create|make|remind|run|schedule|send|set)\b/.test(
      text,
    ) ||
    /\b(?:i(?:'d| would) like|i(?:\s+still)?\s+(?:need|want)|let's)\b/.test(text) ||
    /\b(?:can|could|may|will|would)\s+(?:i|you)\b/.test(text)
  );
}

/** Exact, context-free surfaces whose meaning is completely determined by the
 * shared lexicon. Keeping these in the decoder makes short inputs robust to a
 * sparse-hash collision without treating a weekday mentioned in prose as a
 * schedule. Longer conversational inputs still go through the tagger. */
function exactLexicalIntent(entries: readonly Entry[], sourceText?: string): DecodedIntent | null {
  const text = joinEntries(entries);
  const unsafeContext = unsafeScheduleContext(sourceText ?? text);

  if (unsafeContext !== null) throw new UnsafeScheduleContextError(unsafeContext);

  // Explicit corrections, cancellations, questions, and hedged alternatives
  // are not schedules even when they contain otherwise valid schedule words.
  if (
    /\b(?:do not|don't|did not|didn't)\s+mean\b/.test(text) ||
    /^(?:no,?\s+not\b|actually,?\s+never mind\b)/.test(text) ||
    /^maybe\b.*\bor\b/.test(text) ||
    /^how (?:do|does|would|can)\b.*\b(?:work|mean)\b/.test(text) ||
    /^(?:should|who|why)\b/.test(text) ||
    /^forget what i said\b/.test(text) ||
    /\b(?:meeting happened|deadline moved|people joined|finished the report|arrived at|used to)\b/.test(
      text,
    ) ||
    /\b(?:yesterday|formerly|previously|in the past)\b/.test(text) ||
    /^if\b.*\b(?:would|could|might)\b/.test(text) ||
    /\b(?:documentation|docs?|tutorial|example|phrase|sample|quoted?|text)\b.*(?:contains?|says?|shows?|uses?|reads?)\b.*["'‘’“”].*\b(?:every|each|daily|weekly|monthly|hourly|at)\b/.test(
      text,
    ) ||
    /\b\d{1,2}(?:st|nd|rd|th)\s+(?:person|people|place|item|chapter|page|version|floor|attempt|runner|customer)\b/.test(
      text,
    ) ||
    /\b(?:\d+|[a-z]+)\s+(?:minutes?|hours?|days?|weeks?|months?)\s+(?:old|long|ago|remaining)\b/.test(
      text,
    ) ||
    /^the office closes on\b/.test(text) ||
    /\bby another system$/.test(text) ||
    /\b(?:except|excluding|through|from|of)$/.test(text) ||
    /\b(?:at half past|monthly on the)$/.test(text)
  ) {
    throw new UnsafeScheduleContextError("non-scheduling-language");
  }

  const weekdaySurface = text.replace(/^(?:every|each|on)\s+/, "");

  const weekday =
    WEEKDAY_ALIASES[weekdaySurface] ??
    WEEKDAY_PLURAL_ALIASES[weekdaySurface] ??
    WEEKDAY_DESCRIPTIONS[weekdaySurface];

  if (weekday !== undefined) {
    return {
      intent: { frequency: { unit: "week", interval: 1 }, weekdays: [weekday] },
      confidence: 1,
      evidence: exactEvidence(),
    };
  }

  const group = WEEKDAY_GROUPS[weekdaySurface];

  if (group !== undefined) {
    return {
      intent: { frequency: { unit: "week", interval: 1 }, weekdays: [...group] },
      confidence: 1,
      evidence: exactEvidence(),
    };
  }

  const ordinalWeekday = /^([a-z0-9]+)\s+([a-z]+)$/.exec(text);

  if (ordinalWeekday !== null) {
    const ordinal = ORDINAL_ALIASES[ordinalWeekday[1]!];
    const ordinalDay = WEEKDAY_ALIASES[ordinalWeekday[2]!];

    if (ordinal !== undefined && ordinalDay !== undefined) {
      return {
        intent: {
          frequency: { unit: "month", interval: 1 },
          ordinalWeekday: { ordinal, weekday: ordinalDay },
        },
        confidence: 1,
        evidence: exactEvidence(),
      };
    }
  }

  const numericTime = /^at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/.exec(text);

  if (numericTime !== null) {
    const time = resolveTime(
      Number(numericTime[1]),
      numericTime[2] === undefined ? 0 : Number(numericTime[2]),
      numericTime[3] as "am" | "pm" | undefined,
    );

    if (validateTime(time) === null) {
      return {
        intent: { frequency: { unit: "day", interval: 1 }, time },
        confidence: 1,
        evidence: exactEvidence(),
      };
    }
  }

  const wordTime = /^at\s+([a-z]+)(?:\s+(am|pm))?$/.exec(text);

  if (wordTime !== null) {
    const hourValue = numberFrom(wordTime[1]!);

    if (hourValue !== undefined) {
      const time = resolveTime(hourValue, 0, wordTime[2] as "am" | "pm" | undefined);

      if (validateTime(time) === null) {
        return {
          intent: { frequency: { unit: "day", interval: 1 }, time },
          confidence: 1,
          evidence: exactEvidence(),
        };
      }
    }
  }

  return null;
}

function regexEscape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Recover an explicit, schedule-scoped day group even when the tagger assigns
 * its token a structural role. The cue is deliberately required: an incidental
 * mention such as "weekday prices changed" must not become a schedule. */
function explicitDayGroup(entries: readonly Entry[]): Weekday[] | undefined {
  const text = joinEntries(entries);

  for (const [surface, weekdays] of Object.entries(WEEKDAY_GROUPS).sort(
    ([left], [right]) => right.length - left.length,
  )) {
    const phrase = regexEscape(surface).replace(/\\ /g, "\\s+");

    const pattern = new RegExp(
      `\\b(?:every|each|on|weekly\\s+on|every\\s+week\\s+on|each\\s+week\\s+on)\\s+(?:1\\s+)?${phrase}\\b`,
    );

    if (
      pattern.test(text) ||
      (hasDirectScheduleRequest(text) && new RegExp(`^${phrase}\\b`).test(text))
    ) {
      return [...weekdays];
    }
  }

  return undefined;
}

/** Join a run of tokens the way the renderer emitted them. */
function joinTokens(tokens: readonly RawToken[]): string {
  let text = "";

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    const previous = tokens[index - 1];

    if (previous !== undefined && previous.kind === 0 && token.kind === 0) text += " ";
    text += token.text;
  }

  return text.toLowerCase();
}

/** Reconstruct a normalized sentence from non-whitespace tokens. Unlike
 * joinTokens(), this preserves a whitespace gap between unlike token kinds
 * (for example the word/number boundary in `at 9`). */
function joinEntries(entries: readonly Entry[]): string {
  let text = "";

  for (let index = 0; index < entries.length; index++) {
    const token = entries[index]!.token;
    const previous = entries[index - 1]?.token;

    if (previous !== undefined && token.start > previous.end) text += " ";
    text += token.text;
  }

  return text.toLowerCase();
}

function numberFrom(text: string): number | undefined {
  if (/^\d+$/.test(text)) return Number(text);
  const parts = text.replace(/\boh\b/g, "zero").split(/[\s-]+/);
  let total = 0;

  for (const part of parts) {
    const value = NUMBER_WORDS[part];

    if (value === undefined) return undefined;
    total += value;
  }

  return total;
}

function weekdayOf(text: string): Weekday | undefined {
  const lower = text.toLowerCase();

  const exact =
    WEEKDAY_ALIASES[lower] ?? WEEKDAY_PLURAL_ALIASES[lower] ?? WEEKDAY_DESCRIPTIONS[lower];

  if (exact !== undefined) return exact;

  // Callers either have a WEEKDAY model role or an explicit schedule-scoped
  // grammar. Resolve one character edit only when every matching surface maps
  // to the same weekday. This also covers noisy plurals (for example
  // `mondayys`) without guessing an arbitrary word in prose.
  return uniqueSingleEditAlias(
    lower,
    {
      ...WEEKDAY_ALIASES,
      ...WEEKDAY_PLURAL_ALIASES,
    },
    5,
  );
}

/** Resolve a one-character deletion, insertion, substitution, repetition, or
 * adjacent transposition. Multiple surface aliases are fine when they all
 * denote the same value; disagreement is deliberately treated as ambiguous. */
function uniqueSingleEditAlias<T>(
  surface: string,
  aliases: Readonly<Record<string, T>>,
  minimumLength: number,
): T | undefined {
  const lower = surface.toLowerCase();

  if (!/^[a-z]+$/.test(lower) || lower.length < minimumLength) return undefined;
  const candidates = new Set<T>();

  for (const [alias, value] of Object.entries(aliases)) {
    if (
      alias.length >= minimumLength &&
      /^[a-z]+$/.test(alias) &&
      isSingleEditVariant(lower, alias)
    ) {
      candidates.add(value);
    }
  }

  return candidates.size === 1 ? candidates.values().next().value : undefined;
}

/** Normalize only members of a small grammar vocabulary. This is used inside
 * already schedule-shaped phrases, never as a general-purpose spellchecker. */
function normalizeGrammarWords(text: string, words: readonly string[], minimumLength = 4): string {
  const aliases = Object.fromEntries(words.map((word) => [word, word]));

  return normalizeGrammarAliases(text, aliases, minimumLength);
}

function normalizeGrammarAliases(
  text: string,
  aliases: Readonly<Record<string, string>>,
  minimumLength = 4,
): string {
  return text.replace(/[a-z]+/g, (surface) => {
    if (aliases[surface] !== undefined) return aliases[surface]!;

    return uniqueSingleEditAlias(surface, aliases, minimumLength) ?? surface;
  });
}

function isSingleEditVariant(left: string, right: string): boolean {
  if (left === right || Math.abs(left.length - right.length) > 1) return false;

  if (left.length === right.length) {
    const differences: number[] = [];

    for (let index = 0; index < left.length; index++) {
      if (left[index] !== right[index]) differences.push(index);
    }

    if (differences.length === 1) return true;

    return (
      differences.length === 2 &&
      differences[1] === differences[0]! + 1 &&
      left[differences[0]!] === right[differences[1]!] &&
      left[differences[1]!] === right[differences[0]!]
    );
  }

  const shorter = left.length < right.length ? left : right;
  const longer = left.length < right.length ? right : left;
  let shortIndex = 0;
  let longIndex = 0;
  let skipped = false;

  while (shortIndex < shorter.length && longIndex < longer.length) {
    if (shorter[shortIndex] === longer[longIndex]) {
      shortIndex += 1;
      longIndex += 1;
    } else {
      if (skipped) return false;
      skipped = true;
      longIndex += 1;
    }
  }

  return true;
}

function isRangeConnector(text: string): boolean {
  const lower = text.toLowerCase();

  return (
    lower === WORDS.range.through ||
    lower === "thru" ||
    lower === "up to and including" ||
    lower === WORDS.range.to ||
    lower === WORDS.glue.dash ||
    uniqueSingleEditAlias(lower, { through: true }, 4) === true
  );
}

function weekdayRange(start: Weekday, end: Weekday): Weekday[] {
  const result: Weekday[] = [];
  const endIndex = WEEKDAY_SEQUENCE.indexOf(end);
  let index = WEEKDAY_SEQUENCE.indexOf(start);

  for (;;) {
    result.push(WEEKDAY_SEQUENCE[index]!);
    if (index === endIndex) break;
    index = (index + 1) % WEEKDAY_SEQUENCE.length;
  }

  return result;
}

interface ExplicitFrequencyEvidence {
  unit: FrequencyUnit;
  interval: number;
  valueStart: number;
  valueEnd: number;
  /** Contextual evidence still requires agreement from the global kind head. */
  contextual?: boolean;
}

/** Locate a numeric cadence behind an explicit recurrence noun or verb. The
 * value offsets let decoding distinguish an interval number from a clock even
 * when the neural role for that number is wrong. */
function explicitFrequency(entries: readonly Entry[]): ExplicitFrequencyEvidence | undefined {
  const text = joinEntries(entries);

  if (
    /\bmonthly\b/.test(text) &&
    hasDirectScheduleRequest(text) &&
    !/\bevery\s+(?:\d+|second|third|fourth)\s+months?\b/.test(text)
  ) {
    return { unit: "month", interval: 1, valueStart: -1, valueEnd: -1 };
  }

  const halfHour = /\bevery\s+(?:one\s+)?half[-\s]+hour\b|\bhalf[-\s]+hour\s+intervals?\b/.exec(
    text,
  );

  if (halfHour !== null) return { unit: "minute", interval: 30, valueStart: -1, valueEnd: -1 };

  if (
    /\b(?:(?:every\s+(?:other|second))|alternat(?:e|ing))\s+(?:mon(?:day)?s?|tue(?:sday)?s?|wed(?:nesday)?s?|thu(?:rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)\b/.test(
      text,
    )
  ) {
    return { unit: "week", interval: 2, valueStart: -1, valueEnd: -1 };
  }

  if (/\b(?:every|each)\s+(?:morning|evening|night)\b/.test(text)) {
    return { unit: "day", interval: 1, valueStart: -1, valueEnd: -1 };
  }

  if (/\bfortnightly\b/.test(text)) {
    return { unit: "week", interval: 2, valueStart: -1, valueEnd: -1 };
  }

  const numberWords = ["oh", ...Object.keys(NUMBER_WORDS)]
    .sort((left, right) => right.length - left.length)
    .map(regexEscape)
    .join("|");

  const numberSurface = `(?:\\d+|(?:${numberWords})(?:[-\\s](?:${numberWords}))?)`;

  const ordinalInterval = new RegExp(
    `\\bevery\\s+(second|third|fourth)\\s+(minutes?|hours?|days?|weeks?|months?)\\b`,
  ).exec(text);

  if (ordinalInterval !== null) {
    const interval = { second: 2, third: 3, fourth: 4 }[
      ordinalInterval[1]! as "second" | "third" | "fourth"
    ];

    const unit = UNIT_ALIASES[ordinalInterval[2]!];

    if (typeof interval === "number" && interval > 0 && unit !== undefined) {
      return { unit, interval, valueStart: -1, valueEnd: -1 };
    }
  }

  const hyphenInterval = new RegExp(
    `\\b(?:at\\s+)?(${numberSurface})[-\\s]+(minutes?|hours?|days?|weeks?|months?)[-\\s]+intervals?\\b`,
  ).exec(text);

  if (hyphenInterval !== null) {
    const interval = numberFrom(hyphenInterval[1]!);
    const unit = UNIT_ALIASES[hyphenInterval[2]!];

    if (interval !== undefined && unit !== undefined) {
      return { unit, interval, valueStart: -1, valueEnd: -1 };
    }
  }

  // Anchor the pair to its recurrence cue first. A broad pair scan can consume
  // `me every` and otherwise skip the overlapping `twenty minutes` pair.
  const direct = new RegExp(
    `\\b(?:every|each|once\\s+(?:per|every)|recurring\\s+every)\\s+(${numberSurface})\\s+([a-z]+)\\b`,
  ).exec(text);

  if (direct !== null) {
    const interval = numberFrom(direct[1]!);
    const unit = UNIT_ALIASES[direct[2]!] ?? uniqueSingleEditAlias(direct[2]!, UNIT_ALIASES, 3);

    if (interval !== undefined && unit !== undefined) {
      const valueStart = direct.index + direct[0].indexOf(direct[1]!);

      return {
        unit,
        interval,
        valueStart,
        valueEnd: valueStart + direct[1]!.length,
      };
    }
  }

  const pattern = new RegExp(`\\b(${numberSurface})\\s+([a-z]+)\\b`, "g");

  for (const match of text.matchAll(pattern)) {
    const interval = numberFrom(match[1]!);
    const unit = UNIT_ALIASES[match[2]!] ?? uniqueSingleEditAlias(match[2]!, UNIT_ALIASES, 3);

    if (interval === undefined || unit === undefined) continue;
    const start = match.index!;

    const prefix = normalizeGrammarWords(text.slice(Math.max(0, start - 48), start), [
      "every",
      "each",
      "interval",
      "cadence",
      "cycle",
      "recurring",
    ]);

    const suffix = normalizeGrammarWords(
      text.slice(start + match[0]!.length, start + match[0]!.length + 24),
      ["between", "runs", "apart", "cycle"],
    );

    if (
      !/\b(?:every|each|once\s+per|interval(?:\s+is|\s+of)?|cadence(?:\s+is|\s+of)?|cycle(?:\s+is|\s+of)?|recurring\s+every)\s*$/.test(
        prefix,
      ) &&
      !/^\s*(?:between\s+runs|apart|cycle)\b/.test(suffix)
    ) {
      continue;
    }

    const valueOffset = match[0]!.indexOf(match[1]!);
    const valueStart = start + valueOffset;

    return {
      unit,
      interval,
      valueStart,
      valueEnd: valueStart + match[1]!.length,
    };
  }

  // Unit-interval schedules have no explicit numeric value to relabel. A
  // recurrence cue plus a uniquely recoverable unit is still complete
  // evidence, e.g. `evrry day`.
  const normalized = normalizeGrammarWords(
    text,
    ["every", "each", ...Object.keys(UNIT_ALIASES)],
    3,
  );

  const unitInterval = /\b(?:every|each)\s+([a-z]+)\b/.exec(normalized);

  if (unitInterval !== null) {
    const unit = UNIT_ALIASES[unitInterval[1]!];

    if (unit !== undefined) return { unit, interval: 1, valueStart: -1, valueEnd: -1 };
  }

  const everySingle = /\b(?:every|each)\s+single\s+([a-z]+)\b/.exec(normalized);

  if (everySingle !== null) {
    const unit = UNIT_ALIASES[everySingle[1]!];

    if (unit !== undefined) return { unit, interval: 1, valueStart: -1, valueEnd: -1 };
  }

  const directiveAdverb =
    /\b(?:please|have|keep|make|let|schedule|run|send|notify|remind|alert|ping|check|refresh|repeat|do)\b.*\b(daily|hourly|weekly)\b/.exec(
      normalized,
    );

  if (directiveAdverb !== null) {
    const unit = ADVERB_FREQUENCIES[directiveAdverb[1]!];

    if (unit !== undefined) {
      return {
        unit,
        interval: 1,
        valueStart: -1,
        valueEnd: -1,
        ...(hasDirectScheduleRequest(normalized) ? {} : { contextual: true }),
      };
    }
  }

  const requestedAdverb = /\b(daily|hourly|weekly|monthly)\b/.exec(normalized);

  if (requestedAdverb !== null && hasDirectScheduleRequest(normalized)) {
    const unit =
      requestedAdverb[1] === "monthly" ? "month" : ADVERB_FREQUENCIES[requestedAdverb[1]!];

    if (unit !== undefined) return { unit, interval: 1, valueStart: -1, valueEnd: -1 };
  }

  const idiom = Object.entries(FREQUENCY_IDIOMS)
    .sort(([left], [right]) => right.length - left.length)
    .find(([surface]) =>
      new RegExp(`\\b${regexEscape(surface).replace(/\\ /g, "\\s+")}\\b`).test(text),
    );

  if (idiom !== undefined) return { ...idiom[1], valueStart: -1, valueEnd: -1 };

  const constrainedAdverb =
    /^(daily|hourly|weekly)\b.*\b(?:apart from|but not(?: on)?|except(?: on)?|excluding|omit(?:ting)?|skip(?:ping)?)\b/.exec(
      normalized,
    );

  if (constrainedAdverb !== null) {
    const unit = ADVERB_FREQUENCIES[constrainedAdverb[1]!];

    if (unit !== undefined) {
      return {
        unit,
        interval: 1,
        valueStart: -1,
        valueEnd: -1,
        contextual: true,
      };
    }
  }

  return undefined;
}

function normalizeWeekdays(days: readonly Weekday[]): Weekday[] {
  const selected = new Set(days);

  return WEEKDAY_SEQUENCE.filter((day) => selected.has(day));
}

function meridiemOf(text: string): "am" | "pm" | undefined {
  const lower = text.toLowerCase().replace(/[.\s]/g, "");

  if (lower === WORDS.meridiem.am || lower === "a.m") return "am";
  if (lower === WORDS.meridiem.pm || lower === "p.m") return "pm";
  if (lower === WORDS.qualifier.morning.replace(/\s/g, "")) return "am";

  if (
    lower === WORDS.qualifier.afternoon.replace(/\s/g, "") ||
    lower === WORDS.qualifier.evening.replace(/\s/g, "") ||
    lower === WORDS.qualifier.night.replace(/\s/g, "")
  ) {
    return "pm";
  }

  return undefined;
}

/** Normalize only unambiguous calendar-date surfaces. The intent contract uses
 * ISO dates even when the user says a month name. */
function calendarDateFrom(text: string): string | undefined {
  const normalized = text.toLowerCase().replace(/[,.]/g, " ").replace(/\s+/g, " ").trim();
  const compact = normalized.replace(/\s+/g, "");
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(compact);
  let year: number;
  let month: number;
  let day: number;

  if (iso !== null) {
    year = Number(iso[1]);
    month = Number(iso[2]);
    day = Number(iso[3]);
  } else {
    const monthFirst = /^([a-z]+)(\d{1,2})(?:st|nd|rd|th)?(\d{4})$/.exec(compact);
    const dayFirst = /^(\d{1,2})(?:st|nd|rd|th)?([a-z]+)(\d{4})$/.exec(compact);

    if (monthFirst !== null && MONTH_ALIASES[monthFirst[1]!] !== undefined) {
      month = MONTH_ALIASES[monthFirst[1]!]!;
      day = Number(monthFirst[2]);
      year = Number(monthFirst[3]);
    } else if (dayFirst !== null && MONTH_ALIASES[dayFirst[2]!] !== undefined) {
      day = Number(dayFirst[1]);
      month = MONTH_ALIASES[dayFirst[2]!]!;
      year = Number(dayFirst[3]);
    } else {
      return undefined;
    }
  }

  const candidate = new Date(Date.UTC(year, month - 1, day));

  if (
    year < 1 ||
    candidate.getUTCFullYear() !== year ||
    candidate.getUTCMonth() !== month - 1 ||
    candidate.getUTCDate() !== day
  ) {
    return undefined;
  }

  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function offsetIsoDate(iso: string, days: number): string | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);

  if (match === null) return undefined;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days));

  return `${String(date.getUTCFullYear()).padStart(4, "0")}-${String(
    date.getUTCMonth() + 1,
  ).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

interface DateCandidate {
  start: number;
  end: number;
  iso: string;
}

function calendarDateCandidates(text: string): DateCandidate[] {
  const months = Object.keys(MONTH_ALIASES)
    .sort((left, right) => right.length - left.length)
    .map(regexEscape)
    .join("|");

  const ordinal = "(?:st|nd|rd|th)?";

  const source =
    `\\d{4}-\\d{2}-\\d{2}` +
    `|(?:${months})\\s+\\d{1,2}${ordinal}(?:,)?\\s+\\d{4}` +
    `|\\d{1,2}${ordinal}\\s+(?:${months})(?:,)?\\s+\\d{4}`;

  const result: DateCandidate[] = [];

  for (const match of text.matchAll(new RegExp(source, "gi"))) {
    const iso = calendarDateFrom(match[0]);

    if (iso !== undefined) {
      result.push({ start: match.index!, end: match.index! + match[0].length, iso });
    }
  }

  return result;
}

/** Extract only visibly schedule-scoped recurrence bounds. */
function explicitBounds(entries: readonly Entry[]): RecurrenceBounds | undefined {
  const text = joinEntries(entries);
  const dates = calendarDateCandidates(text);

  if (dates.length === 0) {
    if (
      /\b(?:every|each|daily|weekly|monthly)\b/.test(text) &&
      /\b(?:from|starting|beginning|until|before|ending|effective|commencing|between)\b/.test(
        text,
      ) &&
      /\b\d{4}-\d{2}-\d{2}\b/.test(text)
    ) {
      throw new RangeError("decodeIntent: invalid bound date");
    }

    return undefined;
  }

  if (/\b(?:apart from|but not|except|excluding|leave|omit|skip|unless)\b/.test(text)) {
    return undefined;
  }

  if (dates.length >= 2) {
    const first = dates[0]!;
    const second = dates[1]!;

    const before = normalizeGrammarWords(text.slice(Math.max(0, first.start - 32), first.start), [
      "between",
      "from",
      "starting",
      "beginning",
    ]);

    const between = normalizeGrammarWords(text.slice(first.end, second.start), [
      "through",
      "until",
      "and",
    ]);

    if (
      !/\b(?:between|from|starting|beginning)\s*$/.test(before) &&
      !/\b(?:through|to|until|and)\b/.test(between)
    ) {
      return undefined;
    }

    const inclusive = /\bthrough(?:\s+and\s+including)?\b/.test(between);
    const end = inclusive ? offsetIsoDate(second.iso, 1) : second.iso;

    if (end === undefined) throw new RangeError("decodeIntent: invalid inclusive end date");

    return { start: first.iso, end };
  }

  const date = dates[0]!;

  const before = normalizeGrammarWords(text.slice(Math.max(0, date.start - 48), date.start), [
    "ending",
    "through",
    "including",
    "until",
    "before",
    "stopping",
    "from",
    "starting",
    "beginning",
    "after",
    "effective",
    "commencing",
    "begin",
    "begin on",
    "as of",
    "only through",
  ]);

  const after = normalizeGrammarWords(text.slice(date.end, Math.min(text.length, date.end + 48)), [
    "every",
    "each",
    "daily",
    "weekly",
    "monthly",
  ]);

  if (/\b(?:ending\s+on|through|through\s+and\s+including)\s*$/.test(before)) {
    const end = offsetIsoDate(date.iso, 1);

    if (end === undefined) throw new RangeError("decodeIntent: invalid inclusive end date");

    return { end };
  }

  if (/\bonly\s+through\s*$/.test(before)) {
    const end = offsetIsoDate(date.iso, 1);

    if (end === undefined) throw new RangeError("decodeIntent: invalid inclusive end date");

    return { end };
  }

  if (
    /\b(?:until|before|up\s+until|ending|ending\s+before|stopping\s+before|but\s+not\s+on\s+or\s+after)\s*$/.test(
      before,
    )
  ) {
    return { end: date.iso };
  }

  if (
    /\b(?:from|start\b.*\bon|starting|starting\s+on|starting\s+from|begin|begin\b.*\bon|beginning\s+on|beginning\s+from|as\s+of|on\s+or\s+after|effective|effective\s+on|commencing|commencing\s+on)\s*$/.test(
      before,
    )
  ) {
    return { start: date.iso };
  }

  if (/^\s*,\s*(?:every|each|daily|weekly|monthly)\b/.test(after)) return { start: date.iso };

  return undefined;
}

function reconcileBounds(
  predicted: RecurrenceBounds | undefined,
  explicit: RecurrenceBounds,
): RecurrenceBounds {
  if (
    (predicted?.start !== undefined &&
      explicit.start !== undefined &&
      predicted.start !== explicit.start) ||
    (predicted?.end !== undefined && explicit.end !== undefined && predicted.end !== explicit.end)
  ) {
    throw new RangeError("decodeIntent: predicted bounds conflict with explicit dates");
  }

  // The lexical cue determines which side(s) are present. Discard an extra
  // neural side rather than preserving a bound that has no textual evidence.
  return explicit;
}

/** Recover exclusions from the text after an explicit exception cue. The
 * decoder has already established that the utterance is a schedule; this
 * helper only prevents a tagger error from dropping or changing visible
 * exception values. */
interface ExplicitExceptionEvidence {
  exceptions: ScheduleExceptions;
  /** Weekdays mentioned outside the exception clause, if any. */
  selectorWeekdays: Weekday[];
}

interface WeekdayMention {
  start: number;
  end: number;
  weekday: Weekday;
}

function weekdayMentions(text: string, allowFuzzy = false): WeekdayMention[] {
  const mentions: WeekdayMention[] = [];

  const surfaces = [
    ...Object.entries(WEEKDAY_DESCRIPTIONS),
    ...Object.entries(WEEKDAY_PLURAL_ALIASES),
    ...Object.entries(WEEKDAY_ALIASES),
  ].sort(([left], [right]) => right.length - left.length) as Array<[string, Weekday]>;

  for (const [surface, weekday] of surfaces) {
    const pattern = regexEscape(surface).replace(/\\ /g, "\\s+");
    const matches = text.matchAll(new RegExp(`(?:^|[^a-z])(${pattern})(?=$|[^a-z])`, "g"));

    for (const match of matches) {
      const prefix = match[0]!.length - match[1]!.length;
      const start = match.index! + prefix;
      const end = start + match[1]!.length;

      if (!mentions.some((item) => start < item.end && end > item.start)) {
        mentions.push({ start, end, weekday });
      }
    }
  }

  if (allowFuzzy) {
    for (const match of text.matchAll(/[a-z]+/g)) {
      const start = match.index!;
      const end = start + match[0].length;

      if (mentions.some((item) => start < item.end && end > item.start)) continue;
      const weekday = weekdayOf(match[0]);

      if (weekday !== undefined) mentions.push({ start, end, weekday });
    }
  }

  return mentions.sort((left, right) => left.start - right.start || left.end - right.end);
}

function weekdaysIn(text: string, allowFuzzy = false): Weekday[] {
  const weekdays = new Set<Weekday>();

  for (const [surface, members] of Object.entries(WEEKDAY_GROUPS).sort(
    ([left], [right]) => right.length - left.length,
  )) {
    const pattern = regexEscape(surface).replace(/\\ /g, "\\s+");

    if (new RegExp(`(?:^|[^a-z])${pattern}(?:$|[^a-z])`).test(text)) {
      for (const weekday of members) weekdays.add(weekday);
    }
  }

  for (const mention of weekdayMentions(text, allowFuzzy)) weekdays.add(mention.weekday);

  return normalizeWeekdays([...weekdays]);
}

/** Preserve an explicitly written weekday range even when the model misses
 * its connector or second endpoint. */
function explicitWeekdayRange(text: string, allowFuzzy = false): Weekday[] | undefined {
  const mentions = weekdayMentions(text, allowFuzzy);

  for (let index = 0; index + 1 < mentions.length; index++) {
    const first = mentions[index]!;
    const second = mentions[index + 1]!;
    const connector = text.slice(first.end, second.start).toLowerCase().replace(/[.,]/g, "").trim();

    if (isRangeConnector(connector)) return weekdayRange(first.weekday, second.weekday);
  }

  return undefined;
}

/** Recover an adjacent ordinal-weekday selector after the neural decoder has
 * already established that the utterance is a schedule. */
function explicitOrdinalWeekday(
  entries: readonly Entry[],
): ScheduleIntent["ordinalWeekday"] | undefined {
  const text = joinEntries(entries);

  const ordinals = Object.entries(ORDINAL_ALIASES).sort(
    ([left], [right]) => right.length - left.length,
  );

  const weekdays = [
    ...Object.entries(WEEKDAY_DESCRIPTIONS),
    ...Object.entries(WEEKDAY_PLURAL_ALIASES),
    ...Object.entries(WEEKDAY_ALIASES),
  ].sort(([left], [right]) => right.length - left.length) as Array<[string, Weekday]>;

  for (const [ordinalSurface, ordinal] of ordinals) {
    const ordinalPattern = regexEscape(ordinalSurface).replace(/\\ /g, "\\s+");

    for (const [weekdaySurface, weekday] of weekdays) {
      const weekdayPattern = regexEscape(weekdaySurface).replace(/\\ /g, "\\s+");

      if (
        new RegExp(`(?:^|[^a-z0-9])${ordinalPattern}\\s+${weekdayPattern}(?=$|[^a-z])`).test(text)
      ) {
        return { ordinal, weekday };
      }
    }
  }

  // The exact phrase scan above handles descriptive weekday expressions. For
  // a single noisy word pair, require both an ordinal value and weekday value
  // to resolve uniquely before preserving the explicit selector.
  const words = [...text.matchAll(/[a-z]+/g)].map((match) => match[0]);

  for (let index = 0; index + 1 < words.length; index += 1) {
    const ordinalSurface = words[index]!;
    const weekdaySurface = words[index + 1]!;

    const ordinal =
      ORDINAL_ALIASES[ordinalSurface] ?? uniqueSingleEditAlias(ordinalSurface, ORDINAL_ALIASES, 4);

    const weekday = weekdayOf(weekdaySurface);

    if (ordinal !== undefined && weekday !== undefined) return { ordinal, weekday };
  }

  return undefined;
}

/** Extract a monthly calendar-day selector when both the value and monthly
 * recurrence are explicit. Requiring both halves keeps ordinary mentions of a
 * numbered day from becoming schedules. */
function explicitDayOfMonth(entries: readonly Entry[]): number | undefined {
  const rawText = joinEntries(entries);
  // These substitutions are evaluated only by the strict monthly day grammar
  // below. Neither a near-`day` word nor a near-`monthly` word is sufficient
  // by itself to create a schedule.
  const text = normalizeGrammarWords(rawText, ["day", "monthly", "month"], 2);

  if (!/\b(?:monthly|every\s+month|each\s+month)\b/.test(text)) return undefined;

  const wordOrdinals = [...DAY_ORDINALS]
    .sort((left, right) => right.length - left.length)
    .map((value) => regexEscape(value).replace(/\ /g, "\\s+"))
    .join("|");

  for (const [surface, day] of DAY_ORDINALS.map(
    (surface, index) => [surface, index + 1] as const,
  ).sort(([left], [right]) => right.length - left.length)) {
    const phrase = regexEscape(surface).replace(/\ /g, "\\s+");

    if (
      new RegExp(
        `\\b(?:the\\s+)?${phrase}\\s+(?:(?:a\\s+)?monthly|(?:every|each)\\s+month)\\b`,
      ).test(text)
    ) {
      return day;
    }
  }

  const valuePattern = `(\\d{1,2})(?:st|nd|rd|th)?|(${wordOrdinals})`;

  const patterns = [
    new RegExp(`\\b(?:the\\s+)?(?:${valuePattern})\\s+day\\b`),
    new RegExp(`\\bday\\s+(?:${valuePattern})\\b`),
    new RegExp(`\\b(?:the\\s+)?(?:${valuePattern})\\s+of\\s+(?:every|each)?\\s*month\\b`),
    new RegExp(`\\b(?:the\\s+)?(?:${valuePattern})\\s+(?:a\\s+)?monthly\\b`),
    new RegExp(`\\b(?:the\\s+)?(?:${valuePattern})\\s+(?:every|each)\\s+month\\b`),
    new RegExp(`\\bmonthly\\b.*?\\b(?:on|for)\\s+(?:the\\s+)?(?:${valuePattern})\\b`),
    new RegExp(`\\bon\\s+(?:the\\s+)?(?:${valuePattern})\\b(?=\\s*(?:,|$|at\\b|if\\b))`),
  ];

  for (const pattern of patterns) {
    const match = pattern.exec(text);

    if (match === null) continue;
    const surface = match.slice(1).find((value) => value !== undefined);

    if (surface === undefined) continue;
    const numeric = /^\d+$/.test(surface) ? Number(surface) : DAY_ORDINALS.indexOf(surface) + 1;

    if (validateDayOfMonth(numeric) === null) return numeric;
    throw new RangeError(`decodeIntent: invalid explicit day of month: ${numeric}`);
  }

  const cardinal = new RegExp(`\\bday\\s+(${numberSurfaceForDay()})\\b`).exec(text);

  if (cardinal !== null) {
    const numeric = numberFrom(cardinal[1]!);

    if (numeric !== undefined && validateDayOfMonth(numeric) === null) return numeric;
  }

  return undefined;
}

function numberSurfaceForDay(): string {
  return Object.keys(NUMBER_WORDS)
    .sort((left, right) => right.length - left.length)
    .map(regexEscape)
    .join("|");
}

/** Preserve a finite month-end rule rather than allowing "business day" to
 * collapse into the Monday-Friday weekday group. */
function explicitDaySpec(entries: readonly Entry[]): DaySpec | undefined {
  const text = joinEntries(entries);

  if (!/\b(?:monthly|every\s+month|each\s+month|of\s+the\s+month|the\s+month)\b/.test(text)) {
    return undefined;
  }

  for (const [surface, type] of Object.entries(DAY_RULES).sort(
    ([left], [right]) => right.length - left.length,
  )) {
    const phrase = regexEscape(surface).replace(/\ /g, "\\s+");

    if (new RegExp(`\\b${phrase}\\b`).test(text)) return { type };
  }

  return undefined;
}

/** Extract a weekday selector only behind an explicit recurrence/list cue.
 * This intentionally does not turn arbitrary prose containing "Monday" into
 * a schedule; bare short weekday inputs are handled by exactLexicalIntent. */
function explicitWeekdaySelector(entries: readonly Entry[]): Weekday[] | undefined {
  const text = joinEntries(entries);
  const range = explicitWeekdayRange(text, true);

  if (range !== undefined) return range;

  const cueText = normalizeGrammarAliases(
    text,
    { day: "day", days: "day", are: "are", is: "is" },
    2,
  );

  const selectorCue =
    /\b(?:every|each|weekly\s+on|every\s+week\s+on|each\s+week\s+on|day\s+(?:are|is))\b/.test(
      cueText,
    ) ||
    (hasDirectScheduleRequest(cueText) &&
      /(?:\bon\b|^(?:mon|tue|wed|thu|fri|sat|sun)\b|\b(?:fortnightly|weekly)\b|\b(?:schedule|set)\b|\b(?:let|make)\s+[a-z]+\s+(?:be|the)\b)/.test(
        cueText,
      ));

  if (!selectorCue) return undefined;
  const weekdays = weekdaysIn(text, true);

  return weekdays.length === 0 ? undefined : weekdays;
}

/** Preserve a numeric or number-word clock only when the surrounding utterance
 * is explicitly a request. The request requirement prevents an ordinary
 * observation such as "the train arrived at 06:45" from becoming a schedule. */
function explicitRequestedTime(entries: readonly Entry[]): TimeOfDay | undefined {
  const text = joinEntries(entries);

  if (
    !hasDirectScheduleRequest(text) &&
    !/^(?:daily|each\s+day|every\s+day|monthly|weekly)\b/.test(text) &&
    !/^at\b.*\b(?:daily|each\s+day|every\s+day)\b/.test(text)
  ) {
    return undefined;
  }

  let match =
    /\b(?:at|around|everyday|make|(?:alert|notification|reminder)(?:\s+for)?)\s+([a-z]+|\d{1,2})(?![-\s]+(?:past|after|to|before|minutes?|hours?|days?|weeks?|months?)\b)(?::(\d{2}))?\s*(am|pm)?\b/.exec(
      text,
    );

  // A meridiem makes a clock value unambiguous even when it precedes the
  // noun ("run the 9 am reminder") rather than following `at`.
  let hour =
    match === null ? undefined : /^\d+$/.test(match[1]!) ? Number(match[1]) : numberFrom(match[1]!);

  if (match === null || hour === undefined) {
    match = /\b([a-z]+|\d{1,2})(?::(\d{2}))?\s*(am|pm)\b/.exec(text);

    hour =
      match === null
        ? undefined
        : /^\d+$/.test(match[1]!)
          ? Number(match[1])
          : numberFrom(match[1]!);
  }

  if (match === null) return undefined;
  if (hour === undefined) return undefined;
  let minute = match[2] === undefined ? 0 : Number(match[2]);

  if (match[2] === undefined) {
    const continuation = text.slice(match.index! + match[0].length);
    const word = /^\s*([a-z]+)\b/.exec(continuation)?.[1];
    const wordMinute = word === undefined ? undefined : numberFrom(word);

    if (wordMinute !== undefined && wordMinute >= 0 && wordMinute <= 59) minute = wordMinute;
  }

  const meridiem = match[3] as "am" | "pm" | undefined;

  if (!Number.isInteger(hour) || !Number.isInteger(minute) || minute > 59) {
    throw new RangeError("decodeIntent: invalid explicit numeric time");
  }

  let resolved = resolveTime(hour, minute, meridiem);

  if (
    meridiem === undefined &&
    resolved.hour < 12 &&
    /\b(?:in the afternoon|in the evening|at night|tonight|every evening|each evening)\b/.test(text)
  ) {
    resolved = { ...resolved, hour: resolved.hour + 12 };
  }

  if (
    meridiem === undefined &&
    resolved.hour === 12 &&
    /\b(?:in the morning|at night)\b/.test(text)
  ) {
    resolved = { ...resolved, hour: 0 };
  }

  if (resolved.hour > 23) throw new RangeError("decodeIntent: invalid explicit numeric time");

  return resolved;
}

/** Recover a complete relative clock such as "half past eight". Both values
 * are visible and finite, so a mistaken token role must not erase the hour. */
function explicitRelativeTime(entries: readonly Entry[]): TimeOfDay | undefined {
  const text = joinEntries(entries);

  const match =
    /\b(half|quarter|[a-z]+|\d{1,2})(?:\s+minutes?)?\s+(past|after|to|before)\s+([a-z]+|\d{1,2})\b/.exec(
      text,
    );

  if (match === null) return undefined;
  const minute = MINUTE_IDIOMS[match[1]!] ?? numberFrom(match[1]!);
  const hour = NAMED_HOURS[match[3]!] ?? numberFrom(match[3]!);

  if (minute === undefined || hour === undefined || minute > 59 || hour > 23) return undefined;

  const total =
    match[2] === "to" || match[2] === "before"
      ? (hour * 60 - minute + 1440) % 1440
      : (hour * 60 + minute) % 1440;

  let time = { hour: Math.floor(total / 60), minute: total % 60 };

  if (
    time.hour < 12 &&
    /\b(?:in the afternoon|in the evening|at night|tonight|every evening|each evening|every night|each night|pm)\b/.test(
      text,
    )
  ) {
    time = { ...time, hour: time.hour + 12 };
  }

  return time;
}

/** Recover an exact named clock anchor only behind a clock preposition. */
function explicitNamedTime(entries: readonly Entry[]): TimeOfDay | undefined {
  const text = joinEntries(entries);

  if (
    !hasDirectScheduleRequest(text) &&
    !/^(?:at\b|daily\b|each\b|every\b|hourly\b|monthly\b|on\b|weekly\b|weekdays?\b|weekends?\b)/.test(
      text,
    )
  ) {
    return undefined;
  }

  const match = /\b(?:at|for|around|make|use)\s+(midnight|noon|midday)\b/.exec(text);

  if (match === null) return undefined;
  const hour = NAMED_HOURS[match[1]!];

  return hour === undefined ? undefined : { hour, minute: 0 };
}

function dailyCueIntroducesWeekdayRange(text: string): boolean {
  return (
    /\b(?:every|each)\s+day\s+from\b/.test(text) && explicitWeekdayRange(text, true) !== undefined
  );
}

function hasContradictoryDailyCue(text: string): boolean {
  return /\b(?:every|each)\s+day\b/.test(text) && !dailyCueIntroducesWeekdayRange(text);
}

function explicitExceptions(entries: readonly Entry[]): ExplicitExceptionEvidence | undefined {
  const text = joinEntries(entries);

  const alternatives = [...EXCEPTION_PREFIXES]
    .sort((left, right) => right.length - left.length)
    .map((value) => regexEscape(value).replace(/\\ /g, "\\s+"))
    .join("|");

  let cue = new RegExp(`\\b(?:${alternatives})\\b`).exec(text);

  if (cue === null) cue = /\b(?:but\s+)?leave\b(?=.*\balone\b)/.exec(text);
  let cueStart = cue?.index;
  let cueEnd = cue === null ? undefined : cue.index + cue[0].length;

  if (cueStart === undefined || cueEnd === undefined) {
    const singleWordPrefixes = Object.fromEntries(
      EXCEPTION_PREFIXES.filter((prefix) => /^[a-z]+$/.test(prefix)).map((prefix) => [
        prefix,
        prefix,
      ]),
    );

    for (const word of text.matchAll(/[a-z]+/g)) {
      if (uniqueSingleEditAlias(word[0], singleWordPrefixes, 5) === undefined) continue;
      cueStart = word.index!;
      cueEnd = cueStart + word[0].length;
      break;
    }
  }

  if (cueStart === undefined || cueEnd === undefined) return undefined;
  let rawSuffix = text.slice(cueEnd);

  rawSuffix = rawSuffix.replace(/\balone\b.*$/, "");
  const separator = /\b(?:every|each|daily|weekly|monthly|at)\b/.exec(rawSuffix);
  const exceptionText = separator === null ? rawSuffix : rawSuffix.slice(0, separator.index);

  const outsideText =
    text.slice(0, cueStart) + (separator === null ? "" : ` ${rawSuffix.slice(separator.index)}`);

  const weekdays = weekdaysIn(exceptionText, true);
  const dates = [...new Set(calendarDateCandidates(exceptionText).map((date) => date.iso))];

  if (weekdays.length === 0 && dates.length === 0) {
    throw new RangeError("decodeIntent: exception cue without a recognized value");
  }

  return {
    exceptions: {
      ...(weekdays.length === 0 ? {} : { weekdays }),
      ...(dates.length === 0 ? {} : { dates: dates.sort() }),
    },
    selectorWeekdays: weekdaysIn(outsideText, true),
  };
}

function splitClauses(entries: readonly Entry[]): Entry[][] {
  const clauses: Entry[][] = [];
  let current: Entry[] = [];

  for (const entry of entries) {
    if (entry.boundary && current.length > 0) {
      // A boundary marks the clause's first value slot. Structural filler that
      // immediately precedes it (e.g. the `from` of a bound) belongs to the
      // clause it opens, not the one it closes.
      const moved: Entry[] = [];

      while (current.length > 0 && isStructural(current[current.length - 1]!.label)) {
        moved.unshift(current.pop()!);
      }

      if (current.length > 0) clauses.push(current);
      current = moved;
    }

    current.push(entry);
  }

  if (current.length > 0) clauses.push(current);

  return clauses;
}

function groupClause(entries: readonly Entry[]): Group[] {
  const groups: Group[] = [];

  for (const entry of entries) {
    const last = groups[groups.length - 1];

    if (last !== undefined && last.label === entry.label) last.tokens.push(entry.token);
    else groups.push({ label: entry.label, tokens: [entry.token] });
  }

  return groups;
}

/**
 * Read one clause's groups into partial facts. Range roles expand here; bounds
 * and day-of-month DOMs are told apart by an adjacent `BOUND` keyword.
 */
function decodeClause(groups: readonly Group[]): ClauseFacts {
  const facts: ClauseFacts = {
    other: false,
    weekdays: [],
    bounds: {},
    exceptions: {},
  };

  const doms: number[] = [];
  let pending: "start" | "end" | "end-inclusive" | "between" | null = null;
  let excluding = false;

  const addWeekdays = (days: readonly Weekday[]): void => {
    if (excluding) (facts.exceptions.weekdays ??= []).push(...days);
    else facts.weekdays.push(...days);
  };

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index]!;
    const text = joinTokens(group.tokens);

    switch (group.label) {
      case "WEEKDAY": {
        const start = weekdayOf(text);

        if (start === undefined) break;
        const connector = groups[index + 1];
        const endGroup = groups[index + 2];

        const end =
          endGroup?.label === "WEEKDAY" ? weekdayOf(joinTokens(endGroup.tokens)) : undefined;

        if (
          connector?.label === "GLUE" &&
          end !== undefined &&
          isRangeConnector(joinTokens(connector.tokens))
        ) {
          addWeekdays(weekdayRange(start, end));
          index += 2;
        } else {
          addWeekdays([start]);
        }

        break;
      }

      case "DAYGROUP": {
        const members = WEEKDAY_GROUPS[text];

        if (members !== undefined) addWeekdays(members);
        break;
      }

      case "ORDINAL": {
        const ordinal = ORDINAL_ALIASES[text.replace(/\s+(st|nd|rd|th)$/, "$1")];

        if (ordinal !== undefined) facts.ordinal = ordinal;
        break;
      }

      case "DOM": {
        const calendarDate = calendarDateFrom(text);

        if (pending === "start" || pending === "between") {
          if (calendarDate !== undefined) facts.bounds.start = calendarDate;
          pending = pending === "between" ? "end" : null;
        } else if (pending === "end") {
          if (calendarDate !== undefined) facts.bounds.end = calendarDate;
          pending = null;
        } else if (pending === "end-inclusive") {
          const exclusiveEnd =
            calendarDate === undefined ? undefined : offsetIsoDate(calendarDate, 1);

          if (exclusiveEnd !== undefined) facts.bounds.end = exclusiveEnd;
          pending = null;
        } else if (excluding && calendarDate !== undefined) {
          (facts.exceptions.dates ??= []).push(calendarDate);
        } else if (text in DAY_RULES) {
          facts.day = { type: DAY_RULES[text as keyof typeof DAY_RULES] };
        } else if (DAY_ORDINALS.includes(text)) {
          doms.push(DAY_ORDINALS.indexOf(text) + 1);
        } else if (/^\d+$/.test(text)) {
          doms.push(Number(text));
        }

        break;
      }

      case "BOUND": {
        pending = BOUND_ALIASES[text] ?? null;
        excluding = false;
        break;
      }

      case "EXCEPT": {
        excluding = true;
        break;
      }

      case "INTERVAL": {
        const idiom = FREQUENCY_IDIOMS[text];
        const adverb = ADVERB_FREQUENCIES[text];

        if (idiom !== undefined) {
          facts.unit = idiom.unit;
          facts.interval = idiom.interval;
        } else if (adverb !== undefined) {
          facts.unit = adverb;
        } else {
          const value = numberFrom(text);

          if (value !== undefined) facts.interval = value;
        }

        break;
      }

      case "COUNT": {
        const value = numberFrom(text);

        if (value !== undefined && facts.interval === undefined) facts.interval = value;
        break;
      }

      case "HOUR": {
        if (/^\d{4}$/.test(text)) {
          facts.hour = Number(text.slice(0, 2));
          facts.minute = Number(text.slice(2));
        } else {
          const value = NAMED_HOURS[text] ?? numberFrom(text);

          if (value !== undefined) facts.hour = value;
        }

        break;
      }

      case "MINUTE": {
        const value = MINUTE_IDIOMS[text] ?? numberFrom(text);
        const previous = groups[index - 1];
        const previousText = previous?.label === "GLUE" ? joinTokens(previous.tokens) : "";

        // A bare number after "at" is an hour in ordinary English. Recover
        // this unambiguous clock shorthand when the tagger chooses MINUTE;
        // relative clocks ("ten past five") and values above 23 remain
        // minutes and follow the normal validation path.
        if (
          value !== undefined &&
          value <= 23 &&
          facts.hour === undefined &&
          /(?:^|\s)at(?:\s|$)/.test(previousText)
        ) {
          facts.hour = value;
        } else if (value !== undefined) {
          facts.minute = value;
        }

        const connector = groups[index + 1];

        if (connector?.label === "GLUE") {
          const relation = joinTokens(connector.tokens);

          if (
            relation === "past" ||
            relation === "after" ||
            relation.endsWith(" past") ||
            relation.endsWith(" after")
          ) {
            facts.minuteRelation = "past";
          } else if (
            relation === "to" ||
            relation === "before" ||
            relation.endsWith(" to") ||
            relation.endsWith(" before")
          ) {
            facts.minuteRelation = "to";
          }
        }

        break;
      }

      case "MERIDIEM": {
        const meridiem = meridiemOf(text);

        if (meridiem !== undefined) facts.meridiem = meridiem;
        break;
      }

      case "RECUR": {
        let recurrenceUnit: FrequencyUnit | undefined =
          text === WORDS.recur.monthly ? "month" : undefined;

        for (const word of text.split(/\s+/)) {
          const normalized = word
            .toLowerCase()
            .replace(/['’]s$/, "")
            .replace(/[^a-z]/g, "");

          const unit = UNIT_ALIASES[normalized];

          if (unit !== undefined && recurrenceUnit === undefined) recurrenceUnit = unit;
        }

        if (recurrenceUnit !== undefined) facts.unit = recurrenceUnit;
        break;
      }

      case "GLUE": {
        const relativeClock =
          groups[index - 1]?.label === "MINUTE" && groups[index + 1]?.label === "HOUR";

        if (!relativeClock) {
          for (const word of text.split(/[^a-z]+/)) {
            const unit = UNIT_ALIASES[word];

            if (unit !== undefined) facts.unit = unit;
          }
        }

        if (
          text === WORDS.glue.other ||
          text === WORDS.glue.alternate ||
          text === WORDS.glue.alternating
        ) {
          facts.other = true;
        }

        break;
      }

      default:
        break;
    }
  }

  if (doms.length > 0) facts.dayOfMonth = doms[doms.length - 1]!;

  return facts;
}

function resolveTime(
  hour: number | undefined,
  minute: number | undefined,
  meridiem: "am" | "pm" | undefined,
): TimeOfDay {
  let resolved = hour ?? 0;

  if (meridiem === "am") {
    if (resolved < 1 || resolved > 12) throw new RangeError(`invalid 12-hour time: ${resolved}`);
    resolved = resolved === 12 ? 0 : resolved;
  } else if (meridiem === "pm") {
    if (resolved < 1 || resolved > 12) throw new RangeError(`invalid 12-hour time: ${resolved}`);
    resolved = resolved === 12 ? 12 : resolved + 12;
  }

  return { hour: resolved, minute: minute ?? 0 };
}

/**
 * Turn labeled tokens into a validated intent. Throws {@link UnknownRoleError}
 * for an out-of-range role id and `RangeError` for an unrecognized or
 * out-of-bounds schedule.
 */
export function decodeIntent(
  labeled: readonly LabeledToken[],
  scheduleKind?: ScheduleKindEvidence,
  sourceText?: string,
): DecodedIntent {
  const entries: Entry[] = [];

  for (const item of labeled) {
    if (!Number.isInteger(item.role) || item.role < 0 || item.role >= LABEL_COUNT) {
      throw new UnknownRoleError(item.role);
    }

    if (item.token.kind === 3) continue;

    entries.push({
      token: item.token,
      label: labelFromId(item.role),
      boundary: item.boundary,
      score: item.score,
      boundaryScore: item.boundaryScore ?? 1,
    });
  }

  if (entries.length === 0) throw new RangeError("decodeIntent: no tokens");

  const lexical = exactLexicalIntent(entries, sourceText);

  if (lexical !== null) return lexical;

  const trustedScheduleKind =
    scheduleKind !== undefined && scheduleKind.score >= 0.5 ? scheduleKind.kind : undefined;

  const frequencyEvidence = explicitFrequency(entries);

  const decodingEntries =
    frequencyEvidence === undefined
      ? entries
      : entries.map((entry) =>
          entry.token.start >= frequencyEvidence.valueStart &&
          entry.token.end <= frequencyEvidence.valueEnd
            ? { ...entry, label: "INTERVAL" as const }
            : entry,
        );

  let interval: number | undefined;
  let unit: FrequencyUnit | undefined;
  let other = false;
  let ordinal: Ordinal | undefined;
  let dayOfMonth: number | undefined;
  let hour: number | undefined;
  let minute: number | undefined;
  let meridiem: "am" | "pm" | undefined;
  let minuteRelation: "past" | "to" | undefined;
  let day: DaySpec | undefined;
  const weekdays: Weekday[] = [];
  const bounds: RecurrenceBounds = {};
  const exceptions: ScheduleExceptions = {};

  for (const clause of splitClauses(decodingEntries)) {
    const facts = decodeClause(groupClause(clause));

    interval ??= facts.interval;
    unit ??= facts.unit;
    other ||= facts.other;
    weekdays.push(...facts.weekdays);
    ordinal ??= facts.ordinal;
    dayOfMonth ??= facts.dayOfMonth;

    if (bounds.start === undefined && facts.bounds.start !== undefined) {
      bounds.start = facts.bounds.start;
    }

    if (bounds.end === undefined && facts.bounds.end !== undefined) bounds.end = facts.bounds.end;
    hour ??= facts.hour;
    minute ??= facts.minute;
    meridiem ??= facts.meridiem;
    minuteRelation ??= facts.minuteRelation;
    day ??= facts.day;

    if (facts.exceptions.weekdays !== undefined) {
      (exceptions.weekdays ??= []).push(...facts.exceptions.weekdays);
    }

    if (facts.exceptions.dates !== undefined) {
      (exceptions.dates ??= []).push(...facts.exceptions.dates);
    }
  }

  const ordinalEvidence = explicitOrdinalWeekday(entries);
  const text = joinEntries(entries);

  const scheduleScopeText = normalizeGrammarAliases(text, {
    monthly: "monthly",
    month: "monthly",
  });

  const scopedOrdinalEvidence =
    ordinalEvidence !== undefined &&
    !/\b(?:(?:every\s+(?:other|second))|alternat(?:e|ing))\s+(?:mon(?:day)?s?|tue(?:sday)?s?|wed(?:nesday)?s?|thu(?:rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)\b/.test(
      scheduleScopeText,
    ) &&
    (ordinal !== undefined ||
      /\b(?:monthly|every\s+month|each\s+month|remind|schedule)\b/.test(scheduleScopeText))
      ? ordinalEvidence
      : undefined;

  if (scopedOrdinalEvidence !== undefined) {
    ordinal = scopedOrdinalEvidence.ordinal;
    if (weekdays.length === 0) weekdays.push(scopedOrdinalEvidence.weekday);
  }

  const dayOfMonthEvidence = explicitDayOfMonth(entries);

  if (dayOfMonthEvidence !== undefined) dayOfMonth = dayOfMonthEvidence;
  const daySpecEvidence = explicitDaySpec(entries);

  if (daySpecEvidence !== undefined) {
    day = daySpecEvidence;
    dayOfMonth = undefined;
    ordinal = undefined;
    weekdays.length = 0;
  }

  if (frequencyEvidence !== undefined) {
    unit = frequencyEvidence.unit;
    interval = frequencyEvidence.interval;

    if (
      frequencyEvidence.unit !== "month" &&
      dayOfMonthEvidence === undefined &&
      scopedOrdinalEvidence === undefined &&
      daySpecEvidence === undefined
    ) {
      dayOfMonth = undefined;
      ordinal = undefined;
      day = undefined;
    }
  }

  if (
    dayOfMonthEvidence !== undefined &&
    /\b(?:monthly|every\s+month|each\s+month|once\s+per\s+month)\b/.test(text) &&
    !/\bevery\s+(?:\d+|second|third|fourth)\s+months?\b/.test(text)
  ) {
    unit = "month";
    interval = 1;
  }

  const exceptionEvidence = explicitExceptions(entries);

  if (exceptionEvidence?.selectorWeekdays.length) {
    weekdays.length = 0;
    weekdays.push(...exceptionEvidence.selectorWeekdays);
  }

  const weekdaySelectorEvidence = explicitWeekdaySelector(entries);

  if (weekdaySelectorEvidence !== undefined && scopedOrdinalEvidence === undefined) {
    if (dayOfMonthEvidence === undefined && daySpecEvidence === undefined) dayOfMonth = undefined;
    if (daySpecEvidence === undefined) day = undefined;
  }

  if (weekdays.length === 0 && weekdaySelectorEvidence !== undefined) {
    weekdays.push(...weekdaySelectorEvidence);
  }

  const dayGroupEvidence = explicitDayGroup(entries);

  if (weekdays.length === 0 && dayGroupEvidence !== undefined) weekdays.push(...dayGroupEvidence);
  const boundEvidence = explicitBounds(entries);

  if (boundEvidence !== undefined) Object.assign(bounds, boundEvidence);
  const relativeTimeEvidence = explicitRelativeTime(entries);

  if (relativeTimeEvidence !== undefined) {
    hour = relativeTimeEvidence.hour;
    minute = relativeTimeEvidence.minute;
    meridiem = undefined;
    minuteRelation = undefined;
  }

  const requestedTimeEvidence =
    relativeTimeEvidence === undefined ? explicitRequestedTime(entries) : undefined;

  if (requestedTimeEvidence !== undefined) {
    hour = requestedTimeEvidence.hour;
    minute = requestedTimeEvidence.minute;
    meridiem = undefined;
    minuteRelation = undefined;
  }

  const namedTimeEvidence = explicitNamedTime(entries);

  if (namedTimeEvidence !== undefined) {
    hour = namedTimeEvidence.hour;
    minute = namedTimeEvidence.minute;
    meridiem = undefined;
  }

  if (weekdays.length === 0 && trustedScheduleKind === "weekly") {
    const visibleWeekdays = weekdaysIn(joinEntries(entries));

    if (visibleWeekdays.length > 0) weekdays.push(...visibleWeekdays);
  }

  let resolvedUnit = unit;

  if (resolvedUnit === undefined) {
    if (ordinal !== undefined || day !== undefined) {
      resolvedUnit = "month";
    } else if (dayOfMonth !== undefined) {
      resolvedUnit = "month";
    } else if (weekdays.length > 0) {
      resolvedUnit = "week";
    } else if (bounds.start !== undefined || bounds.end !== undefined) {
      resolvedUnit = "day";
    } else if (hour !== undefined || minute !== undefined || meridiem !== undefined) {
      resolvedUnit = "day";
    }
  }

  if (resolvedUnit === undefined) throw new RangeError("decodeIntent: no recognizable schedule");

  const resolvedInterval = other ? 2 : (interval ?? 1);

  const intent: ScheduleIntent = {
    frequency: { unit: resolvedUnit, interval: resolvedInterval },
  };

  if (ordinal !== undefined) {
    const weekday = normalizeWeekdays(weekdays)[0];

    if (weekday === undefined) throw new RangeError("decodeIntent: ordinal without a weekday");
    intent.ordinalWeekday = { ordinal, weekday };
  } else if (weekdays.length > 0) {
    intent.weekdays = normalizeWeekdays(weekdays);
  }

  if (dayOfMonth !== undefined) intent.dayOfMonth = dayOfMonth;
  if (day !== undefined) intent.day = day;
  if (bounds.start !== undefined || bounds.end !== undefined) intent.bounds = bounds;

  if (exceptions.weekdays !== undefined) {
    exceptions.weekdays = normalizeWeekdays(exceptions.weekdays);
  }

  if (exceptions.weekdays !== undefined || exceptions.dates !== undefined) {
    intent.exceptions = exceptions;
  }

  if (hour !== undefined || minute !== undefined || meridiem !== undefined) {
    intent.time = resolveTime(hour, minute, meridiem);

    if (minuteRelation === "to") {
      const total = (intent.time.hour * 60 - intent.time.minute + 1440) % 1440;

      intent.time = { hour: Math.floor(total / 60), minute: total % 60 };
    }
  }

  if (ordinalEvidence !== undefined) {
    if (
      (intent.ordinalWeekday !== undefined &&
        (intent.ordinalWeekday.ordinal !== ordinalEvidence.ordinal ||
          intent.ordinalWeekday.weekday !== ordinalEvidence.weekday)) ||
      (intent.dayOfMonth !== undefined && dayOfMonthEvidence !== undefined) ||
      intent.day !== undefined ||
      (intent.weekdays !== undefined &&
        intent.weekdays.some((weekday) => weekday !== ordinalEvidence.weekday))
    ) {
      throw new RangeError(
        "decodeIntent: predicted selector conflicts with explicit ordinal weekday",
      );
    }

    intent.frequency.unit = "month";
    intent.ordinalWeekday = ordinalEvidence;
    delete intent.weekdays;
    if (dayOfMonthEvidence === undefined) delete intent.dayOfMonth;
  }

  if (dayGroupEvidence !== undefined) {
    // An explicit, separate daily recurrence is genuinely contradictory. A
    // spurious daily unit inferred from the word "weekday" is safe to repair.
    if (hasContradictoryDailyCue(text)) {
      throw new RangeError("decodeIntent: weekdays conflict with daily frequency");
    }

    intent.frequency.unit = "week";
    intent.weekdays = normalizeWeekdays([...(intent.weekdays ?? []), ...dayGroupEvidence]);
  }

  if (daySpecEvidence !== undefined) {
    intent.frequency.unit = "month";
    intent.day = daySpecEvidence;
    delete intent.weekdays;
    delete intent.dayOfMonth;
    delete intent.ordinalWeekday;
  }

  if (boundEvidence !== undefined) intent.bounds = reconcileBounds(intent.bounds, boundEvidence);

  if (exceptionEvidence !== undefined) {
    intent.exceptions = exceptionEvidence.exceptions;

    if (exceptionEvidence.selectorWeekdays.length > 0) {
      intent.frequency.unit = "week";
      intent.weekdays = exceptionEvidence.selectorWeekdays;
    } else {
      delete intent.weekdays;
    }
  } else {
    if (
      weekdaySelectorEvidence !== undefined &&
      intent.ordinalWeekday === undefined &&
      intent.dayOfMonth === undefined &&
      intent.day === undefined
    ) {
      intent.weekdays = weekdaySelectorEvidence;
    }

    // A neural EXCEPT role without any visible exception cue is not evidence
    // that the user requested an exception. Never invent that constraint.
    delete intent.exceptions;
  }

  // In recurrence language, "every second Monday" means alternate Mondays;
  // an ordinal monthly selector needs an explicit monthly context such as
  // "the second Monday of every month". Resolve this visible contrast after
  // model reconciliation so an ORDINAL role cannot silently change cadence.
  if (
    /\b(?:(?:every\s+(?:other|second))|alternat(?:e|ing))\s+(?:mon(?:day)?s?|tue(?:sday)?s?|wed(?:nesday)?s?|thu(?:rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)\b/.test(
      text,
    )
  ) {
    const visibleWeekdays = weekdaysIn(text, true);

    if (visibleWeekdays.length !== 1) {
      throw new RangeError("decodeIntent: ambiguous alternating weekday");
    }

    intent.frequency = { unit: "week", interval: 2 };
    intent.weekdays = visibleWeekdays;
    delete intent.dayOfMonth;
    delete intent.ordinalWeekday;
    delete intent.day;
  }

  // Selector shape is a semantic constraint, not a model preference. Repair a
  // mismatched frequency only when the decoded selector itself is explicit;
  // contradictory independent selectors fail instead of being merged.
  const monthlySelectors =
    Number(intent.dayOfMonth !== undefined) +
    Number(intent.day !== undefined) +
    Number(intent.ordinalWeekday !== undefined);

  if (monthlySelectors > 1 || (monthlySelectors > 0 && intent.weekdays !== undefined)) {
    throw new RangeError("decodeIntent: conflicting calendar selectors");
  }

  if (monthlySelectors > 0) intent.frequency.unit = "month";

  if (
    intent.dayOfMonth !== undefined &&
    /\b(?:monthly|every\s+month|each\s+month|once\s+per\s+month)\b/.test(text) &&
    !/\bevery\s+(?:\d+|second|third|fourth)\s+months?\b/.test(text)
  ) {
    intent.frequency.interval = 1;
  }

  if (intent.weekdays !== undefined && intent.weekdays.length > 0) {
    if (hasContradictoryDailyCue(text)) {
      throw new RangeError("decodeIntent: weekdays conflict with daily frequency");
    }

    intent.frequency.unit = "week";
    if (frequencyEvidence === undefined) intent.frequency.interval = 1;
  }

  const explicitScheduleEvidence =
    (frequencyEvidence !== undefined && frequencyEvidence.contextual !== true) ||
    scopedOrdinalEvidence !== undefined ||
    dayOfMonthEvidence !== undefined ||
    daySpecEvidence !== undefined ||
    weekdaySelectorEvidence !== undefined ||
    dayGroupEvidence !== undefined ||
    boundEvidence !== undefined ||
    exceptionEvidence !== undefined ||
    requestedTimeEvidence !== undefined ||
    namedTimeEvidence !== undefined ||
    (intent.time !== undefined &&
      /\b(?:at|around|noon|midnight|half\s+past|quarter\s+(?:past|to)|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b/.test(
        text,
      ));

  const globalKindConsistent =
    scheduleKind !== undefined &&
    (scheduleKind.kind === "interval" ||
      (scheduleKind.kind === "daily" && intent.frequency.unit === "day") ||
      (scheduleKind.kind === "weekly" && intent.frequency.unit === "week") ||
      (scheduleKind.kind === "monthly" && intent.frequency.unit === "month") ||
      (scheduleKind.kind === "ordinal" && intent.ordinalWeekday !== undefined) ||
      (scheduleKind.kind === "bounded" && intent.bounds !== undefined) ||
      (scheduleKind.kind === "exception" && intent.exceptions !== undefined));

  if (!explicitScheduleEvidence && unrequestedDeclarativeContext(sourceText ?? text)) {
    throw new UnsafeScheduleContextError("descriptive-statement");
  }

  if (trustedScheduleKind !== undefined && !explicitScheduleEvidence) {
    if (trustedScheduleKind === "none" || trustedScheduleKind === "ambiguous") {
      throw new RangeError(`decodeIntent: global schedule kind is ${trustedScheduleKind}`);
    }

    if (!globalKindConsistent) {
      throw new RangeError(
        `decodeIntent: token intent conflicts with global ${trustedScheduleKind} kind`,
      );
    }
  }

  const failure =
    validateInterval(intent.frequency.unit, intent.frequency.interval) ??
    validateTime(intent.time) ??
    validateWeekdays(intent.weekdays) ??
    validateDayOfMonth(intent.dayOfMonth) ??
    validateOrdinalWeekday(intent.ordinalWeekday);

  if (failure !== null) throw new RangeError(failure.reason);

  let tokenMargin = 1;
  let boundaryMargin = 1;

  for (const entry of entries) {
    if (entry.label === "O" || isStructural(entry.label)) continue;
    tokenMargin = Math.min(tokenMargin, entry.score);
    boundaryMargin = Math.min(boundaryMargin, entry.boundaryScore);
  }

  // Preserve the established public behavior while exposing the individual
  // evidence below. A replacement score must be calibrated end to end before
  // it is allowed to reject schedules that the active parser accepts.
  let confidence = tokenMargin;

  if (explicitScheduleEvidence || globalKindConsistent) confidence = 1;

  const evidence: SequenceConfidenceEvidence = {
    tokenMargin,
    boundaryMargin,
    ...(scheduleKind === undefined
      ? {}
      : {
          scheduleKind: scheduleKind.kind,
          scheduleKindMargin: scheduleKind.score,
        }),
    scheduleKindAgreement: globalKindConsistent,
    explicitScheduleEvidence,
    ...(explicitScheduleEvidence ? { lexiconAgreement: true } : {}),
    unconsumedScheduleLikeValues: countUnconsumedScheduleLikeValues(text, intent),
    semanticConstraintViolations: 0,
    compilerConsistent: compilerConsistent(intent),
  };

  return { intent, confidence, evidence };
}
