import { compile } from "../compiler/cron.js";
/**
 * Deterministic decoder: predicted token roles -> validated {@link ScheduleIntent}.
 *
 * The label supplies every closed-class value. Only `*_LIT` roles read token
 * digits (ISO `YYYY-MM-DD` for `DOM_LIT`).
 * Out-of-range role ids are a hard {@link UnknownRoleError}.
 */
import {
  isLiteralLabel,
  isScheduleLabel,
  isStructural,
  LABEL_COUNT,
  labelFromId,
  valueOf,
} from "../model/labels.js";
import type { BoundKind, DayGroupKind, Label } from "../model/labels.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import type { SpeechAct } from "../model/speech-acts.js";
import type { RawToken } from "../model/tokenizer.js";
import {
  exclusiveEndFromInclusiveShown,
  hour24From12,
  parseDateLiteral,
} from "../schedule-semantics.js";
import type {
  DaySpec,
  FrequencyUnit,
  MinuteOfHour,
  Ordinal,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../types/intent.js";
import { intentFailure } from "../validate.js";
import { WEEKDAY_SEQUENCE } from "./lexicon.js";

/** One token with its predicted role, clause boundary, and margin score. */
export interface LabeledToken {
  token: RawToken;
  role: number;
  boundary: boolean;
  score: number;
  /** Optional for oracle callers; runtime inference always supplies it. */
  boundaryScore?: number;
  /** 1 iff this token is in the schedule to execute. */
  actionable?: boolean;
}

/** A validated intent plus the weakest margin of schedule facts and context claims. */
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
  | "non-scheduling-language"
  | "descriptive"
  | "question"
  | "quoted"
  | "one-off"
  | "cancel"
  | "incomplete"
  | "none"
  | "no-actionable-span";

const SPEECH_ACT_REFUSE = new Set<SpeechAct>([
  "descriptive",
  "question",
  "quoted",
  "one-off",
  "cancel",
  "incomplete",
  "none",
]);

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
  actionable?: boolean;
}

interface Group {
  label: Label;
  tokens: RawToken[];
}

type WeekdaySelector = Weekday | DayGroupKind;

interface ScheduleFacts {
  interval?: number;
  unit?: FrequencyUnit;
  weekdaySelectors: WeekdaySelector[];
  ordinal?: Ordinal;
  dayOfMonth?: number;
  bounds: RecurrenceBounds;
  hour24?: number;
  hour12?: number;
  minute?: number;
  meridiem?: "am" | "pm";
  day?: DaySpec;
  exceptions: ScheduleExceptions;
}

function countUnconsumedScheduleLikeValues(text: string, intent: ScheduleIntent): number {
  const representedDates = new Set<string>([
    ...(intent.bounds?.start === undefined ? [] : [intent.bounds.start]),
    ...(intent.bounds?.end === undefined ? [] : [intent.bounds.end]),
    ...(intent.exceptions?.dates ?? []),
  ]);

  let unconsumed = 0;

  for (const match of text.matchAll(/\b\d{4}-\d{2}-\d{2}\b/g)) {
    if (!representedDates.has(match[0])) unconsumed++;
  }

  return unconsumed;
}

function compilerConsistent(intent: ScheduleIntent): boolean {
  const result = compile(intent);

  if (result.supported) return true;

  return ["non-unit-interval", "ordinal-weekday", "day-spec", "bounds", "exceptions"].includes(
    result.code,
  );
}

/** Join a run of tokens the way the renderer emitted them. */
function joinTokens(tokens: readonly RawToken[]): string {
  let text = "";

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    const previous = tokens[index - 1];

    if (previous !== undefined && token.start > previous.end) text += " ";
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

function normalizeWeekdays(days: readonly WeekdaySelector[]): Weekday[] {
  const selected = new Set(days);

  for (const group of ["weekday", "weekend", "day"] as const) {
    if (selected.delete(group)) for (const day of DAYGROUP_WEEKDAYS[group]) selected.add(day);
  }

  return WEEKDAY_SEQUENCE.filter((day) => selected.has(day));
}

function isWeekdayLabel(label: Label): boolean {
  return label.startsWith("WEEKDAY_");
}

function isBoundLabel(label: Label): boolean {
  return label.startsWith("BOUND_");
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

const DAYGROUP_WEEKDAYS: Record<DayGroupKind, Weekday[]> = {
  weekday: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  weekend: ["saturday", "sunday"],
  day: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
};

function readLiteralDigits(text: string, label: Label): number {
  if (!/^\d+$/.test(text)) throw new RangeError(`decodeIntent: invalid ${label} literal`);

  return Number(text);
}

type ClauseOperator = BoundKind | "except-empty" | "except";

/** Selected scalar facts may repeat, but cannot disagree. */
function agreeFact<T extends number | string>(previous: T | undefined, value: T, name: string): T {
  if (previous !== undefined && previous !== value) {
    throw new RangeError(`decodeIntent: conflicting ${name}`);
  }

  return value;
}

function applyDomLiteral(
  facts: ScheduleFacts,
  text: string,
  operator: ClauseOperator | null,
): ClauseOperator | null {
  const calendarDate = parseDateLiteral(text);

  if (calendarDate === undefined) {
    if (operator !== null) {
      throw new RangeError("decodeIntent: bound or exception requires an ISO date");
    }

    facts.dayOfMonth = agreeFact(
      facts.dayOfMonth,
      readLiteralDigits(text, "DOM_LIT"),
      "dayOfMonth",
    );

    return null;
  }

  if (operator === "start" || operator === "between") {
    facts.bounds.start = agreeFact(facts.bounds.start, calendarDate, "start");

    return operator === "between" ? "end-excl" : null;
  }

  if (operator === "end-excl") {
    facts.bounds.end = agreeFact(facts.bounds.end, calendarDate, "end");

    return null;
  }

  if (operator === "end-incl") {
    const exclusiveEnd = exclusiveEndFromInclusiveShown(calendarDate);

    if (exclusiveEnd === undefined) throw new RangeError("decodeIntent: invalid inclusive bound");
    facts.bounds.end = agreeFact(facts.bounds.end, exclusiveEnd, "end");

    return null;
  }

  if (operator === "except-empty" || operator === "except") {
    (facts.exceptions.dates ??= []).push(calendarDate);

    return "except";
  }

  if (facts.bounds.start === undefined) facts.bounds.start = calendarDate;
  else facts.bounds.end = agreeFact(facts.bounds.end, calendarDate, "end");

  return null;
}

/**
 * Read one clause's groups into partial facts. Closed labels are the values;
 * `*_LIT` reads digits (and ISO dates for `DOM_LIT`).
 */
function decodeClause(groups: readonly Group[]): ScheduleFacts {
  const facts: ScheduleFacts = {
    weekdaySelectors: [],
    bounds: {},
    exceptions: {},
  };

  // Bounds end after their operands; exceptions retain ownership of their list.
  let operator: ClauseOperator | null = null;
  let hourForm: "absolute" | "literal" | undefined;

  const addWeekdays = (days: readonly WeekdaySelector[]): void => {
    if (operator === "except-empty" || operator === "except") {
      (facts.exceptions.weekdays ??= []).push(...normalizeWeekdays(days));
      operator = "except";
    } else {
      facts.weekdaySelectors.push(...days);
    }
  };

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index]!;
    const label = group.label;
    const closed = valueOf(label);

    if (isBoundLabel(label) || label === "EXCEPT") {
      if (operator !== null && operator !== "except") {
        throw new RangeError("decodeIntent: incomplete schedule operator");
      }

      operator = label === "EXCEPT" ? "except-empty" : (closed as BoundKind);
      continue;
    }

    if (isWeekdayLabel(label)) {
      const start = closed as Weekday | undefined;

      if (start === undefined) continue;
      const connector = groups[index + 1];
      const endGroup = groups[index + 2];

      const end =
        endGroup !== undefined && isWeekdayLabel(endGroup.label)
          ? (valueOf(endGroup.label) as Weekday | undefined)
          : undefined;

      const rangeConnector =
        connector !== undefined &&
        isBoundLabel(connector.label) &&
        connector.label !== "BOUND_START";

      const bound = operator !== null && operator !== "except-empty" && operator !== "except";
      const ranged = rangeConnector || (bound && operator !== "start");

      if (ranged) {
        if (end === undefined || (!rangeConnector && connector?.label !== "GLUE")) {
          throw new RangeError("decodeIntent: incomplete weekday range");
        }

        addWeekdays(weekdayRange(start, end));
        index += 2;
        if (bound) operator = null;
      } else {
        if (bound) throw new RangeError("decodeIntent: incomplete schedule operator");
        addWeekdays([start]);
      }

      continue;
    }

    if (label.startsWith("DAYGROUP_")) {
      if (operator !== null && operator !== "except-empty" && operator !== "except") {
        throw new RangeError("decodeIntent: incomplete schedule operator");
      }

      const kind = closed as DayGroupKind | undefined;

      if (kind !== undefined) addWeekdays([kind]);
      continue;
    }

    if (label.startsWith("ORDINAL_")) {
      if (typeof closed !== "string") throw new RangeError("decodeIntent: unsupported ordinal");
      facts.ordinal = agreeFact(facts.ordinal, closed as Ordinal, "ordinal");
      continue;
    }

    if (label === "DOM_LIT") {
      operator = applyDomLiteral(facts, joinTokens(group.tokens), operator);
      continue;
    }

    if (label.startsWith("INTERVAL_")) {
      facts.interval = agreeFact(
        facts.interval,
        isLiteralLabel(label)
          ? readLiteralDigits(joinTokens(group.tokens), label)
          : (closed as number),
        "interval",
      );

      continue;
    }

    if (label.startsWith("HOUR_")) {
      if (isLiteralLabel(label)) {
        const text = joinTokens(group.tokens);
        const digits = readLiteralDigits(text, label);

        if (text.length >= 3) {
          hourForm = "absolute";
          facts.hour24 = agreeFact(facts.hour24, Number(text.slice(0, text.length - 2)), "hour24");
          facts.minute = agreeFact(facts.minute, Number(text.slice(-2)), "minute");
        } else {
          hourForm = digits > 12 ? "absolute" : "literal";
          if (digits > 12) facts.hour24 = agreeFact(facts.hour24, digits, "hour24");
          else facts.hour12 = agreeFact(facts.hour12, digits, "hour12");
        }
      } else if (typeof closed === "number") {
        hourForm = "absolute";
        facts.hour24 = agreeFact(facts.hour24, closed, "hour24");
      }

      continue;
    }

    if (label.startsWith("MINUTE_")) {
      facts.minute = agreeFact(
        facts.minute,
        isLiteralLabel(label)
          ? readLiteralDigits(joinTokens(group.tokens), label)
          : (closed as number),
        "minute",
      );

      continue;
    }

    if (label.startsWith("MERIDIEM_")) {
      // Closed hours already include clock arithmetic, including midnight carry.
      if (hourForm !== "absolute" && (closed === "am" || closed === "pm")) {
        facts.meridiem = agreeFact(facts.meridiem, closed, "meridiem");
      }

      continue;
    }

    if (label.startsWith("UNIT_")) {
      if (typeof closed === "string") {
        facts.unit = agreeFact(facts.unit, closed as FrequencyUnit, "unit");
      }

      continue;
    }
  }

  if (operator !== null && operator !== "except") {
    throw new RangeError("decodeIntent: incomplete schedule operator");
  }

  if (facts.hour24 !== undefined && facts.hour12 === undefined) delete facts.meridiem;

  if (facts.ordinal === "last") {
    const group = facts.weekdaySelectors[0] ?? "day";

    if (
      (group === "weekday" || group === "day") &&
      facts.weekdaySelectors.every((selector) => selector === group)
    ) {
      facts.day = {
        type: group === "weekday" ? "last_business_day" : "last_day_of_month",
      };

      delete facts.ordinal;
      facts.weekdaySelectors = [];
    }
  }

  return facts;
}

function resolveTime(facts: ScheduleFacts, unit: FrequencyUnit): TimeOfDay | MinuteOfHour {
  const { hour24, hour12, minute, meridiem } = facts;

  if (unit === "hour" && hour24 === undefined && hour12 === undefined && meridiem === undefined) {
    return { minute: minute ?? 0 };
  }

  const literalHour =
    hour12 === undefined
      ? undefined
      : meridiem === undefined
        ? hour12
        : hour24From12(hour12, meridiem);

  if (hour24 !== undefined && literalHour !== undefined && hour24 !== literalHour) {
    throw new RangeError("decodeIntent: conflicting hour");
  }

  const hour = hour24 ?? literalHour ?? (meridiem === undefined ? 0 : hour24From12(0, meridiem));

  return { hour, minute: minute ?? 0 };
}

function entriesFromLabeled(labeled: readonly LabeledToken[]): Entry[] {
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
      ...(item.actionable === undefined ? {} : { actionable: item.actionable }),
    });
  }

  return entries;
}

function clauseIsActionable(clause: readonly Entry[]): boolean {
  let hasContent = false;

  for (const entry of clause) {
    if (!isScheduleLabel(entry.label) || isStructural(entry.label)) continue;
    hasContent = true;
    if (entry.actionable === true) return true;
  }

  if (hasContent) return false;

  return clause.some((entry) => isScheduleLabel(entry.label) && entry.actionable === true);
}

/**
 * Select executable content and retain quantity evidence before filtering.
 * Missing actionability metadata keeps every entry; otherwise structural tokens
 * follow active content. Discarded clauses contribute neither facts nor quantities.
 */
function selectActionableEntries(entries: readonly Entry[]): {
  entries: Entry[];
  unresolvedCount: boolean;
  unknownDigit: boolean;
} {
  const hasActionability = entries.some((entry) => entry.actionable !== undefined);
  const kept: Entry[] = [];
  let unresolvedCount = false;
  let unknownDigit = false;

  for (const clause of splitClauses(entries)) {
    if (hasActionability && !clauseIsActionable(clause)) continue;

    for (const entry of clause) {
      unresolvedCount ||= entry.label === "COUNT";
      unknownDigit ||= entry.token.kind === 1 && entry.label === "O";

      if (
        !hasActionability ||
        (isScheduleLabel(entry.label) && (isStructural(entry.label) || entry.actionable === true))
      ) {
        kept.push(entry);
      }
    }
  }

  return { entries: kept, unresolvedCount, unknownDigit };
}

/**
 * Turn labeled tokens into a validated intent. Throws {@link UnknownRoleError}
 * for an out-of-range role id and `RangeError` for an unrecognized or
 * out-of-bounds schedule. The speech-act decides whether to refuse, the
 * actionable span decides which tokens count, and facts come from labels plus
 * closed lexicon tables.
 *
 * `_sourceText` is unused; it keeps the positional order of existing callers.
 */
export function decodeIntent(
  labeled: readonly LabeledToken[],
  scheduleKind?: ScheduleKindEvidence,
  _sourceText?: string,
  speechAct?: SpeechAct,
): DecodedIntent {
  if (speechAct === undefined) {
    throw new RangeError("decodeIntent: heads-only requires a speech-act");
  }

  if (SPEECH_ACT_REFUSE.has(speechAct)) {
    throw new UnsafeScheduleContextError(speechAct as UnsafeScheduleContextReason);
  }

  const labeledEntries = entriesFromLabeled(labeled);
  let contextMargin = 1;

  for (const entry of labeledEntries) {
    if (entry.label !== "CONTEXT_NUMBER") continue;

    if (entry.token.kind !== 1 || entry.actionable === true || entry.boundary) {
      throw new RangeError("decodeIntent: invalid context-number claim");
    }

    contextMargin = Math.min(contextMargin, entry.score);
  }

  const { entries, unresolvedCount, unknownDigit } = selectActionableEntries(labeledEntries);

  if (entries.length === 0) throw new UnsafeScheduleContextError("no-actionable-span");

  const facts: ScheduleFacts = { weekdaySelectors: [], bounds: {}, exceptions: {} };

  for (const clause of splitClauses(entries)) {
    const current = decodeClause(groupClause(clause));

    if (current.interval !== undefined) {
      facts.interval = agreeFact(facts.interval, current.interval, "interval");
    }

    if (current.unit !== undefined) facts.unit = agreeFact(facts.unit, current.unit, "unit");

    if (current.ordinal !== undefined) {
      facts.ordinal = agreeFact(facts.ordinal, current.ordinal, "ordinal");
    }

    if (current.dayOfMonth !== undefined) {
      facts.dayOfMonth = agreeFact(facts.dayOfMonth, current.dayOfMonth, "dayOfMonth");
    }

    if (current.bounds.start !== undefined) {
      facts.bounds.start = agreeFact(facts.bounds.start, current.bounds.start, "start");
    }

    if (current.bounds.end !== undefined) {
      facts.bounds.end = agreeFact(facts.bounds.end, current.bounds.end, "end");
    }

    if (current.hour24 !== undefined) {
      facts.hour24 = agreeFact(facts.hour24, current.hour24, "hour24");
    }

    if (current.hour12 !== undefined) {
      facts.hour12 = agreeFact(facts.hour12, current.hour12, "hour12");
    }

    if (current.minute !== undefined) {
      facts.minute = agreeFact(facts.minute, current.minute, "minute");
    }

    if (current.meridiem !== undefined) {
      facts.meridiem = agreeFact(facts.meridiem, current.meridiem, "meridiem");
    }

    if (current.day !== undefined) {
      if (facts.day !== undefined && facts.day.type !== current.day.type) {
        throw new RangeError("decodeIntent: conflicting day");
      }

      facts.day = current.day;
    }

    facts.weekdaySelectors.push(...current.weekdaySelectors);

    if (current.exceptions.weekdays !== undefined) {
      (facts.exceptions.weekdays ??= []).push(...current.exceptions.weekdays);
    }

    if (current.exceptions.dates !== undefined) {
      (facts.exceptions.dates ??= []).push(...current.exceptions.dates);
    }
  }

  const {
    interval,
    unit,
    ordinal,
    dayOfMonth,
    hour24,
    hour12,
    minute,
    meridiem,
    day,
    weekdaySelectors,
    bounds,
    exceptions,
  } = facts;

  const text = joinEntries(entries);

  let resolvedUnit = unit;

  if (resolvedUnit === undefined) {
    if (ordinal !== undefined || day !== undefined) {
      resolvedUnit = "month";
    } else if (dayOfMonth !== undefined) {
      resolvedUnit = "month";
    } else if (weekdaySelectors.length > 0) {
      resolvedUnit = "week";
    } else if (bounds.start !== undefined || bounds.end !== undefined) {
      resolvedUnit = "day";
    } else if (
      hour24 !== undefined ||
      hour12 !== undefined ||
      minute !== undefined ||
      meridiem !== undefined
    ) {
      resolvedUnit = "day";
    }
  }

  if (resolvedUnit === undefined) throw new RangeError("decodeIntent: no recognizable schedule");

  if (unresolvedCount || (unknownDigit && interval === undefined)) {
    throw new RangeError("decodeIntent: uncounted quantity");
  }

  const intent: ScheduleIntent = {
    frequency: { unit: resolvedUnit, interval: interval ?? 1 },
  };

  if (ordinal !== undefined) {
    const weekdays = normalizeWeekdays(weekdaySelectors);

    if (weekdays.length !== 1) {
      throw new RangeError("decodeIntent: ordinal requires exactly one weekday");
    }

    intent.ordinalWeekday = { ordinal, weekday: weekdays[0]! };
  } else if (weekdaySelectors.length > 0) {
    intent.weekdays = normalizeWeekdays(weekdaySelectors);
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

  if (
    hour24 !== undefined ||
    hour12 !== undefined ||
    minute !== undefined ||
    meridiem !== undefined
  ) {
    intent.time = resolveTime(facts, resolvedUnit);
  }

  const monthlySelectors =
    Number(intent.dayOfMonth !== undefined) +
    Number(intent.day !== undefined) +
    Number(intent.ordinalWeekday !== undefined);

  if (monthlySelectors > 1 || (monthlySelectors > 0 && intent.weekdays !== undefined)) {
    throw new RangeError("decodeIntent: conflicting calendar selectors");
  }

  if (monthlySelectors > 0) intent.frequency.unit = "month";
  if (intent.weekdays !== undefined && intent.weekdays.length > 0) intent.frequency.unit = "week";

  const labeledEvidence =
    intent.weekdays !== undefined ||
    intent.ordinalWeekday !== undefined ||
    intent.dayOfMonth !== undefined ||
    intent.day !== undefined ||
    intent.bounds !== undefined ||
    intent.exceptions !== undefined ||
    intent.time !== undefined ||
    interval !== undefined ||
    unit !== undefined;

  const globalKindConsistent =
    scheduleKind !== undefined &&
    (scheduleKind.kind === "interval" ||
      (scheduleKind.kind === "daily" && intent.frequency.unit === "day") ||
      (scheduleKind.kind === "weekly" && intent.frequency.unit === "week") ||
      (scheduleKind.kind === "monthly" && intent.frequency.unit === "month") ||
      (scheduleKind.kind === "ordinal" && intent.ordinalWeekday !== undefined) ||
      (scheduleKind.kind === "bounded" && intent.bounds !== undefined) ||
      (scheduleKind.kind === "exception" && intent.exceptions !== undefined));

  const failure = intentFailure(intent);

  if (failure !== null) throw new RangeError(failure.reason);

  let tokenMargin = contextMargin;
  let boundaryMargin = 1;

  for (const entry of entries) {
    if (!isScheduleLabel(entry.label) || isStructural(entry.label)) continue;
    tokenMargin = Math.min(tokenMargin, entry.score);
    boundaryMargin = Math.min(boundaryMargin, entry.boundaryScore);
  }

  const confidence = tokenMargin;

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
    explicitScheduleEvidence: labeledEvidence,
    ...(labeledEvidence ? { lexiconAgreement: true } : {}),
    unconsumedScheduleLikeValues: countUnconsumedScheduleLikeValues(text, intent),
    semanticConstraintViolations: 0,
    compilerConsistent: compilerConsistent(intent),
  };

  return { intent, confidence, evidence };
}
