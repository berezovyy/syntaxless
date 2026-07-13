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
 * The label is the value for every closed class. Token text is read only for
 * `*_LIT` labels, and only via `parseInt` (ISO `YYYY-MM-DD` for `DOM_LIT`).
 * Out-of-range role ids are a hard {@link UnknownRoleError}.
 */
import {
  isLiteralLabel,
  isStructural,
  LABEL_COUNT,
  labelFromId,
  parseLiteralDigits,
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
  Ordinal,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../types/intent.js";
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

/** Diagnostic overlay flag. Public parse is heads-only either way. */
export type OverlayMode = "on" | "off";

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
  hourIs24: boolean;
  day?: DaySpec;
  dayGroup?: DayGroupKind;
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

function normalizeWeekdays(days: readonly Weekday[]): Weekday[] {
  const selected = new Set(days);

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

function applyDomLiteral(
  facts: ClauseFacts,
  text: string,
  pending: BoundKind | null,
  excluding: boolean,
): BoundKind | null {
  const calendarDate = parseDateLiteral(text);
  const day = parseLiteralDigits(text);

  if (pending === "start" || pending === "between") {
    if (calendarDate !== undefined) facts.bounds.start = calendarDate;

    return pending === "between" ? "end-excl" : null;
  }

  if (pending === "end-excl") {
    if (calendarDate !== undefined) facts.bounds.end = calendarDate;

    return null;
  }

  if (pending === "end-incl") {
    const exclusiveEnd =
      calendarDate === undefined ? undefined : exclusiveEndFromInclusiveShown(calendarDate);

    if (exclusiveEnd !== undefined) facts.bounds.end = exclusiveEnd;

    return null;
  }

  if (excluding && calendarDate !== undefined) {
    (facts.exceptions.dates ??= []).push(calendarDate);

    return null;
  }

  if (calendarDate !== undefined) {
    if (facts.bounds.start === undefined) facts.bounds.start = calendarDate;
    else facts.bounds.end = calendarDate;

    return null;
  }

  if (day !== undefined) facts.dayOfMonth = day;

  return pending;
}

/**
 * Read one clause's groups into partial facts. Closed labels are the values;
 * `*_LIT` reads digits (and ISO dates for `DOM_LIT`).
 */
function decodeClause(groups: readonly Group[]): ClauseFacts {
  const facts: ClauseFacts = {
    other: false,
    weekdays: [],
    bounds: {},
    exceptions: {},
    hourIs24: false,
  };

  let pending: BoundKind | null = null;
  let excluding = false;
  let rangeNext = false;

  const addWeekdays = (days: readonly Weekday[]): void => {
    if (excluding) (facts.exceptions.weekdays ??= []).push(...days);
    else facts.weekdays.push(...days);
  };

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index]!;
    const label = group.label;
    const text = joinTokens(group.tokens);
    const closed = valueOf(label);

    if (isWeekdayLabel(label)) {
      const start = closed as Weekday | undefined;

      if (start === undefined) continue;
      const connector = groups[index + 1];
      const endGroup = groups[index + 2];

      const end =
        endGroup !== undefined && isWeekdayLabel(endGroup.label)
          ? (valueOf(endGroup.label) as Weekday | undefined)
          : undefined;

      const ranged =
        rangeNext ||
        (connector !== undefined &&
          isBoundLabel(connector.label) &&
          connector.label !== "BOUND_START");

      if (ranged && end !== undefined) {
        addWeekdays(weekdayRange(start, end));
        index += 2;
        rangeNext = false;
      } else {
        addWeekdays([start]);
        rangeNext = false;
      }

      continue;
    }

    if (label.startsWith("DAYGROUP_")) {
      const kind = closed as DayGroupKind | undefined;

      if (kind !== undefined) {
        facts.dayGroup = kind;
        addWeekdays(DAYGROUP_WEEKDAYS[kind]);
      }

      continue;
    }

    if (label.startsWith("ORDINAL_")) {
      if (closed === "last" || typeof closed === "string") facts.ordinal = closed as Ordinal;
      continue;
    }

    if (label === "DOM_LIT") {
      pending = applyDomLiteral(facts, text, pending, excluding);
      continue;
    }

    if (isBoundLabel(label)) {
      pending = closed as BoundKind;

      rangeNext =
        label === "BOUND_BETWEEN" || label === "BOUND_END_INCL" || label === "BOUND_END_EXCL";

      excluding = false;
      continue;
    }

    if (label === "EXCEPT") {
      excluding = true;
      continue;
    }

    if (label.startsWith("INTERVAL_")) {
      const value = isLiteralLabel(label) ? parseLiteralDigits(text) : (closed as number);

      if (value !== undefined) facts.interval = value;
      continue;
    }

    if (label.startsWith("HOUR_")) {
      if (isLiteralLabel(label)) {
        const digits = parseLiteralDigits(text);

        if (digits === undefined) continue;

        if (text.length >= 3) {
          facts.hour = Number(text.slice(0, text.length - 2));
          facts.minute = Number(text.slice(-2));
          facts.hourIs24 = true;
        } else {
          facts.hour = digits;
          facts.hourIs24 = digits > 12;
        }
      } else if (typeof closed === "number") {
        facts.hour = closed;
        facts.hourIs24 = true;
      }

      continue;
    }

    if (label.startsWith("MINUTE_")) {
      const value = isLiteralLabel(label) ? parseLiteralDigits(text) : (closed as number);

      if (value !== undefined) facts.minute = value;
      continue;
    }

    if (label.startsWith("MERIDIEM_")) {
      if (closed === "am" || closed === "pm") facts.meridiem = closed;
      continue;
    }

    if (label.startsWith("UNIT_")) {
      if (typeof closed === "string") facts.unit = closed as FrequencyUnit;
      continue;
    }
  }

  if (facts.ordinal === "last" && facts.day === undefined) {
    if (facts.dayGroup === "weekday") {
      facts.day = { type: "last_business_day" };
      delete facts.ordinal;
      facts.weekdays = [];
    } else if (facts.weekdays.length === 0) {
      facts.day = { type: "last_day_of_month" };
      delete facts.ordinal;
    }
  }

  return facts;
}

function resolveTime(
  hour: number | undefined,
  minute: number | undefined,
  meridiem: "am" | "pm" | undefined,
  hourIs24: boolean,
): TimeOfDay {
  if (hourIs24 || meridiem === undefined) return { hour: hour ?? 0, minute: minute ?? 0 };

  return { hour: hour24From12(hour ?? 0, meridiem), minute: minute ?? 0 };
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
    if (entry.label === "O" || isStructural(entry.label)) continue;
    hasContent = true;
    if (entry.actionable === true) return true;
  }

  if (hasContent) return false;

  return clause.some((entry) => entry.label !== "O" && entry.actionable === true);
}

/** Keep structural tokens of a clause whose content is actionable. Never drop glue alone. */
function selectActionableEntries(entries: readonly Entry[]): Entry[] {
  if (!entries.some((entry) => entry.actionable !== undefined)) return [...entries];
  const kept: Entry[] = [];

  for (const clause of splitClauses(entries)) {
    if (!clauseIsActionable(clause)) continue;

    for (const entry of clause) {
      if (entry.label === "O") continue;
      if (isStructural(entry.label) || entry.actionable === true) kept.push(entry);
    }
  }

  return kept;
}

/**
 * Turn labeled tokens into a validated intent. Throws {@link UnknownRoleError}
 * for an out-of-range role id and `RangeError` for an unrecognized or
 * out-of-bounds schedule. Public decode is heads-only: speech-act owns refuse,
 * the actionable span owns which tokens count, and facts come from labels plus
 * closed lexicon tables. Overlay-on is not a regex fallback.
 */
export function decodeIntent(
  labeled: readonly LabeledToken[],
  scheduleKind?: ScheduleKindEvidence,
  sourceText?: string,
  overlay: OverlayMode = "off",
  speechAct?: SpeechAct,
): DecodedIntent {
  void overlay;
  void sourceText;

  if (speechAct === undefined) {
    throw new RangeError("decodeIntent: heads-only requires a speech-act");
  }

  if (SPEECH_ACT_REFUSE.has(speechAct)) {
    throw new UnsafeScheduleContextError(speechAct as UnsafeScheduleContextReason);
  }

  const entries = selectActionableEntries(entriesFromLabeled(labeled));

  if (entries.length === 0) throw new UnsafeScheduleContextError("no-actionable-span");

  let interval: number | undefined;
  let unit: FrequencyUnit | undefined;
  let other = false;
  let ordinal: Ordinal | undefined;
  let dayOfMonth: number | undefined;
  let hour: number | undefined;
  let minute: number | undefined;
  let meridiem: "am" | "pm" | undefined;
  let hourIs24 = false;
  let day: DaySpec | undefined;
  const weekdays: Weekday[] = [];
  const bounds: RecurrenceBounds = {};
  const exceptions: ScheduleExceptions = {};

  for (const clause of splitClauses(entries)) {
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
    hourIs24 = hourIs24 || facts.hourIs24;
    day ??= facts.day;

    if (facts.exceptions.weekdays !== undefined) {
      (exceptions.weekdays ??= []).push(...facts.exceptions.weekdays);
    }

    if (facts.exceptions.dates !== undefined) {
      (exceptions.dates ??= []).push(...facts.exceptions.dates);
    }
  }

  const text = joinEntries(entries);

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
    intent.time = resolveTime(hour, minute, meridiem, hourIs24);
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
