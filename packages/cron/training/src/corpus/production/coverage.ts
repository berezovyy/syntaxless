import { validateInterval } from "../../../../src/compiler/validate";
import { valueOf } from "../../../../src/model/labels";
/** Coverage evidence is distinct from row integrity and from model acceptance. */
import specification from "../../../data/production/coverage-v1.json";
import type { CorpusRecord } from "../merge";
import { SPLITS } from "../merge";

type Split = CorpusRecord["split"];

interface Cell {
  records: number;
  groups: Set<string>;
}

const cell = (): Cell => ({ records: 0, groups: new Set() });

interface SplitCoverage {
  semantics: Map<string, Cell>;
  language: Map<string, Cell>;
  negatives: Map<string, Cell>;
  interactions: Map<string, Cell>;
  values: Map<string, Set<string | number>>;
}

const empty = (): SplitCoverage => ({
  semantics: new Map(),
  language: new Map(),
  negatives: new Map(),
  interactions: new Map(),
  values: new Map(),
});

function increment(cells: Map<string, Cell>, key: string, group: string): void {
  const entry = cells.get(key) ?? cell();

  entry.records++;
  entry.groups.add(group);
  cells.set(key, entry);
}

function serial(cells: Map<string, Cell>) {
  return Object.fromEntries(
    [...cells.entries()]
      .sort(([a], [b]) => a.localeCompare(b, "en"))
      .map(([key, value]) => [key, { records: value.records, groups: value.groups.size }]),
  );
}

export class ConstructionCoverage {
  readonly #splits: Record<Split, SplitCoverage> = {
    train: empty(),
    validation: empty(),
    heldout: empty(),
  };

  add(row: CorpusRecord, semanticFamily: string, declaredTags: readonly string[]): void {
    if (!row.intent || !row.provenance) return;
    const current = this.#splits[row.split];
    const group = `${row.provenance.sourceId}/${row.provenance.groupId}`;
    const request = row.speechAct === "request" || row.speechAct === "correction";

    if (!request) {
      for (const tag of new Set(declaredTags)) {
        if (specification.hardNegativeConstructions.includes(tag)) {
          increment(current.negatives, tag, group);
        }
      }

      return;
    }

    increment(current.semantics, semanticFamily, group);

    for (const tag of new Set(declaredTags)) {
      if (specification.languageConstructions.includes(tag)) {
        increment(current.language, tag, group);
      }
    }

    const intent = row.intent;

    const value = (key: string, item: string | number): void => {
      const values = current.values.get(key) ?? new Set();

      values.add(item);
      current.values.set(key, values);
    };

    value(`interval/${intent.frequency.unit}`, intent.frequency.interval);

    if (intent.time) {
      value("hour", intent.time.hour);
      value("minute", intent.time.minute);

      if (intent.weekdays?.length === 1) {
        increment(current.interactions, "weekday-and-clock", group);
      }

      if ((intent.weekdays?.length ?? 0) > 1) {
        increment(current.interactions, "weekday-list-and-clock", group);
      }

      if (semanticFamily === "weekday-range") {
        increment(current.interactions, "weekday-range-and-clock", group);
      }

      if (intent.dayOfMonth) increment(current.interactions, "monthly-date-and-clock", group);

      if (intent.ordinalWeekday) {
        increment(current.interactions, "ordinal-weekday-and-clock", group);
      }

      if (intent.day) increment(current.interactions, "day-rule-and-clock", group);

      if (
        intent.frequency.interval > 1 &&
        ["day", "week", "month"].includes(intent.frequency.unit)
      ) {
        increment(current.interactions, "interval-and-clock-where-valid", group);
      }

      if (intent.bounds) increment(current.interactions, "bounds-and-clock", group);
      if (intent.exceptions) increment(current.interactions, "exception-and-clock", group);
    }

    if (intent.weekdays && intent.exceptions) {
      increment(current.interactions, "selector-and-exception", group);
    }

    if (
      row.speechAct === "correction" &&
      row.spans.some((s) => s.label !== "O" && !s.actionable) &&
      row.spans.some((s) => s.actionable)
    ) {
      increment(current.interactions, "old-and-new-schedule-with-actionability", group);
    }

    for (const day of intent.weekdays ?? []) value("weekday", day);

    if (semanticFamily === "weekday-range") {
      const endpoints = row.spans.filter(
        (span) => span.actionable && span.label.startsWith("WEEKDAY_"),
      );

      if (
        endpoints.length === 2 &&
        row.spans.some(
          (span) =>
            span.actionable &&
            span.start >= endpoints[0]!.end &&
            span.end <= endpoints[1]!.start &&
            ["BOUND_BETWEEN", "BOUND_END_INCL", "BOUND_END_EXCL"].includes(span.label),
        )
      ) {
        const first = valueOf(endpoints[0]!.label);
        const last = valueOf(endpoints[1]!.label);

        const days = specification.valueCoverage.weekdays;

        const start = days.indexOf(String(first));
        const end = days.indexOf(String(last));

        const expected =
          start >= 0 && end >= 0
            ? Array.from({ length: ((end - start + 7) % 7) + 1 }, (_, n) => days[(start + n) % 7]!)
            : [];

        if (
          first !== last &&
          expected.length === intent.weekdays?.length &&
          expected.every((day) =>
            intent.weekdays!.includes(day as NonNullable<typeof intent.weekdays>[number]),
          )
        ) {
          value(`weekday-range/${intent.time ? "timed" : "untimed"}`, `${first}/${last}`);
        }
      }
    }

    if (intent.dayOfMonth) value("day-of-month", intent.dayOfMonth);

    if (intent.ordinalWeekday) {
      value("ordinal", intent.ordinalWeekday.ordinal);
      value("ordinal-weekday", intent.ordinalWeekday.weekday);
    }

    for (const tag of declaredTags) {
      if (specification.valueCoverage.calendarCases.includes(tag)) value("calendar", tag);
    }
  }

  finish() {
    const missing: Array<{ split: Split; axis: string; key: string; reason: string }> = [];
    const evidence: Record<string, unknown> = {};

    for (const split of SPLITS) {
      const current = this.#splits[split];

      const check = (
        axis: "semantics" | "language" | "negatives" | "interactions",
        keys: readonly string[],
        minGroups = 1,
      ) => {
        for (const key of keys) {
          const found = current[axis].get(key);

          if (!found || found.records === 0 || found.groups.size < minGroups) {
            missing.push({
              split,
              axis,
              key,
              reason: `requires nonempty evidence from at least ${minGroups} source group(s)`,
            });
          }
        }
      };

      // Every semantic family needs more than a single renamed construction.
      check("semantics", specification.semanticFamilies, 2);
      check("language", specification.languageConstructions);
      check("negatives", specification.hardNegativeConstructions);
      check("interactions", specification.requiredInteractions);

      const requireValue = (key: string, expected: string | number) => {
        if (!current.values.get(key)?.has(expected)) {
          missing.push({
            split,
            axis: "values",
            key: `${key}/${expected}`,
            reason: "no positive gold evidence",
          });
        }
      };

      for (
        let h = specification.valueCoverage.hours.min;
        h <= specification.valueCoverage.hours.max;
        h++
      ) {
        requireValue("hour", h);
      }

      for (
        let m = specification.valueCoverage.minutes.min;
        m <= specification.valueCoverage.minutes.max;
        m++
      ) {
        requireValue("minute", m);
      }

      for (
        let d = specification.valueCoverage.daysOfMonth.min;
        d <= specification.valueCoverage.daysOfMonth.max;
        d++
      ) {
        requireValue("day-of-month", d);
      }

      for (const day of specification.valueCoverage.weekdays) {
        requireValue("weekday", day);
        requireValue("ordinal-weekday", day);
      }

      if (specification.valueCoverage.weekdayRanges.orderedDistinctEndpoints) {
        for (const first of specification.valueCoverage.weekdays) {
          for (const last of specification.valueCoverage.weekdays) {
            if (first !== last) {
              for (const timing of specification.valueCoverage.weekdayRanges.timing) {
                requireValue(`weekday-range/${timing}`, `${first}/${last}`);
              }
            }
          }
        }
      }

      for (const ordinal of specification.valueCoverage.ordinals) requireValue("ordinal", ordinal);

      for (const calendar of specification.valueCoverage.calendarCases) {
        requireValue("calendar", calendar);
      }

      const intervals = new Set([
        ...Array.from(
          {
            length:
              specification.valueCoverage.intervals.denseClosedRange.max -
              specification.valueCoverage.intervals.denseClosedRange.min +
              1,
          },
          (_, i) => i + specification.valueCoverage.intervals.denseClosedRange.min,
        ),
        ...specification.valueCoverage.intervals.additionalDigitValues,
      ]);

      for (const unit of ["minute", "hour", "day", "week", "month"]) {
        for (const n of intervals) {
          if (validateInterval(unit, n) === null) requireValue(`interval/${unit}`, n);
        }
      }

      evidence[split] = {
        semantics: serial(current.semantics),
        language: serial(current.language),
        negatives: serial(current.negatives),
        interactions: serial(current.interactions),
        values: Object.fromEntries(
          [...current.values].map(([key, values]) => [
            key,
            [...values].sort((a, b) => String(a).localeCompare(String(b), "en", { numeric: true })),
          ]),
        ),
      };
    }

    return {
      version: 1,
      scope: "declared-construction-and-gold-value-coverage-not-model-acceptance",
      passed: missing.length === 0,
      missing,
      splits: evidence,
    };
  }
}
