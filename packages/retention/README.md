# @syntaxless/retention

Convert English backup retention requests to typed policies and deterministic
keep/remove previews. The local model identifies policy roles, and TypeScript
preserves exact counts, durations and calendar units. The package never deletes
files or cloud objects, and no backup metadata leaves the device.

## Usage

```ts
import { compile, explain, parse } from "@syntaxless/retention";

const parsed = await parse("keep daily backups for a week and monthly backups for a year");
if (parsed.ok) {
  console.log(explain(parsed.intent));
  const preview = compile(parsed.intent, {
    timeZone: "UTC",
    referenceInstant: Date.parse("2024-03-15T12:00:00Z"),
    backups: [
      { id: "current", timestamp: Date.parse("2024-03-15T00:00:00Z") },
      { id: "old", timestamp: Date.parse("2022-01-01T00:00:00Z") },
    ],
  });
  // When supported, keep is [{ id: "current", tierIndexes: [0, 1] }].
  // remove is ["old"]. Neither list executes a removal.
}
```

Parsing doesn't require backup metadata or calendar context. Every compilation
requires an explicit supported timezone and reference instant, so missing context
doesn't invalidate a parsed policy. `compile` returns `supported: false` with a
specific code and reason for invalid context, policy or metadata. `validate` and
`explain` throw for invalid edited policies. There is no rules fallback.

## Calendar and elapsed windows

A policy has one to four tiers, with at most one tier per day, week, month or year
bucket. Each tier selects the newest eligible backup in each bucket. Calendar
windows include the period containing the reference instant. Daily retention
for a week counts seven local dates, and monthly retention for a year counts
twelve local months. Weekly buckets begin on Monday, and years begin on January 1.

Calendar windows retain their named unit. A weekly window in a weekly tier counts
Monday weeks, and a yearly window in a yearly tier counts calendar years. Other
weekly windows count seven daily periods per week, and other yearly windows
count twelve monthly periods per year. Daily or weekly retention for a calendar
month uses the caller's Gregorian calendar, without a fixed day approximation.
A window boundary can clip the earliest selected bucket.

Elapsed windows compare exact timestamps and include the lower cutoff.
Milliseconds, seconds, minutes and hours mean fixed elapsed durations without
a rolling marker. Days and weeks mean calendar periods unless "last", "past",
"rolling" or "elapsed" requests elapsed time. Elapsed months/years and explicit
calendar clock durations refuse. Decimal durations must resolve to positive
safe whole milliseconds, and calendar counts range from one to 10000.

## Preview and metadata

Overlapping selections preserve every zero-based `tierIndexes` value.
An empty bucket contributes no backup. Timestamp ties use ascending Unicode
scalar identifier order, and both output lists preserve input order.
The caller reviews and executes any removal recommendations separately.

A preview accepts at most 100000 backups. Each backup has only `id` and
`timestamp`, and identifiers must be unique nonempty Unicode strings of at most
256 UTF-16 code units. NUL and lone surrogates refuse. Timestamps are whole
milliseconds from the Unix epoch through Gregorian year 9999, and future
timestamps refuse. The caller's local calendar must also remain within years
one to 9999. Invalid metadata refuses the complete preview.

The parser accepts text up to 4096 UTF-16 code units and 64 lexemes. Numeric
literals and word numbers through one hundred are supported. Explicit requests
for multiple representatives per bucket, duplicate bucket tiers, questions,
uncertain requests and descriptions of past policies refuse.

`preload` initializes the same model as parsing. Supplied UTF-8 JSON `weights`
bytes own isolated engines and calibration, and initialization copies caller
bytes. `dispose` clears caches. Timezone behavior uses the runtime's calendar data.

## Verification

The promoted model uses 8000 rows and hidden width 32. Synthetic heldout matched
600/600 positive intents and refused all 302 negatives. Fresh acceptance matched
761/761 positive intents across ten families and refused all 637 negatives.
All prior and fresh critical controls passed. Independent UTC previews,
hand-checked DST witnesses, 32-sequence exported-wire parity, actual packed
consumers and real Chromium lazy/concurrent/offline checks passed.

The aggregate runtime is 8995 Brotli bytes, and the model is 23303 Brotli bytes.
The complete report is
`training/retention/runs/history32-2026-10-11T22-16-03-348Z/gate-report.json`.
Reserved constructions were AI-authored by the same implementation assistant.
The separate evaluation writer and independent calculations don't establish
independent human evidence or population accuracy.
