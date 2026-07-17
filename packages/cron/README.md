# @syntaxless/cron

Natural-language schedules to a typed intent AST and cron expressions.

`parse(text)` returns a `ScheduleIntent`; `compile(intent)` returns a cron expression or an
explicit unsupported result. The package has no runtime dependencies.

## Intent validation and explanations

`validate(intent)` checks the full intent and throws `IntentError` with a typed
`code` for malformed facts. `compile(intent)` uses the same checks before
reporting unsupported cron semantics. Unknown fields, invalid calendar dates,
conflicting selectors and incomplete supplied clocks are rejected rather than
ignored. Dates use years 0001 to 9999, with inclusive start and exclusive end.
Every frequency unit accepts a positive safe integer interval. For example,
120 minutes is a valid intent for recurrence previews, even though the current
compiler doesn't convert it to an hourly cron expression. An unsupported
compiler result preserves the intent rather than treating the interval as invalid.

`explain(intent)` describes every supplied fact, defaults and cron support,
without loading the model. Calendar schedules default to local 00:00. Hourly
schedules default to minute 0 and accept a `MinuteOfHour`, such as
`{ minute: 15 }`, without inventing a daily hour anchor. Weekly and monthly
schedules without a selector use the recurrence anchor's weekday or day of the
month. The anchor defaults to the caller's reference instant.
The last business day means the last Monday to Friday day of the month, without
holiday exclusions.

Cron expressions use local clock fields starting at zero. Minute intervals must
divide 60, and hour intervals must divide 24. For example, cron compilation
refuses a 7-minute interval because `*/7` has a 4-minute gap at each hour boundary.
The complete field cycles use `0`, so 60 minutes compiles to `0 * * * *` and
24 hours compiles to `0 0 * * *` at the default selected minute.
The target scheduler supplies timezone and daylight-saving behavior. Cron cannot
encode recurrence bounds, exceptions or ordinal weekday rules.

## Clarification for missing cron days

`clarify(intent)` returns a typed question when a valid weekly or monthly intent
only lacks the day needed for cron compilation. A question identifies `weekdays`
with allowed weekday values or `dayOfMonth` with limits 1 to 31. Other intents
return `null`, including unsupported intervals, bounds and ordinal rules.
The API doesn't infer ambiguous clocks or invent alternatives from confidence.

```ts
import { clarify, compile, resolveClarification } from "@syntaxless/cron";
import type { ScheduleIntent } from "@syntaxless/cron";

const intent: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  time: { hour: 9, minute: 25 },
};
const question = clarify(intent);
if (question?.field === "weekdays") {
  // Supply the user's answer rather than choosing a day automatically.
  const resolved = resolveClarification(intent, { weekdays: ["tuesday", "thursday"] });
  const result = compile(resolved); // 25 9 * * 2,4
}
```

`resolveClarification(intent, answer)` accepts only the requested field. It
validates the answer and returns a separate intent without losing existing
facts or sharing nested data. Invalid answers and already complete intents throw
`IntentError`. `cron(text)` includes `clarification` on missing-anchor failures
while preserving the recognized intent. Parsing failures receive no invented
intent or clarification. The host chooses when to ask, apply or recompile.
The clarification functions are synchronous and load no model.

## Recurrence previews

`recurrences(intent, options)` returns a bounded preview and never starts timers
or executes tasks. Provide a timezone, an absolute reference instant, and either
`count` or `before`. The reference and `before` horizon are exclusive. Absolute
instants accept epoch milliseconds or ISO strings with an explicit offset and
up to millisecond precision. The supported Gregorian year range is 0001 to 9999.

```ts
import { recurrences } from "@syntaxless/cron";

const preview = recurrences(
  { frequency: { unit: "month", interval: 1 }, dayOfMonth: 31 },
  {
    timezone: "UTC",
    reference: "2028-01-31T12:00:00Z",
    count: 2,
    policies: { monthEnd: "constrain" },
  },
);
// The preview contains February 29 and March 31 at local midnight.
```

All frequency units follow local calendar fields, rather than fixed elapsed
durations. Optional `anchor` sets the interval phase and defaults to `reference`.
The anchor isn't an implicit start bound. Minute phases use the anchor's local
minute, and hour phases use its local hour with the intent's selected minute.
Day phases use its local date. Weeks begin on Monday, and month phases use its
local month. Missing weekly selectors use the anchor's weekday, while missing
monthly selectors use its day of the month. Cron's zero-based field phase can
differ from a recurrence preview's explicit anchor phase.

Previews implement day-of-month, ordinal weekday, last-day and last-business-day
selectors. Inclusive start dates, exclusive end dates and exceptions apply to
the actual local date after any timezone adjustment. Results are sorted by
instant and contain no duplicate instant. Each occurrence includes requested
and actual local fields, so a clock shifted through a gap remains visible.

The default policies skip nonexistent local times, select the earlier occurrence
of repeated local times, and skip missing days such as February 31. `gap` also
accepts `earlier`, `later` or `reject`. `fold` also accepts `later`, `both` or
`reject`, and `monthEnd` also accepts `constrain` or `reject`. Earlier and later
gap policies shift by the timezone transition's length. Rejection policies
report a failure within the requested window, without returning partial results.

A preview permits at most 1000 occurrences. Candidate work defaults to 10000
steps, and `maxSteps` may raise it to 100000. An unattainable count returns
`work-limit` instead of searching without a bound. A horizon containing more
than 1000 occurrences returns `output-limit`. Successful results identify
whether the count, horizon, intent end or supported year range stopped the
preview. Malformed intents throw `IntentError`, while context and policy failures
return an explicit result.

## Optional timezone adapter

The default calendar supports UTC and fixed offsets such as `+05:45`. IANA
timezones require an explicit adapter. The separate `@syntaxless/cron/temporal`
entry accepts a host's native Temporal or polyfill namespace, without importing
a polyfill or changing a global itself.

```ts
import { Temporal } from "@js-temporal/polyfill";
import { createTemporalCalendar } from "@syntaxless/cron/temporal";

const calendar = createTemporalCalendar(Temporal);
// Supply calendar in the recurrence options alongside an IANA timezone.
```

The adapter uses Temporal's earlier and later resolutions to distinguish ordinary
times, gaps and folds. The host supplies the timezone database. See the
[Temporal ZonedDateTime documentation](https://tc39.es/proposal-temporal/docs/zoneddatetime.html).
The polyfill is a development dependency for package checks, not a runtime
dependency of cron. Custom adapters must return valid Gregorian fields and
offsets strictly below 24 hours, with ordered earlier and later resolutions.

## Remaining cron work

The bundled model still interprets the development case `every weekday at 9`
as 21:00 instead of the declared unqualified 24-hour convention, while explicit
`9am` and `9pm` work. No sentence-specific decoder correction or rules fallback
has been added. Model clock consistency and a fresh full promotion gate are
still required by the implementation plan. Compact training also requires the
original wording for five missing historical acceptance partitions.

## Privacy

Inference runs entirely on-device. Fetching static model bytes is allowed; sending user text,
tokens, intents, or confidence anywhere is not. There is no silent hosted fallback: the bundled
default weights work offline, and a self-hoster supplies bytes through `parse`'s `weights` option
or `preload`/`getEngine` config. Importing the types or the compiler downloads and initializes
nothing; the model module is imported on the first `parse()`, `preload()`, or `getEngine()` call.

Each engine uses its own weights and calibration. Repeated calls with the same
weight buffer share initialization; treat that buffer as immutable and use a new
buffer for another artifact. `dispose()` releases cached engines; an engine
already held by a caller remains usable.

## Weights format

The bundled model artifact carries a `WEIGHTS_FORMAT_VERSION`. A self-hosted artifact whose version
the runtime does not understand is a hard error (`unsupported-weights-version`) raised before any
decode — never a silent mis-decode. The weight bytes live in a separately emitted, lazily loaded
model chunk; the compiler entry does not inline them.
