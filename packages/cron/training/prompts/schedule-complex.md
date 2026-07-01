---
id: schedule-complex
version: 1
angle: schedule
schemaVersion: 1
---

# schedule-complex

You generate compound English schedules: recurrence bounded by start/end dates,
or recurrence with weekday exceptions. Both are load-bearing and easy to phrase
wrongly, so follow the rules exactly.

## Hard rules (non-negotiable)

1. Respond with exactly one JSON object per line, and nothing else. Each line must
   match this schema: {{schema}}
2. Use only words from the lexicon below for any schedule meaning. Bounds,
   exceptions, and every other selector must come from the lexicon. Carrier prose
   outside the lexicon must not itself sound like a schedule. Lexicon: {{lexicon}}
3. Never invent a schedule the compiler does not support. A bound or exception the
   lexicon cannot express is out of scope; do not approximate it.

## Task

Produce {{count}} examples for the family `{{family}}`. Each example must express
the seed intent exactly, including its bounds or exceptions:

```json
{{seedIntent}}
```

Vary the carrier prose, not the schedule. Do not add explanatory text, markdown,
or code fences.
