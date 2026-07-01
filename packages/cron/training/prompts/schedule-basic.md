---
id: schedule-basic
version: 1
angle: schedule
schemaVersion: 1
---

# schedule-basic

You generate short English sentences that describe a recurring schedule, together
with the schedule's intent. The deterministic compiler is the semantic authority;
you only propose text.

## Hard rules (non-negotiable)

1. Respond with exactly one JSON object per line, and nothing else. Each line must
   match this schema: {{schema}}
2. Use only words from the lexicon below for any schedule meaning. Words outside
   the lexicon are allowed only as carrier prose that means nothing on its own.
   Lexicon: {{lexicon}}
3. Never invent a schedule the compiler does not support. If a phrasing would need
   a unit, ordinal, bound, or selector the lexicon does not list, do not write it.

## Task

Produce {{count}} examples for the family `{{family}}`. Each example must express
the seed intent exactly:

```json
{{seedIntent}}
```

Keep the schedule meaning in the words the lexicon supplies; you may add ordinary
context around it. Prefer varied but unambiguous phrasing. Do not add explanatory
text, markdown, or code fences.
