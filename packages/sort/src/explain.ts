import type { SortIntent } from "./types.js";

const INPUT_ORDER_TIES =
  "Equal keys preserve JavaScript input order. SQL compilation requires a caller tie breaker.";

const VALUE_ORDER =
  "Strings use Unicode scalar order, matching SQLite BINARY for valid UTF-8. " +
  "Missing, null and invalid field values use the stated null placement.";

function describeTies(intent: SortIntent): string {
  const { tieBreaker } = intent;

  if (!tieBreaker) return INPUT_ORDER_TIES;
  const isSortKey = intent.keys.some((key) => key.field === tieBreaker);
  const direction = isSortKey ? ", with its stated direction" : ", ascending";

  return `Ties use the caller's unique non-null field ${JSON.stringify(tieBreaker)}${direction}.`;
}

/** Explain an intent already checked against the caller's catalog. */
export function explain(intent: SortIntent): string {
  const keys = intent.keys.map((key, index) => {
    const field = JSON.stringify(key.field);
    const direction = key.direction === "asc" ? "ascending" : "descending";
    const nullsSource = key.nullsSource === "caller-default" ? " (caller default)" : " (explicit)";

    return `${index + 1}. ${field} ${direction}, null values ${key.nulls}${nullsSource}.`;
  });

  return [...keys, describeTies(intent), VALUE_ORDER].join(" ");
}
