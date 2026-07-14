/**
 * Linearise a ScheduleIntent into the Track B target language.
 * Copy/closed-class tokens only; no free-form strings.
 */
import type { ScheduleIntent } from "../../src/types/intent";

export function linearizeIntent(intent: ScheduleIntent | null): string[] {
  if (intent === null) return ["(", "reject", ")"];
  const tokens = ["(", "every", String(intent.frequency.interval), intent.frequency.unit];

  if (intent.time !== undefined) {
    tokens.push("at", String(intent.time.hour), String(intent.time.minute));
  }

  if (intent.weekdays !== undefined && intent.weekdays.length > 0) {
    tokens.push("days", ...intent.weekdays);
  }

  tokens.push(")");

  return tokens;
}
