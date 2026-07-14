/**
 * Combinatorial easy-no English used only when Tatoeba is not pinned.
 * Not a KEEP train source. Each line must pass {@link acceptEasyNo}.
 */
import { acceptEasyNo } from "./filter";

const VERBS = [
  "open",
  "close",
  "read",
  "review",
  "print",
  "send",
  "check",
  "copy",
  "approve",
  "archive",
] as const;

const NOUNS = [
  "file",
  "document",
  "report",
  "page",
  "section",
  "list",
  "table",
  "message",
  "draft",
  "option",
] as const;

const NAMES = ["Alex", "Priya", "Jordan", "Riley", "Sam", "Taylor", "Casey", "Quinn"] as const;
const ORDERS = ["next", "previous", "other", "main", "full", "blank"] as const;

function candidates(): string[] {
  const out: string[] = [];

  for (const verb of VERBS) {
    for (const noun of NOUNS) {
      for (const name of NAMES) {
        out.push(`Please ${verb} the ${noun} for ${name}.`);
        out.push(`Could you ${verb} the ${noun} for ${name}?`);
      }

      for (const order of ORDERS) {
        out.push(`Please ${verb} the ${order} ${noun} in the workspace.`);
      }
    }
  }

  for (const noun of NOUNS) {
    for (let count = 2; count <= 40; count++) {
      out.push(`The ${noun} contains ${count} examples in this list.`);
      out.push(`The ${noun} has ${count} rows and eight columns.`);
    }
  }

  for (const name of NAMES) {
    for (const noun of NOUNS) {
      out.push(`${name} wrote the ${noun} before the review started.`);
      out.push(`Send the ${noun} to ${name} after you finish.`);
    }
  }

  out.push("Please close the window.");
  out.push("Room 10 contains 8 chairs.");
  out.push("Please open the file.");
  out.push("Archive this conversation.");
  out.push("Turn on dark mode.");
  out.push("The build completed successfully.");

  return out;
}

/** Unique filter-passing bootstrap sentences, stable order. */
export function bootstrapEasySentences(): string[] {
  const seen = new Set<string>();
  const kept: string[] = [];

  for (const text of candidates()) {
    if (!acceptEasyNo(text, { seen })) continue;
    kept.push(text);
  }

  return kept;
}
