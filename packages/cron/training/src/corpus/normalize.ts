/**
 * Dedup text keys.
 *
 * The key is for identity, never for training: it discards case, Unicode
 * compatibility differences, whitespace, trailing punctuation, and the spelling
 * of cardinal numbers, all of which vary across sources for the same schedule.
 * The stored `text` is never touched — two records that collapse to one key keep
 * exactly one of their original strings.
 *
 * Digit-word canonicalization reuses the runtime `NUMBER_WORDS` table so the
 * merge and the parser agree on what "twenty one" means. Only the key is
 * rewritten; the corpus keeps the author's surface form.
 */
import { NUMBER_WORDS } from "../../../src/parser/lexicon.js";

/**
 * Everything that is neither a letter nor a number, at either edge of a token.
 * Applied per token so `"Monday,"` and `"Monday"` share a key while `"9:30"`
 * and `"o'clock"` keep their interior punctuation.
 */
const EDGE_PUNCTUATION = /^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu;

function trimToken(token: string): string {
  return token.replace(EDGE_PUNCTUATION, "");
}

/**
 * Rewrite spelled-out cardinals as their canonical decimal. Handles the tens
 * plus ones composition the lexicon allows (`twenty one` -> `21`) so a word
 * number and a digit number meet at the same key.
 */
function canonicalizeNumbers(tokens: readonly string[]): string[] {
  const out: string[] = [];

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    const value = NUMBER_WORDS[token];

    if (value === undefined) {
      out.push(token);
      continue;
    }

    const next = tokens[index + 1];

    if (value >= 20 && value % 10 === 0 && next !== undefined) {
      const ones = NUMBER_WORDS[next];

      if (ones !== undefined && ones >= 1 && ones <= 9) {
        out.push(String(value + ones));
        index++;
        continue;
      }
    }

    out.push(String(value));
  }

  return out;
}

/**
 * The normalized identity of `text`: Unicode NFKC, lowercase, token-edge
 * punctuation removed, cardinal words canonicalized to digits, whitespace
 * collapsed, and any remaining trailing punctuation stripped.
 */
export function normalizeForKey(text: string): string {
  const folded = text.normalize("NFKC").toLowerCase();

  const tokens = folded
    .split(/\s+/)
    .map(trimToken)
    .filter((token) => token.length > 0);

  return canonicalizeNumbers(tokens)
    .join(" ")
    .replace(/[^\p{L}\p{N}]+$/u, "")
    .trim();
}
