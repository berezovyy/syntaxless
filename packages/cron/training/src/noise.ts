/** Deterministic, meaning-preserving spelling noise for paired corpus rows. */
import { createHash } from "node:crypto";

import type { Label } from "../../src/model/labels.js";
import { tokenize } from "../../src/model/tokenizer.js";
import type { RenderedExample } from "./renderer.js";

export type SpellingNoise =
  "deletion" | "insertion" | "transposition" | "repetition" | "keyboard-neighbor";

export interface NoisyExample {
  noise: SpellingNoise;
  example: RenderedExample;
}

export interface SpellingVariantOptions {
  /** Restrict mutation to these labels. Default is {@link NOISE_SAFE_LABELS}. */
  labels?: readonly Label[];
  /** Bypass the ordinary one-in-four sampling for targeted equivalence rows. */
  force?: boolean;
}

/**
 * Labels whose surface string the decoder never looks up. Mutating anything
 * else (WEEKDAY, HOUR, BOUND, GLUE units, …) makes gold unreachable: the
 * oracle cannot invert `wrd` back to wednesday.
 */
export const NOISE_SAFE_LABELS: readonly Label[] = ["O", "EDGE"];

const NOISES: readonly SpellingNoise[] = [
  "deletion",
  "insertion",
  "transposition",
  "repetition",
  "keyboard-neighbor",
];

const KEYBOARD_NEIGHBORS: Readonly<Record<string, string>> = {
  a: "sq",
  b: "vn",
  c: "xv",
  d: "sf",
  e: "wr",
  f: "dg",
  g: "fh",
  h: "gj",
  i: "uo",
  j: "hk",
  k: "jl",
  l: "k",
  m: "n",
  n: "bm",
  o: "ip",
  p: "o",
  q: "w",
  r: "et",
  s: "ad",
  t: "ry",
  u: "yi",
  v: "cb",
  w: "qe",
  x: "zc",
  y: "tu",
  z: "x",
};

function digest(seed: string): Buffer {
  return createHash("sha256").update(seed).digest();
}

function mutateWord(word: string, noise: SpellingNoise, choice: number): string | undefined {
  const characters = [...word];

  if (characters.length < 3) return undefined;
  const interior = 1 + (choice % Math.max(1, characters.length - 2));

  switch (noise) {
    case "deletion":
      characters.splice(interior, 1);
      break;

    case "insertion": {
      const source = characters[interior]!;
      const alternatives = KEYBOARD_NEIGHBORS[source] ?? "x";

      characters.splice(interior, 0, alternatives[choice % alternatives.length]!);
      break;
    }

    case "transposition": {
      let index = choice % (characters.length - 1);

      if (characters[index] === characters[index + 1]) {
        index = (index + 1) % (characters.length - 1);
      }

      [characters[index], characters[index + 1]] = [characters[index + 1]!, characters[index]!];
      break;
    }

    case "repetition":
      characters.splice(interior, 0, characters[interior]!);
      break;

    case "keyboard-neighbor": {
      const source = characters[interior]!;
      const alternatives = KEYBOARD_NEIGHBORS[source];

      if (alternatives === undefined) return undefined;
      characters[interior] = alternatives[choice % alternatives.length]!;
      break;
    }
  }

  const result = characters.join("");

  return result === word ? undefined : result;
}

/**
 * Produce one reproducible typo for one quarter of positive rows. That is
 * enough to cover every large schedule lattice without doubling the already
 * multi-million-row corpus. The caller keeps the clean row beside this row and
 * pins both to the same split frame.
 */
export function spellingVariant(
  example: RenderedExample,
  seed: string,
  options: SpellingVariantOptions = {},
): NoisyExample | undefined {
  const entropy = digest(`${seed}\0${example.text}`);

  if (options.force !== true && entropy[0]! % 4 !== 0) return undefined;
  const eligibleLabels = new Set(options.labels ?? NOISE_SAFE_LABELS);

  const candidates = tokenize(example.text).filter((token) => {
    if (token.kind !== 0 || [...token.text].length < 3) return false;

    return example.spans.some(
      (span) =>
        eligibleLabels.has(span.label) && token.start >= span.start && token.end <= span.end,
    );
  });

  if (candidates.length === 0) return undefined;

  const token = candidates[entropy[1]! % candidates.length]!;
  const noise = NOISES[entropy[2]! % NOISES.length]!;
  const replacement = mutateWord(token.text, noise, entropy[3]!);

  if (replacement === undefined) return undefined;
  const delta = replacement.length - token.text.length;
  const text = example.text.slice(0, token.start) + replacement + example.text.slice(token.end);

  const spans = example.spans.map((span) => {
    if (span.end <= token.start) return { ...span };

    if (span.start >= token.end) {
      return { ...span, start: span.start + delta, end: span.end + delta };
    }

    return { ...span, end: span.end + delta };
  });

  return { noise, example: { ...example, text, spans } };
}
