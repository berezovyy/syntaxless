/** Bounded corpus preparation using the existing cron renderers and shard format. */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import type { Label } from "../../src/model/labels.js";
import type { SpeechAct } from "../../src/model/speech-acts.js";
import { tokenize } from "../../src/model/tokenizer.js";
import type { ScheduleIntent } from "../../src/types/intent.js";
import { validate } from "../../src/validate.js";
import { sentence as ordinarySentence } from "./background.js";
import { canonicalizeIntent } from "./canonical-intent.js";
import { loadCompactProtection } from "./compact-protection.js";
import { type CompactKind, HOURLY_MINUTE, intentScope, REQUIRED_SCOPE } from "./compact-scope.js";
import { type CorpusRecord, corpusRecordSchema, type Split } from "./corpus/merge.js";
import { loadProtectedText, type ProtectedText } from "./corpus/production/protected-text.js";
import { diskPreflight, replaceJson } from "./corpus/production/publication.js";
import { corpusCodeHashes } from "./corpus/production/source-hashes.js";
import { CORRECTION_FRAMES, wrapCorrection } from "./correction.js";
import { type DiscourseWrap, wrapDiscourse } from "./discourse.js";
import { featurizeKindsStreaming } from "./featurize-kinds.js";
import { featurizeRecord, MAX_TOKENS } from "./featurize.js";
import { coarseText, fingerprintPattern, sha256Hex } from "./fingerprint.js";
import { KINDS } from "./kinds.js";
import { spellingVariant } from "./noise.js";
import { oracleAgrees, oracleDecode } from "./oracle.js";
import { createRng, type Rng } from "./rng.js";
import { normalizeAnnotatedText, Sentence } from "./sentence.js";

export const COMPACT_RECIPE = Object.freeze({
  version: 4,
  seed: 20261012,
  basesPerKind: 120,
  attemptsPerKind: 2048,
  ordinaryRows: 512,
  maximumRows: 10000,
});

const REQUESTS: readonly DiscourseWrap[] = [
  { id: "schedule", prefix: "schedule the task" },
  { id: "run", prefix: "run the task" },
  { id: "repeat", prefix: "repeat the operation", suffix: "for the team" },
];

const NEGATIVES: readonly { speech: SpeechAct; wrap: DiscourseWrap }[] = [
  {
    speech: "descriptive",
    wrap: { id: "description", prefix: "the service used to run" },
  },
  {
    speech: "question",
    wrap: { id: "question", prefix: "does the service run", suffix: "or less often?" },
  },
  {
    speech: "quoted",
    wrap: { id: "quotation", prefix: "the document says", quoted: true },
  },
  {
    speech: "one-off",
    wrap: { id: "one-off", prefix: "only once, not", suffix: "again" },
  },
  { speech: "cancel", wrap: { id: "cancel", prefix: "cancel the task scheduled" } },
  {
    speech: "incomplete",
    wrap: { id: "unfinished", prefix: "if we decide to run", suffix: "then" },
  },
];

const OMIT_KINDS = new Set(["negative", "correction", ...NEGATIVES.map((row) => row.speech)]);

const NOISE_TAGS = [
  "spelling-deletion",
  "spelling-insertion",
  "spelling-transposition",
  "spelling-repetition",
  "spelling-keyboard-neighbor",
  "punctuation-comma",
] as const;

function variant(base: Sentence, name: string, index: number) {
  // Independent of the six speech-act frames and template/value cycles.
  const choice =
    parseInt(sha256Hex(`${COMPACT_RECIPE.seed}/${name}/${index}/variant`).slice(0, 2), 16) % 4;

  if (choice === 0) {
    const noisy = spellingVariant(
      {
        text: base.text,
        spans: base.spans,
        template: name,
        style: "compact",
        speechAct: "request",
      },
      `${COMPACT_RECIPE.seed}/${name}/${index}`,
      { force: true },
    );

    if (noisy) {
      return {
        inner: noisy.example,
        tag: `spelling-${noisy.noise}`,
        punctuation: false,
      };
    }
  }

  return {
    inner: base,
    tag: choice === 1 ? "punctuation-comma" : undefined,
    punctuation: choice === 1,
  };
}

function comma(wrap: DiscourseWrap): DiscourseWrap {
  if (!wrap.prefix) throw new Error("Punctuation variants require a carrier prefix");

  return { ...wrap, prefix: `${wrap.prefix},` };
}

function constructionGroup(sentence: Sentence): string {
  const pattern = fingerprintPattern(sentence.text, sentence.spans).map((entry, index) =>
    entry === "slot" ? `slot:${sentence.spans[index]!.label.split("_")[0]}` : entry,
  );

  return sha256Hex(JSON.stringify(pattern));
}

function compactSplit(fingerprint: string): Split {
  const bucket = parseInt(fingerprint.slice(0, 8), 16) % 8;

  return bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train";
}

/** Reserve construction groups for rare families before the hash allocation. */
function assignSplits(rows: CorpusRecord[]): void {
  const families = new Map<string, Set<string>>();
  const assigned = new Map<string, Split>();

  for (const row of rows) {
    const keys = row.intent
      ? [row.family, ...intentScope(row.intent).map((tag) => `scope:${tag}`)]
      : [`speech:${row.speechAct}`];

    for (const key of keys) {
      const groups = families.get(key) ?? new Set<string>();

      groups.add(row.fingerprint);
      families.set(key, groups);
    }
  }

  const ordered = [...families].sort((a, b) => a[1].size - b[1].size || a[0].localeCompare(b[0]));

  for (const [family, groups] of ordered) {
    const sorted = [...groups].sort();

    for (const split of ["train", "validation", "heldout"] as const) {
      if (sorted.some((group) => assigned.get(group) === split)) continue;
      const available = sorted.find((group) => !assigned.has(group));

      if (!available) throw new Error(`Insufficient construction groups for ${family}`);
      assigned.set(available, split);
    }
  }

  for (const row of rows) {
    row.split = assigned.get(row.fingerprint) ?? compactSplit(row.fingerprint);
  }
}

function drawIntent(kind: CompactKind, rng: Rng, index: number): ScheduleIntent {
  const intent = structuredClone(rng.pick(kind.intents));

  if (!intent) throw new Error(`No schedule intent for ${kind.id}`);

  if (kind.id === "n-interval" && index % 4 === 0) {
    intent.frequency.unit = index % 8 === 0 ? "minute" : "hour";
    intent.frequency.interval = rng.pick([24, 25, 60, 90, 120, 180, 1000]);
  }

  if (kind.id === "hour-at-minute") intent.time = { minute: rng.int(0, 59) };

  // Exercise the full clock, with extra whole hours and quarter hours. No
  // evaluation sentence or model prediction supplies values or labels.
  if (intent.time && intent.time.hour !== undefined) {
    intent.time = {
      hour: index % 24,
      minute: index % 3 === 0 ? 0 : index % 3 === 1 ? rng.pick([15, 30, 45]) : rng.int(0, 59),
    };
  }

  validate(intent);

  return intent;
}

function render(kind: CompactKind, intent: ScheduleIntent, template: string): Sentence | undefined {
  const sentence = new Sentence();

  sentence.clause();
  if (!kind.render(sentence, intent, template)) return undefined;

  // Digits supply the value; the model only needs to identify their role.
  // Named times and word numbers retain their established semantic labels.
  for (const span of sentence.spans) {
    const role = /^(HOUR|MINUTE|INTERVAL)_\d+$/.exec(span.label)?.[1];

    if (role && /^\d+$/.test(sentence.text.slice(span.start, span.end))) {
      span.label = `${role}_LIT` as Label;
    }
  }

  return sentence;
}

export function compactRecords(
  protectedText: Pick<ProtectedText, "matches" | "sources"> = loadProtectedText(),
): CorpusRecord[] {
  const rng = createRng(COMPACT_RECIPE.seed);
  const rows: CorpusRecord[] = [];
  const seen = new Map<string, string>();

  function append(
    sentence: Pick<Sentence, "text" | "spans">,
    intent: ScheduleIntent | null,
    speech: SpeechAct,
    family: CorpusRecord["family"],
    template: string,
    fingerprint: string,
  ): boolean {
    const normalized = normalizeAnnotatedText(sentence.text, sentence.spans);

    if (protectedText.matches(normalized.text)) return false;
    const tokens = tokenize(normalized.text);

    if (tokens.length > MAX_TOKENS) return false;

    const record: CorpusRecord = {
      id: `compact-v${COMPACT_RECIPE.version}-${rows.length}`,
      split: compactSplit(fingerprint),
      family,
      fingerprint,
      ...normalized,
      intent,
      speechAct: speech,
      template,
      source: "rule",
    };

    corpusRecordSchema.parse(record);

    if (!oracleAgrees(oracleDecode({ ...record, source: "kind-seed" }), intent, speech)) {
      throw new Error(`Compact labels don't reconstruct intent: ${template}, ${record.text}`);
    }

    // Token alignment and all five supervision heads are checked before writing.
    const features = featurizeRecord(record, tokens);

    const supervision = JSON.stringify({
      intent: intent && canonicalizeIntent(intent),
      speech,
      labels: [...features.labels],
      boundaries: [...features.boundaries],
      actionable: [...features.actionable],
    });

    const existing = seen.get(record.text);

    if (existing !== undefined) {
      if (existing !== supervision) {
        throw new Error(`Conflicting compact supervision: ${record.text}`);
      }

      return false;
    }

    if (rows.length >= COMPACT_RECIPE.maximumRows) {
      throw new Error("Compact corpus row limit exceeded");
    }

    seen.set(record.text, supervision);
    rows.push(record);

    return true;
  }

  for (const kind of [...KINDS, HOURLY_MINUTE]) {
    if (OMIT_KINDS.has(kind.id)) continue;
    let bases = 0;

    for (
      let attempt = 0;
      attempt < COMPACT_RECIPE.attemptsPerKind && bases < COMPACT_RECIPE.basesPerKind;
      attempt++
    ) {
      const template = kind.templates[attempt % kind.templates.length]!.id;
      const intent = drawIntent(kind, rng, attempt);
      const base = render(kind, intent, template);

      if (!base || protectedText.matches(base.text)) continue;

      const fingerprint = constructionGroup(base);
      const name = `${kind.id}/${template}`;

      if (!append(base, intent, "request", kind.family, name, fingerprint)) continue;

      // The clean base and group stay unchanged. Replace existing carrier rows
      // rather than growing the corpus with a second set of variants.
      const changed = variant(base, name, bases);
      const request = REQUESTS[bases % REQUESTS.length]!;
      const suffix = changed.tag ? `/${changed.tag}` : "";

      const carrierAccepted = append(
        wrapDiscourse(changed.inner, "request", changed.punctuation ? comma(request) : request),
        intent,
        "request",
        kind.family,
        `${name}/${request.id}${suffix}`,
        fingerprint,
      );

      const negative = NEGATIVES[bases % NEGATIVES.length]!;

      append(
        wrapDiscourse(
          carrierAccepted ? changed.inner : base,
          negative.speech,
          carrierAccepted && changed.punctuation ? comma(negative.wrap) : negative.wrap,
        ),
        null,
        negative.speech,
        kind.family,
        `${name}/${negative.wrap.id}${carrierAccepted ? suffix : ""}`,
        fingerprint,
      );

      if (bases % 8 === 0) {
        const previousIntent = drawIntent(kind, rng, attempt + 1);
        const previous = render(kind, previousIntent, template);

        if (
          previous &&
          !protectedText.matches(previous.text) &&
          constructionGroup(previous) === fingerprint
        ) {
          const frame = CORRECTION_FRAMES[Math.floor(bases / 8) % CORRECTION_FRAMES.length]!;
          const correction = wrapCorrection(previous, base, frame);

          if (correction) {
            append(
              correction,
              intent,
              "correction",
              kind.family,
              `${name}/${frame.id}`,
              fingerprint,
            );
          }
        }
      }

      bases++;
    }

    if (bases === 0) throw new Error(`Compact corpus has no ${kind.id} examples`);
  }

  for (
    let attempt = 0, taken = 0;
    taken < COMPACT_RECIPE.ordinaryRows && attempt < COMPACT_RECIPE.ordinaryRows * 16;
    attempt++
  ) {
    const sentence = new Sentence();

    sentence.add(ordinarySentence(rng), "O");
    // Ordinary prose has no schedule construction, so its wording supplies
    // the group. Numeric values cannot move a construction to another split.
    const fingerprint = sha256Hex(coarseText(sentence.text));

    if (append(sentence, null, "none", "negative", "negative/ordinary", fingerprint)) taken++;
  }

  assignSplits(rows);

  return rows;
}

/** Diagnostic preparation doesn't authorize training on its output. */
export async function writeCompactCorpus(
  directory: string,
  rows: CorpusRecord[],
  protectedSources: ProtectedText["sources"],
) {
  if (rows.length === 0 || rows.length > COMPACT_RECIPE.maximumRows) {
    throw new Error("Compact corpus requires 1 to 10000 rows");
  }

  const groups = new Map<string, Split>();
  const coverage: Record<string, Record<string, number>> = {};
  const scopeCoverage: Record<string, Record<string, number>> = {};
  const noiseCoverage: Record<string, Record<string, { positive: number; negative: number }>> = {};

  for (const row of rows) {
    const existing = groups.get(row.fingerprint);

    if (existing && existing !== row.split) throw new Error("Compact construction split leak");
    groups.set(row.fingerprint, row.split);
    const counts = (coverage[row.split] ??= {});
    const key = row.intent ? row.family : `speech:${row.speechAct}`;

    counts[key] = (counts[key] ?? 0) + 1;

    if (row.intent) {
      const scoped = (scopeCoverage[row.split] ??= {});

      for (const tag of intentScope(row.intent)) scoped[tag] = (scoped[tag] ?? 0) + 1;
    }

    const tag = row.template?.split("/").at(-1);

    if (NOISE_TAGS.some((noise) => noise === tag)) {
      const counts = (noiseCoverage[row.split] ??= {});
      const pair = (counts[tag!] ??= { positive: 0, negative: 0 });

      pair[row.intent ? "positive" : "negative"]++;
    }
  }

  for (const split of ["train", "validation", "heldout"] as const) {
    for (const family of new Set(rows.filter((row) => row.intent).map((row) => row.family))) {
      if (!coverage[split]?.[family]) throw new Error(`Missing ${split} positive ${family}`);
    }

    for (const speech of [...NEGATIVES.map((row) => row.speech), "none"]) {
      if (!coverage[split]?.[`speech:${speech}`]) {
        throw new Error(`Missing ${split} negative ${speech}`);
      }
    }

    for (const tag of REQUIRED_SCOPE) {
      if (!scopeCoverage[split]?.[tag]) throw new Error(`Missing ${split} intent scope ${tag}`);
    }

    for (const tag of NOISE_TAGS) {
      if (!noiseCoverage[split]?.[tag]?.positive || !noiseCoverage[split]?.[tag]?.negative) {
        throw new Error(`Missing ${split} paired noise ${tag}`);
      }
    }
  }

  // Verify scope and grouping before allocating corpus or feature files.
  mkdirSync(directory, { recursive: true });
  if (readdirSync(directory).length) throw new Error(`Compact output must be empty: ${directory}`);
  diskPreflight(directory, 128 * 1024 ** 2);

  const prefix = join(directory, "corpus");
  const corpus = `${prefix}.jsonl`;
  const sourceHashes = corpusCodeHashes(import.meta.url);

  const source = writeJsonl(corpus, rows);
  const summary = await featurizeKindsStreaming(corpus, prefix);

  for (const split of ["train", "validation", "heldout"] as const) {
    const shard = summary.splits[split];

    if (
      shard.sequences !== rows.filter((row) => row.split === split).length ||
      shard.skippedEmpty ||
      shard.skippedLong
    ) {
      throw new Error(`Compact ${split} dropped records`);
    }

    const manifestPath = `${prefix}-${split}.json`;
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

    const fingerprints = [...groups]
      .filter(([, value]) => value === split)
      .map(([key]) => key)
      .sort();

    replaceJson(manifestPath, { ...manifest, fingerprints });
  }

  const proof = {
    recipe: COMPACT_RECIPE,
    sourceHashes,
    corpus,
    ...source,
    summary,
    groups: groups.size,
    splitRule:
      "reserve groups for rare-family coverage, then SHA-256 modulo eight; " +
      "requests, carriers, refusals and corrections retain the inner group",
    protectedSources,
    trainingEligible: false,
    coverage,
    scopeCoverage,
    noiseCoverage,
    peakRssKiB: process.resourceUsage().maxRSS,
  };

  writeFileSync(join(directory, "preparation.json"), JSON.stringify(proof, null, 2) + "\n", {
    flag: "wx",
  });

  return { prefix, corpus, proof };
}

export async function prepareCompactCorpus(directory: string) {
  // Validate available evidence before writing, and record historical gaps.
  const protection = loadCompactProtection();

  const result = await writeCompactCorpus(
    directory,
    compactRecords(protection),
    protection.sources,
  );

  const proof = {
    ...result.proof,
    consumption: protection.consumption,
    trainingEligible: true,
  };

  replaceJson(join(directory, "preparation.json"), proof);

  return { ...result, proof };
}
