/** Whole-utterance source contract. Gold is supplied by semantic cases, not parsing. */
import { readFileSync } from "node:fs";

import { z } from "zod";

import { isScheduleLabel, type Label, LABELS } from "../../../../src/model/labels";
import { SPEECH_ACTS } from "../../../../src/model/speech-acts";
import type { ScheduleIntent } from "../../../../src/types/intent";
import { fingerprintExample } from "../../fingerprint";
import { oracleAgrees, oracleDecode } from "../../oracle";
import { normalizeAnnotatedText, Sentence } from "../../sentence";
import { type CorpusRecord, SPLITS } from "../merge";

export const CASE_SETS = [
  "interval",
  "interval-clock",
  "daily",
  "weekly",
  "weekly-untimed",
  "monthly",
  "monthly-untimed",
  "bounds",
  "exceptions",
  "correction",
] as const;

export type CaseSet = (typeof CASE_SETS)[number];

const formSchema = z
  .object({
    id: z.string().min(1),
    speechAct: z.enum(SPEECH_ACTS).refine((act) => act !== "none"),
    pattern: z.string().min(1),
    tags: z.array(z.string().min(1)).min(1),
  })
  .strict();

const groupSchema = z
  .object({
    id: z.string().min(1),
    caseSet: z.enum(CASE_SETS),
    split: z.enum(SPLITS),
    forms: z.array(formSchema).min(1),
  })
  .strict();

export const constructionSourceSchema = z
  .object({
    version: z.literal(1),
    sourceId: z.string().min(1),
    authorship: z.literal("ai-authored"),
    review: z
      .object({
        method: z.string(),
        independent: z.literal(false),
        status: z.enum(["draft", "self-reviewed"]),
      })
      .strict(),
    groups: z.array(groupSchema).min(1),
  })
  .strict();

export type ConstructionSource = z.infer<typeof constructionSourceSchema>;
export type ConstructionGroup = ConstructionSource["groups"][number];
export type ConstructionForm = ConstructionGroup["forms"][number];

export interface Fragment {
  text: string;
  label: Label;
  /** Tight joins are used only within a clock or ordinal token. */
  separator?: string;
}

export interface SemanticCase {
  id: string;
  semanticFamily: string;
  family: CorpusRecord["family"];
  intent: ScheduleIntent;
  slots: Readonly<Record<string, readonly Fragment[]>>;
  /** Checked separately before insertion into correction text. */
  discarded?: { intent: ScheduleIntent; slot: string };
  tags: readonly string[];
}

export const CONSTRUCTIONS_PATH = new URL(
  "../../../data/production/constructions-v1.json",
  import.meta.url,
);

export function loadConstructions(path: string | URL = CONSTRUCTIONS_PATH): ConstructionSource {
  const source = constructionSourceSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const groups = new Set<string>();

  for (const group of source.groups) {
    if (groups.has(group.id)) throw new Error(`Duplicate construction group: ${group.id}`);
    groups.add(group.id);
    const forms = new Set<string>();

    for (const form of group.forms) {
      if (forms.has(form.id)) {
        throw new Error(`Duplicate construction form: ${group.id}/${form.id}`);
      }

      forms.add(form.id);
      parsePattern(form.pattern);
    }
  }

  return source;
}

type Piece =
  | { type: "prose"; text: string }
  | { type: "literal"; text: string; label: Label; discarded: boolean }
  | { type: "slot"; name: string; discarded: boolean }
  | { type: "clause" };

const labels = new Set<string>(LABELS);

/** {slot}, {!discardedSlot}, {#} clause, [ROLE:literal], [!ROLE:discarded literal]. */
export function parsePattern(pattern: string): Piece[] {
  const pieces: Piece[] = [];
  const marker = /\{([^{}]+)\}|\[(!?[A-Z_0-9]+):([^\[\]]+)\]/g;
  let end = 0;

  const prose = (text: string): void => {
    if (/[{}\[\]]/.test(text)) throw new Error(`Malformed construction markup: ${text}`);
    if (text.trim()) pieces.push({ type: "prose", text: text.trim() });
  };

  for (const match of pattern.matchAll(marker)) {
    prose(pattern.slice(end, match.index));

    if (match[1] === "#") {
      pieces.push({ type: "clause" });
    } else if (match[1]) {
      const discarded = match[1].startsWith("!");
      const name = match[1].replace(/^!/, "");

      if (!/^[a-z][a-zA-Z0-9]*$/.test(name)) throw new Error(`Invalid slot: ${name}`);
      pieces.push({ type: "slot", name, discarded });
    } else {
      const discarded = match[2]!.startsWith("!");
      const label = match[2]!.replace(/^!/, "");

      if (!labels.has(label)) throw new Error(`Unknown label: ${label}`);

      pieces.push({
        type: "literal",
        text: match[3]!,
        label: label as Label,
        discarded,
      });
    }

    end = match.index + match[0].length;
  }

  prose(pattern.slice(end));

  return pieces;
}

export function renderConstruction(
  source: ConstructionSource,
  group: ConstructionGroup,
  form: ConstructionForm,
  semantic: SemanticCase,
): CorpusRecord {
  const sentence = new Sentence();
  const request = form.speechAct === "request" || form.speechAct === "correction";

  sentence.clause();

  for (const piece of parsePattern(form.pattern)) {
    if (piece.type === "clause") {
      sentence.clause();
      continue;
    }

    if (piece.type === "prose") {
      sentence.add(piece.text, "O", /^[,;:?!]/.test(piece.text) ? "" : " ", false);
      continue;
    }

    const fragments =
      piece.type === "literal"
        ? [{ text: piece.text, label: piece.label }]
        : semantic.slots[piece.name];

    if (!fragments?.length) {
      throw new Error(
        `Missing slot in ${group.id}/${form.id}/${semantic.id}: ${piece.type === "slot" ? piece.name : "literal"}`,
      );
    }

    for (const fragment of fragments) {
      sentence.add(
        fragment.text,
        fragment.label,
        fragment.separator ?? " ",
        request && !piece.discarded && isScheduleLabel(fragment.label),
      );
    }
  }

  const annotated = normalizeAnnotatedText(sentence.text, sentence.spans);

  return {
    id: `${source.sourceId}/${group.id}/${form.id}/${semantic.id}`,
    split: group.split,
    family: semantic.family,
    ...annotated,
    intent: structuredClone(semantic.intent),
    speechAct: form.speechAct,
    fingerprint: fingerprintExample({ ...annotated, speechAct: form.speechAct }),
    template: `${semantic.semanticFamily}/${group.id}/${form.id}`,
    source: "rule",
    provenance: {
      authorship: source.authorship,
      sourceId: source.sourceId,
      groupId: group.id,
      constructionId: `${group.id}/${form.id}`,
      sourceRecordId: semantic.id,
    },
  };
}

/**
 * Validate the old side of a correction independently of the actionable
 * replacement. Otherwise a correctly labeled replacement could hide corrupt
 * discarded-clause supervision.
 */
export function verifyDiscarded(semantic: SemanticCase): void {
  if (!semantic.discarded) return;
  const fragments = semantic.slots[semantic.discarded.slot];

  if (!fragments?.length) throw new Error(`Missing discarded schedule: ${semantic.id}`);
  const sentence = new Sentence();

  sentence.clause();
  for (const part of fragments) sentence.add(part.text, part.label, part.separator ?? " ");

  const outcome = oracleDecode({
    id: semantic.id,
    text: sentence.text,
    spans: sentence.spans,
    intent: semantic.discarded.intent,
    speechAct: "request",
    source: "kinds-sample",
  });

  if (!oracleAgrees(outcome, semantic.discarded.intent)) {
    throw new Error(`Discarded schedule gold mismatch: ${semantic.id}`);
  }
}
