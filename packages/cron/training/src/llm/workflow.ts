/**
 * One angle to one session to a batch of raw candidates.
 *
 * The workflow renders the angle's prompt template, checks the response cache
 * before spending a request, and parses one JSON candidate per line. Parsing is
 * deliberately shallow: the prompt fixes the candidate shape and Phase 11 owns
 * the authoritative schema, so a line that will not parse is a candidate-level
 * reject, never a crash. Each parsed line carries provenance the verifier can
 * trust without re-deriving it.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Label } from "../../../src/model/labels.js";
import { LEXICON } from "../../../src/parser/lexicon.js";
import type { ScheduleIntent } from "../../../src/types/intent.js";
import { createRng } from "../rng.js";
import type { Family } from "../sampler.js";
import { sampleIntent } from "../sampler.js";
import type { Angle } from "./angles.js";
import { cacheKey, readCache, writeCache } from "./cache.js";
import { createSession, model, prompt } from "./client.js";

/** System turn is fixed; all per-angle instruction belongs in the template. */
export const SYSTEM_PROMPT =
  "You emit one JSON object per line and nothing else. Follow the user's rules exactly.";

/**
 * Compact JSON-schema-ish description substituted for `{{schema}}`. Phase 11
 * defines the parsing schema; this text only tells the model the field names.
 */
export const CANDIDATE_SCHEMA = JSON.stringify({
  type: "object",
  required: ["text", "intent", "output"],
  properties: {
    text: { type: "string" },
    intent: { type: ["object", "null"] },
    output: { enum: ["schedule", "negative"] },
  },
});

/**
 * Labels a family's surface words may carry. Used to trim the runtime lexicon
 * to "only the values the angle may use"; negatives get everything so the model
 * knows which words to avoid.
 */
const FAMILY_LABELS: Record<Family, readonly Label[]> = {
  interval: ["RECUR", "INTERVAL", "COUNT", "GLUE"],
  "weekday-list": ["WEEKDAY", "DAYGROUP", "RECUR", "GLUE"],
  "weekday-range": ["WEEKDAY", "RECUR", "GLUE"],
  "ordinal-weekday": ["ORDINAL", "WEEKDAY", "COUNT", "RECUR", "GLUE"],
  "day-of-month": ["COUNT", "ORDINAL", "RECUR", "GLUE"],
  "time-of-day": ["COUNT", "MERIDIEM", "GLUE", "RECUR"],
  bounds: ["BOUND", "COUNT", "RECUR", "GLUE"],
  exception: ["EXCEPT", "WEEKDAY", "DAYGROUP", "RECUR", "GLUE"],
  terse: ["WEEKDAY", "DAYGROUP", "RECUR", "GLUE"],
  negative: [
    "WEEKDAY",
    "DAYGROUP",
    "INTERVAL",
    "ORDINAL",
    "COUNT",
    "RECUR",
    "GLUE",
    "BOUND",
    "EXCEPT",
    "MERIDIEM",
  ],
};

export interface PromptFile {
  fields: Record<string, string>;
  body: string;
}

/** Parse the YAML-ish front-matter used by `prompts/<promptId>.md`. */
export function parsePrompt(raw: string): PromptFile {
  const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(raw);

  if (match === null) throw new Error("prompt: missing front-matter");
  const fields: Record<string, string> = {};

  for (const line of match[1]!.split(/\r?\n/)) {
    const separator = line.indexOf(":");

    if (separator === -1) continue;
    const key = line.slice(0, separator).trim();

    if (key.length > 0) fields[key] = line.slice(separator + 1).trim();
  }

  return { fields, body: raw.slice(match[0].length) };
}

function promptsDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "prompts");
}

export function loadPrompt(promptId: string): PromptFile {
  return parsePrompt(readFileSync(join(promptsDir(), `${promptId}.md`), "utf8"));
}

export interface PromptVars {
  lexicon: string;
  family: string;
  seedIntent: string;
  count: number;
  schema: string;
}

/** Substitute the five template placeholders. The result is the cache input. */
export function renderPrompt(body: string, vars: PromptVars): string {
  return body
    .replaceAll("{{lexicon}}", vars.lexicon)
    .replaceAll("{{family}}", vars.family)
    .replaceAll("{{seedIntent}}", vars.seedIntent)
    .replaceAll("{{count}}", String(vars.count))
    .replaceAll("{{schema}}", vars.schema);
}

/** Render only the runtime lexicon entries a family's surface words may use. */
export function lexiconFor(family: Family): string {
  const wanted = new Set<string>(FAMILY_LABELS[family]);
  const grouped = new Map<string, string[]>();

  for (const [word, entry] of Object.entries(LEXICON)) {
    if (!wanted.has(entry.label)) continue;
    const list = grouped.get(entry.label);

    if (list === undefined) grouped.set(entry.label, [word]);
    else list.push(word);
  }

  return [...grouped.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([label, words]) => `${label}: ${[...words].sort().join(", ")}`)
    .join("\n");
}

/** Deterministic per-angle seed so re-runs render the same prompt and hit cache. */
export function angleSeed(angleId: string): number {
  return createHash("sha256").update(angleId).digest().readUInt32BE(0);
}

export interface CandidateMeta {
  angleId: string;
  promptId: string;
  promptVersion: number;
  promptHash: string;
  model: string;
  runId: string;
}

/** A parsed candidate line plus the provenance attached by the runner. */
export interface RawCandidate {
  angleId: string;
  promptId: string;
  promptVersion: number;
  promptHash: string;
  model: string;
  runId: string;
  rawHash: string;
  text: string;
  intent: ScheduleIntent | null;
  output: "schedule" | "negative";
}

export interface ParsedCandidates {
  candidates: RawCandidate[];
  rejected: number;
}

interface ParsedLine {
  text: string;
  intent: ScheduleIntent | null;
  output: "schedule" | "negative";
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function parseLine(line: string): ParsedLine | null {
  let value: unknown;

  try {
    value = JSON.parse(line);
  } catch {
    return null;
  }

  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }

  const record = value as Record<string, unknown>;

  if (typeof record.text !== "string") return null;
  if (record.output !== "schedule" && record.output !== "negative") return null;
  const intent = record.intent;

  if (intent !== null && typeof intent !== "object") return null;

  return {
    text: record.text,
    intent: (intent ?? null) as ScheduleIntent | null,
    output: record.output,
  };
}

/** Parse JSONL raw model output. Blank lines are ignored; bad lines are counted. */
export function parseCandidates(raw: string, meta: CandidateMeta): ParsedCandidates {
  const candidates: RawCandidate[] = [];
  let rejected = 0;

  for (const line of raw.split("\n")) {
    const trimmed = line.trim();

    if (trimmed.length === 0) continue;
    const parsed = parseLine(trimmed);

    if (parsed === null) {
      rejected++;
      continue;
    }

    candidates.push({
      angleId: meta.angleId,
      promptId: meta.promptId,
      promptVersion: meta.promptVersion,
      promptHash: meta.promptHash,
      model: meta.model,
      runId: meta.runId,
      rawHash: sha256(trimmed),
      text: parsed.text,
      intent: parsed.intent,
      output: parsed.output,
    });
  }

  return { candidates, rejected };
}

export interface RunContext {
  runId: string;
  count: number;
  cacheDir?: string;
}

export interface AngleRun {
  angle: Angle;
  requested: number;
  candidates: RawCandidate[];
  rejected: number;
  cacheHit: boolean;
  promptHash: string;
}

/**
 * Run one angle. The cache is consulted before a session is opened, so an
 * identical re-run pays for nothing. Network errors propagate to the runner,
 * which owns retry; parse failures are already handled here.
 */
export async function runAngle(angle: Angle, ctx: RunContext): Promise<AngleRun> {
  const promptFile = loadPrompt(angle.promptId);
  const schemaVersion = Number(promptFile.fields.schemaVersion);

  if (!Number.isInteger(schemaVersion)) {
    throw new Error(`prompt ${angle.promptId}: invalid schemaVersion`);
  }

  const sample = sampleIntent(createRng(angleSeed(angle.id)), angle.family);

  const body = renderPrompt(promptFile.body, {
    lexicon: lexiconFor(angle.family),
    family: angle.family,
    seedIntent: JSON.stringify(sample.intent, null, 2) ?? "null",
    count: ctx.count,
    schema: CANDIDATE_SCHEMA,
  });

  const promptHash = sha256(body);
  const ref = model();

  const key = cacheKey({
    model: ref.modelID,
    provider: ref.providerID,
    promptHash,
    schemaVersion,
  });

  const meta: CandidateMeta = {
    angleId: angle.id,
    promptId: angle.promptId,
    promptVersion: angle.promptVersion,
    promptHash,
    model: ref.modelID,
    runId: ctx.runId,
  };

  let raw = readCache(key, ctx.cacheDir);
  const cacheHit = raw !== undefined;

  if (raw === undefined) {
    const session = await createSession(`corpus:${angle.id}`);

    raw = await prompt(session, SYSTEM_PROMPT, body);
    writeCache(key, raw, ctx.cacheDir);
  }

  const parsed = parseCandidates(raw, meta);

  return {
    angle,
    requested: ctx.count,
    candidates: parsed.candidates,
    rejected: parsed.rejected,
    cacheHit,
    promptHash,
  };
}
