/**
 * Candidate verifier: untrusted model prose in, trusted labeled example out.
 *
 * The pipeline is fixed and ordered:
 *
 *   schema check -> text policy -> compiler oracle -> lexicon alignment
 *   -> whole-token coverage -> value cross-check -> fingerprint + provenance
 *   -> accepted record or rejected record with a reason code.
 *
 * The compiler is the only authority on representability, the lexicon aligner
 * is the only authority on labels, and the model is the only thing under
 * test. The verifier is pure and offline: it never opens a provider session.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { compile } from "../../../src/compiler/index.js";
import { tokenize } from "../../../src/model/tokenizer.js";
import { WEEKDAY_GROUPS } from "../../../src/parser/lexicon.js";
import { parse } from "../../../src/parser/parse.js";
import type { ScheduleIntent, Weekday } from "../../../src/types/intent.js";
import { fingerprintSpans } from "../fingerprint.js";
import type { Span } from "../sentence.js";
import { alignText, expectedValues } from "./align.js";
import type { AlignedValue } from "./align.js";
import { writeRejections } from "./rejections.js";
import type { RejectedRecord, RejectionReason } from "./rejections.js";
import { candidateSchema } from "./schema.js";
import type { AcceptedRecord } from "./schema.js";

/** Token ceiling shared with the featurizer's sequence limit. */
export const MAX_TOKENS = 128;

export interface VerifyContext {
  angleId: string;
  model: string;
  promptHash: string;
  runId: string;
}

export type VerifyResult =
  { ok: true; record: AcceptedRecord } | { ok: false; rejection: RejectedRecord };

function isWellFormedUtf16(text: string): boolean {
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(index + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
      index++;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }

  return true;
}

/** Text policy: non-empty, control-free, UTF-8 clean, at most 128 tokens. */
function textProblem(text: string): string | null {
  if (text.trim().length === 0) return "empty text";
  if (/[\u0000-\u001f\u007f]/.test(text)) return "control character";
  if (!isWellFormedUtf16(text)) return "unpaired UTF-16 surrogate";
  const tokens = tokenize(text);

  if (tokens.length === 0) return "no tokens";

  if (tokens.length > MAX_TOKENS) {
    return `too many tokens: ${tokens.length} > ${MAX_TOKENS}`;
  }

  return null;
}

/** The intent's required values must all be readable from the aligned text. */
function checkValues(intent: ScheduleIntent, slots: readonly AlignedValue[]): boolean {
  const expected = expectedValues(intent);

  for (const value of expected) {
    if (!value.required) continue;

    const found = slots.some((slot) => {
      if (slot.label === value.label && slot.value === value.value) return true;

      if (value.label.startsWith("WEEKDAY_") && slot.label.startsWith("DAYGROUP_")) {
        const group = WEEKDAY_GROUPS[String(slot.value)];

        return group?.includes(value.value as Weekday) ?? false;
      }

      return false;
    });

    if (!found) return false;
  }

  for (const slot of slots) {
    const allowed = expected.filter((value) => value.label === slot.label);

    if (allowed.length === 0) return false;
    const justified = allowed.some((value) => value.value === slot.value);

    if (!justified) return false;
  }

  return true;
}

function rejection(
  reason: RejectionReason,
  text: string,
  candidate: unknown,
  context: VerifyContext,
  detail: string,
): VerifyResult {
  return {
    ok: false,
    rejection: {
      reason,
      text,
      candidate,
      angleId: context.angleId,
      model: context.model,
      promptHash: context.promptHash,
      runId: context.runId,
      detail,
    },
  };
}

function textOf(candidate: unknown): string {
  if (typeof candidate !== "object" || candidate === null) return "";
  const text = (candidate as { text?: unknown }).text;

  return typeof text === "string" ? text : "";
}

function accepted(
  text: string,
  intent: ScheduleIntent | null,
  output: "schedule" | "negative",
  spans: readonly Span[],
  context: VerifyContext,
): VerifyResult {
  const record: AcceptedRecord = {
    text,
    intent: intent as AcceptedRecord["intent"],
    output,
    spans: spans as AcceptedRecord["spans"],
    fingerprint: fingerprintSpans(text, spans),
    source: "llm",
    angleId: context.angleId,
    model: context.model,
    promptHash: context.promptHash,
    runId: context.runId,
  };

  return { ok: true, record };
}

/**
 * Verify one candidate against its runner provenance. Async only because the
 * Phase 04 rule parser is async; the verifier itself performs no I/O.
 */
export async function verifyCandidate(
  candidate: unknown,
  context: VerifyContext,
): Promise<VerifyResult> {
  const parsed = candidateSchema.safeParse(candidate);

  if (!parsed.success) {
    return rejection(
      "schema-invalid",
      textOf(candidate),
      candidate,
      context,
      parsed.error.issues.map((issue) => issue.message).join("; "),
    );
  }

  const { text, output } = parsed.data;
  const problem = textProblem(text);

  if (problem !== null) {
    return rejection("text-invalid", text, candidate, context, problem);
  }

  if (output === "negative") {
    if (parsed.data.intent !== null) {
      return rejection(
        "negative-has-schedule",
        text,
        candidate,
        context,
        "negative candidate carried an intent",
      );
    }

    const parsedText = await parse(text, { source: "rules" });

    if (parsedText.ok) {
      return rejection(
        "negative-has-schedule",
        text,
        candidate,
        context,
        "the rule parser found a schedule in negative output",
      );
    }

    const spans: Span[] = [
      { start: 0, end: text.length, label: "O", clauseStart: false, actionable: false },
    ];

    return accepted(text, null, "negative", spans, context);
  }

  if (parsed.data.intent === null) {
    return rejection("unsupported", text, candidate, context, "schedule candidate has no intent");
  }

  const intent = parsed.data.intent as ScheduleIntent;
  const compiled = compile(intent);

  if (!compiled.supported) {
    return rejection(compiled.code, text, candidate, context, compiled.reason);
  }

  const aligned = alignText(text, intent);

  if (!aligned.ok) {
    return rejection(aligned.reason, text, candidate, context, aligned.detail);
  }

  if (!checkValues(intent, aligned.slots)) {
    return rejection(
      "intent-text-mismatch",
      text,
      candidate,
      context,
      "aligned values do not match the intent",
    );
  }

  return accepted(
    text,
    intent,
    "schedule",
    aligned.spans.map((span) => ({
      ...span,
      actionable: span.label !== "O",
    })),
    context,
  );
}

export function acceptedPath(out: string): string {
  return `${out}.accepted.jsonl`;
}

export function rejectionsPath(out: string): string {
  return `${out}.rejections.jsonl`;
}

/** Write accepted records to `<out>.accepted.jsonl`. */
export function writeAccepted(path: string, records: readonly AcceptedRecord[]): string {
  mkdirSync(dirname(path), { recursive: true });
  const body = records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(path, body.length > 0 ? `${body}\n` : "");

  return path;
}

export interface VerifyOptions {
  input: string;
  out: string;
}

export interface VerifySummary {
  accepted: number;
  rejected: number;
  byReason: Record<string, number>;
  acceptedPath: string;
  rejectionsPath: string;
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

/** Parse `--in <raw.jsonl> --out <prefix>`. */
export function parseArgs(argv: readonly string[]): VerifyOptions {
  let input: string | undefined;
  let out: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    switch (arg) {
      case "--in":
        input = readOption(argv, index);
        index++;
        break;
      case "--out":
        out = readOption(argv, index);
        index++;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  if (input === undefined || input.length === 0) {
    throw new RangeError("--in is required");
  }

  if (out === undefined || out.length === 0) {
    throw new RangeError("--out is required");
  }

  return { input, out };
}

function stringField(record: unknown, key: string): string {
  if (typeof record !== "object" || record === null) return "unknown";
  const value = (record as Record<string, unknown>)[key];

  return typeof value === "string" && value.length > 0 ? value : "unknown";
}

/** Walk up to the workspace root (the package.json that lists `workspaces`). */
function repoRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));

  for (;;) {
    const manifest = join(dir, "package.json");

    if (existsSync(manifest)) {
      try {
        const pkg = JSON.parse(readFileSync(manifest, "utf8")) as {
          workspaces?: unknown;
        };

        if (Array.isArray(pkg.workspaces)) return dir;
      } catch {
        // Not a readable manifest; keep walking up.
      }
    }

    const parent = dirname(dir);

    if (parent === dir) return process.cwd();
    dir = parent;
  }
}

/** Resolve a CLI path against the repo root so `--filter` cwd does not matter. */
function resolvePath(value: string): string {
  return isAbsolute(value) ? value : join(repoRoot(), value);
}

/** Verify every line of a raw runner JSONL stream and write both outputs. */
export async function verifyFile(options: VerifyOptions): Promise<VerifySummary> {
  const input = resolvePath(options.input);
  const out = resolvePath(options.out);
  const raw = readFileSync(input, "utf8");
  const acceptedRecords: AcceptedRecord[] = [];
  const rejectedRecords: RejectedRecord[] = [];
  const byReason: Record<string, number> = {};

  for (const line of raw.split("\n")) {
    const trimmed = line.trim();

    if (trimmed.length === 0) continue;

    let record: unknown;

    try {
      record = JSON.parse(trimmed);
    } catch {
      rejectedRecords.push({
        reason: "schema-invalid",
        text: "",
        candidate: trimmed,
        angleId: "unknown",
        model: "unknown",
        promptHash: "unknown",
        runId: "unknown",
        detail: "unparseable JSON line",
      });

      byReason["schema-invalid"] = (byReason["schema-invalid"] ?? 0) + 1;
      continue;
    }

    const context: VerifyContext = {
      angleId: stringField(record, "angleId"),
      model: stringField(record, "model"),
      promptHash: stringField(record, "promptHash"),
      runId: stringField(record, "runId"),
    };

    if (typeof record === "object" && record !== null && "slots" in record) {
      rejectedRecords.push({
        reason: "schema-invalid",
        text: stringField(record, "text"),
        candidate: record,
        ...context,
        detail: "candidate carried a slots hint",
      });

      byReason["schema-invalid"] = (byReason["schema-invalid"] ?? 0) + 1;
      continue;
    }

    const candidate =
      typeof record === "object" && record !== null
        ? {
            text: (record as Record<string, unknown>).text,
            intent: (record as Record<string, unknown>).intent,
            output: (record as Record<string, unknown>).output,
          }
        : record;

    const result = await verifyCandidate(candidate, context);

    if (result.ok) {
      acceptedRecords.push(result.record);
    } else {
      rejectedRecords.push(result.rejection);
      byReason[result.rejection.reason] = (byReason[result.rejection.reason] ?? 0) + 1;
    }
  }

  const acceptedFile = acceptedPath(out);
  const rejectedFile = rejectionsPath(out);

  writeAccepted(acceptedFile, acceptedRecords);
  writeRejections(rejectedFile, rejectedRecords);

  return {
    accepted: acceptedRecords.length,
    rejected: rejectedRecords.length,
    byReason,
    acceptedPath: acceptedFile,
    rejectionsPath: rejectedFile,
  };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const summary = await verifyFile(options);

  const reasons = Object.entries(summary.byReason)
    .map(([reason, count]) => `${reason}=${count}`)
    .join(" ");

  process.stdout.write(
    `verified: ${summary.accepted} accepted, ${summary.rejected} rejected` +
      `${reasons.length > 0 ? ` (${reasons})` : ""}\n` +
      `accepted -> ${summary.acceptedPath}\n` +
      `rejections -> ${summary.rejectionsPath}\n`,
  );
}
