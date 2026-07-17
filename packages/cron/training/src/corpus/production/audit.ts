/** Full-stream, disk-indexed corpus audit. No model predictions or eval-derived gold. */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";

import { compile } from "../../../../src/compiler";
import { isScheduleLabel, valueOf } from "../../../../src/model/labels";
import { tokenize } from "../../../../src/model/tokenizer";
import { decodeIntent, UnsafeScheduleContextError } from "../../../../src/parser/decode";
import {
  FREQUENCY_IDIOMS,
  NUMBER_WORDS,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WEEKDAY_PLURAL_ALIASES,
} from "../../../../src/parser/lexicon";
import { normalize } from "../../../../src/parser/normalize";
import { canonicalizeIntent } from "../../canonical-intent";
import { MAX_TOKENS } from "../../featurize";
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import { hasTimeWord, timeShapedNumber } from "../../mix/filter";
import { bandFailures, type Population, sharesOf } from "../../mix/quotas";
import { labeledTokensFromGold, oracleAgrees, oracleDecode } from "../../oracle";
import { assertValid } from "../../sampler";
import { type CorpusRecord, corpusRecordSchema, SPLITS } from "../merge";
import { corpusTextKey, loadProtectedText, ProtectedText } from "./protected-text";

type Split = CorpusRecord["split"];
type Counts = Record<Population, number>;
const counts = (): Counts => ({ R: 0, H: 0, E: 0 });

const bump = (map: Record<string, number>, key: string): void => {
  map[key] = (map[key] ?? 0) + 1;
};

export interface AuditOptions {
  protectedText?: ProtectedText;
  /** Legacy audits can measure missing provenance, but cannot certify production. */
  requireProvenance?: boolean;
  enforceMix?: boolean;
  indexPath?: string;
}

export interface AuditReport {
  version: 1;
  complete: boolean;
  passed: boolean;
  scope: "record-integrity-and-split-isolation-only";
  requirements: { provenance: boolean; mix: boolean; semantics: "every-record" };
  records: number;
  validRecords: number;
  populations: Counts;
  splits: Record<Split, Counts>;
  speechActs: Record<string, number>;
  families: Record<string, number>;
  sources: Record<string, number>;
  constructions: Record<string, number>;
  compiler: { supported: number; unsupported: Record<string, number> };
  values: Record<string, Record<string, number>>;
  missingProvenance: number;
  explicitCorrections: number;
  uniqueTexts: number;
  repeatedRows: number;
  maxTextRepetitions: number;
  uniqueFingerprints: number;
  groups: number;
  violations: Record<string, number>;
  /** IDs and invariant names only: never print protected evaluation wording. */
  examples: Array<{ id: string; violation: string }>;
  protectedSources: ProtectedText["sources"];
}

interface Owner {
  split: string;
  signature: string;
}

export class CorpusAudit {
  readonly #db: Database;
  readonly #protected: ProtectedText;
  readonly report: AuditReport;
  #finished = false;

  constructor(options: AuditOptions = {}) {
    this.#protected = options.protectedText ?? loadProtectedText();

    if (options.indexPath && options.indexPath !== ":memory:" && existsSync(options.indexPath)) {
      throw new Error("Audit index path already exists; refusing to modify it");
    }

    this.#db = new Database(options.indexPath ?? ":memory:");

    this.#db.exec(`PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-32768;
      CREATE TABLE texts (hash TEXT PRIMARY KEY, split TEXT, signature TEXT, n INTEGER);
      CREATE TABLE fingerprints (hash TEXT PRIMARY KEY, split TEXT);
      CREATE TABLE groups (id TEXT PRIMARY KEY, split TEXT);
      CREATE TABLE ids (id TEXT PRIMARY KEY); BEGIN;`);

    this.report = {
      version: 1,
      complete: false,
      passed: false,
      scope: "record-integrity-and-split-isolation-only",
      requirements: {
        provenance: options.requireProvenance ?? true,
        mix: options.enforceMix ?? true,
        semantics: "every-record",
      },
      records: 0,
      validRecords: 0,
      populations: counts(),
      splits: { train: counts(), validation: counts(), heldout: counts() },
      speechActs: {},
      families: {},
      sources: {},
      constructions: {},
      compiler: { supported: 0, unsupported: {} },
      values: {},
      missingProvenance: 0,
      explicitCorrections: 0,
      uniqueTexts: 0,
      repeatedRows: 0,
      maxTextRepetitions: 0,
      uniqueFingerprints: 0,
      groups: 0,
      violations: {},
      examples: [],
      protectedSources: this.#protected.sources,
    };
  }

  #fail(id: string, violation: string): void {
    bump(this.report.violations, violation);

    if (this.report.examples.filter((item) => item.violation === violation).length < 3) {
      this.report.examples.push({ id, violation });
    }
  }

  add(value: unknown): void {
    if (this.#finished) throw new Error("Cannot add records to a finished audit");
    const report = this.report;

    report.records++;

    const normalized =
      value && typeof value === "object" && !Array.isArray(value)
        ? ({ ...value } as Record<string, unknown>)
        : {};

    delete normalized.seed;
    normalized.source ??= "rule";
    const parsed = corpusRecordSchema.safeParse(normalized);

    if (!parsed.success) {
      this.#fail(String(normalized.id ?? report.records), "schema");

      return;
    }

    const record = parsed.data as CorpusRecord;
    const id = record.id;
    const before = Object.values(report.violations).reduce((a, b) => a + b, 0);

    if (!id || this.#db.query("INSERT OR IGNORE INTO ids VALUES (?)").run(id).changes === 0) {
      this.#fail(id, "duplicate-or-empty-id");
    }

    const act = record.speechAct;

    if (!act) {
      this.#fail(id, "missing-speech-act");

      return;
    }

    const request = act === "request" || act === "correction";
    const hasSchedule = record.spans.some((span) => isScheduleLabel(span.label));
    const population: Population = request ? "R" : hasSchedule ? "H" : "E";

    report.populations[population]++;
    report.splits[record.split][population]++;
    bump(report.speechActs, act);
    bump(report.families, record.family);
    if (normalize(record.text) !== record.text) this.#fail(id, "not-normalized");
    if (this.#protected.matches(record.text)) this.#fail(id, "protected-wording");

    if (!record.provenance) {
      report.missingProvenance++;
      if (report.requirements.provenance) this.#fail(id, "missing-provenance");
    } else {
      const p = record.provenance;

      bump(report.sources, `${p.authorship}/${p.sourceId}`);
      bump(report.constructions, `${p.sourceId}/${p.constructionId}`);

      if (p.authorship === "public-source" && (!p.license || !p.attribution || !p.sourceRecordId)) {
        this.#fail(id, "missing-public-source-attribution");
      }

      const group = `${p.sourceId}/${p.groupId}`;

      const owner = this.#db
        .query<{ split: string }, [string]>("SELECT split FROM groups WHERE id=?")
        .get(group);

      if (owner && owner.split !== record.split) this.#fail(id, "group-crosses-splits");
      this.#db.query("INSERT OR IGNORE INTO groups VALUES (?, ?)").run(group, record.split);
    }

    const spans = record.spans;
    let cursor = 0;

    for (const [spanIndex, span] of spans.entries()) {
      if (
        span.start < cursor ||
        span.start < 0 ||
        span.end <= span.start ||
        span.end > record.text.length
      ) {
        this.#fail(id, "invalid-span-range");
      }

      cursor = span.end;
      if (typeof span.actionable !== "boolean") this.#fail(id, "missing-actionability");

      if (!isScheduleLabel(span.label) && (span.actionable || span.clauseStart)) {
        this.#fail(id, "actionable-or-boundary-on-prose");
      }

      if (
        span.label === "CONTEXT_NUMBER" &&
        !/^\d+$/.test(record.text.slice(span.start, span.end))
      ) {
        this.#fail(id, "invalid-context-number");
      }

      if (!request && span.actionable) this.#fail(id, "actionable-negative");
      // Oracle agreement alone trusts supplied roles. Check direct lexical
      // values, allowing validated whole-phrase unit conversion. Clock labels
      // do NOT necessarily equal literals: HOUR_20 may
      // correctly annotate `8 pm`, and MINUTE_45 may annotate `quarter to`.
      // Validate those resolved clock meanings at the construction level.
      const literal = record.text.slice(span.start, span.end).toLowerCase();
      const expected = valueOf(span.label);

      if (span.label.startsWith("INTERVAL_") && typeof expected === "number") {
        const shown = /^\d+$/.test(literal) ? Number(literal) : NUMBER_WORDS[literal];

        if (shown !== undefined && shown !== expected) this.#fail(id, "role-contradicts-literal");
      }

      if (span.label.startsWith("WEEKDAY_")) {
        const shown = WEEKDAY_ALIASES[literal] ?? WEEKDAY_PLURAL_ALIASES[literal];

        if (shown !== undefined && shown !== expected) this.#fail(id, "role-contradicts-literal");
      }

      if (span.label.startsWith("UNIT_")) {
        const shown = UNIT_ALIASES[literal];
        const previous = spans[spanIndex - 1];

        const idiom = previous?.label.startsWith("INTERVAL_")
          ? FREQUENCY_IDIOMS[record.text.slice(previous.start, span.end)]
          : undefined;

        const equivalentPhrase =
          idiom !== undefined &&
          idiom.unit === expected &&
          idiom.interval === valueOf(previous!.label) &&
          previous!.actionable === span.actionable;

        if (shown !== undefined && shown !== expected && !equivalentPhrase) {
          this.#fail(id, "role-contradicts-literal");
        }
      }

      if (span.label.endsWith("_LIT") && NUMBER_WORDS[literal] !== undefined) {
        this.#fail(id, "unrepresentable-spelled-literal");
      }
    }

    const tokens = tokenize(record.text);

    if (!tokens.length || tokens.length > MAX_TOKENS) this.#fail(id, "token-limit");
    let tokenSignature = "invalid";

    try {
      const labeled = labeledTokensFromGold(record.text, spans);

      tokenSignature = JSON.stringify(labeled.map((t) => [t.role, t.boundary, t.actionable]));
      const boundaries = new Set(tokens.flatMap((t) => [t.start, t.end]));

      if (spans.some((s) => !boundaries.has(s.start) || !boundaries.has(s.end))) {
        this.#fail(id, "span-not-token-aligned");
      }
    } catch {
      this.#fail(id, "token-coverage");
    }

    const gold = { ...record, source: "kinds-sample" as const, speechAct: act };

    if (population === "E") {
      if (record.intent !== null || act !== "none" || hasSchedule) {
        this.#fail(id, "invalid-easy-negative");
      }

      if (hasTimeWord(record.text) || timeShapedNumber(record.text)) {
        this.#fail(id, "schedule-language-in-easy-negative");
      }
    } else if (record.intent === null || !hasSchedule) {
      this.#fail(id, "missing-schedule-gold");
    } else {
      try {
        assertValid(record.intent);
      } catch {
        this.#fail(id, "invalid-intent");
      }

      if (request) {
        if (!oracleAgrees(oracleDecode(gold), record.intent)) {
          this.#fail(id, "gold-decode-mismatch");
        }
      } else {
        try {
          decodeIntent(
            labeledTokensFromGold(record.text, spans),
            undefined,
            record.text,
            "off",
            act,
          );

          this.#fail(id, "negative-accepted");
        } catch (error) {
          if (!(error instanceof UnsafeScheduleContextError)) {
            this.#fail(id, "negative-not-semantic-refusal");
          }
        }

        // Refusal alone can hide zeroed/corrupted roles. Check the mentioned schedule too.
        const inner = {
          ...gold,
          speechAct: "request" as const,
          spans: spans.map((span) => ({
            ...span,
            actionable: isScheduleLabel(span.label),
          })),
        };

        if (!oracleAgrees(oracleDecode(inner), record.intent)) {
          this.#fail(id, "negative-inner-gold-mismatch");
        }
      }

      try {
        const result = compile(record.intent);

        if (result.supported) report.compiler.supported++;
        else bump(report.compiler.unsupported, result.code);
      } catch {
        this.#fail(id, "compile-invalid-intent");
      }

      const i = record.intent;

      const value = (key: string, item: string | number): void => {
        bump((report.values[key] ??= {}), String(item));
      };

      value("unit", i.frequency.unit);
      value(`interval/${i.frequency.unit}`, i.frequency.interval);

      if (i.time) {
        if (i.time.hour !== undefined) value("hour", i.time.hour);
        value("minute", i.time.minute);
      }

      for (const day of i.weekdays ?? []) value("weekday", day);
      if (i.dayOfMonth) value("day-of-month", i.dayOfMonth);

      if (i.ordinalWeekday) {
        value("ordinal", i.ordinalWeekday.ordinal);
        value("ordinal-weekday", i.ordinalWeekday.weekday);
      }

      if (i.day) value("day-rule", i.day.type);
      if (i.bounds?.start) value("bounds", "start");
      if (i.bounds?.end) value("bounds", "end");
      if (i.exceptions?.dates?.length) value("exceptions", "dates");
      if (i.exceptions?.weekdays?.length) value("exceptions", "weekdays");
    }

    if (
      act === "correction" &&
      spans.some((s) => isScheduleLabel(s.label) && !s.actionable) &&
      spans.some((s) => s.actionable)
    ) {
      report.explicitCorrections++;
    }

    const fingerprint = hasSchedule
      ? fingerprintExample({
          text: record.text,
          spans: spans.map((s) => ({ ...s, actionable: s.actionable ?? false })),
          speechAct: act,
        })
      : sha256Hex(`negative\0${record.text}`);

    if (record.fingerprint !== fingerprint) this.#fail(id, "incorrect-fingerprint");
    const textHash = sha256Hex(corpusTextKey(record.text));

    const signature = sha256Hex(
      JSON.stringify({
        intent: record.intent && canonicalizeIntent(record.intent),
        act,
        tokens: tokenSignature,
      }),
    );

    const textOwner = this.#db
      .query<Owner, [string]>("SELECT split,signature FROM texts WHERE hash=?")
      .get(textHash);

    if (textOwner?.split !== undefined && textOwner.split !== record.split) {
      this.#fail(id, "text-crosses-splits");
    }

    if (textOwner && textOwner.signature !== signature) this.#fail(id, "contradictory-supervision");

    this.#db
      .query("INSERT INTO texts VALUES (?,?,?,1) ON CONFLICT(hash) DO UPDATE SET n=n+1")
      .run(textHash, record.split, signature);

    const fpOwner = this.#db
      .query<{ split: string }, [string]>("SELECT split FROM fingerprints WHERE hash=?")
      .get(fingerprint);

    if (fpOwner && fpOwner.split !== record.split) this.#fail(id, "fingerprint-crosses-splits");

    this.#db
      .query("INSERT OR IGNORE INTO fingerprints VALUES (?,?)")
      .run(fingerprint, record.split);

    if (Object.values(report.violations).reduce((a, b) => a + b, 0) === before) {
      report.validRecords++;
    }

    if (report.records % 10_000 === 0) this.#db.exec("COMMIT; BEGIN;");
  }

  finish(): AuditReport {
    if (this.#finished) return this.report;
    this.#finished = true;
    const r = this.report;

    if (!r.records) this.#fail("corpus", "empty-corpus");

    if (r.requirements.mix) {
      for (const split of SPLITS) {
        const c = r.splits[split];

        if (!c.R || !c.H || !c.E) this.#fail(split, "empty-split-population");
        for (const message of bandFailures(sharesOf(c))) this.#fail(split, `quota:${message}`);
      }
    }

    const texts = this.#db
      .query<{ n: number; max: number }, []>(
        "SELECT COUNT(*) AS n, COALESCE(MAX(n),0) AS max FROM texts",
      )
      .get()!;

    r.uniqueTexts = texts.n;
    r.maxTextRepetitions = texts.max;
    r.repeatedRows = r.populations.R + r.populations.H + r.populations.E - texts.n;

    r.uniqueFingerprints = this.#db
      .query<{ n: number }, []>("SELECT COUNT(*) AS n FROM fingerprints")
      .get()!.n;

    r.groups = this.#db.query<{ n: number }, []>("SELECT COUNT(*) AS n FROM groups").get()!.n;
    this.#db.exec("COMMIT;");
    r.complete = true;
    r.passed = Object.keys(r.violations).length === 0;

    return r;
  }

  close(): void {
    this.#db.close();
  }
}

export async function auditCorpus(
  input: string,
  options: AuditOptions = {},
): Promise<AuditReport & { input: string; sha256: string }> {
  const scratch = mkdtempSync(join(tmpdir(), "syntaxless-corpus-audit-"));

  const audit = new CorpusAudit({
    ...options,
    indexPath: options.indexPath ?? join(scratch, "index.sqlite"),
  });

  const hash = createHash("sha256");
  const stream = createReadStream(input);

  stream.on("data", (chunk) => hash.update(chunk));
  const lines = createInterface({ input: stream, crlfDelay: Infinity });

  try {
    for await (const line of lines) {
      if (!line.trim()) continue;
      let row: unknown;

      try {
        row = JSON.parse(line);
      } catch {
        row = null;
      }

      audit.add(row);

      if (audit.report.records % 100_000 === 0) {
        process.stderr.write(`audited ${audit.report.records.toLocaleString()} rows\n`);
      }
    }

    return { ...audit.finish(), input: resolve(input), sha256: hash.digest("hex") };
  } finally {
    lines.close();
    stream.destroy();
    audit.close();
    rmSync(scratch, { recursive: true, force: true });
  }
}
