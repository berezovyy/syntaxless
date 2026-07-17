/** Selection metadata is side information, never an input feature or target.
 * Rows are emitted only after tokenization/alignment succeeds, in shard order.
 */
import { createHash } from "node:crypto";
import { writeSync } from "node:fs";

import { isScheduleLabel } from "../../src/model/labels.js";
import { SPEECH_ACTS } from "../../src/model/speech-acts.js";
import type { CorpusRecord } from "./corpus/merge.js";
import { corpusTextKey } from "./corpus/production/protected-text.js";
import type { PerRecord } from "./featurize.js";
import { sha256Hex } from "./fingerprint.js";

export const METADATA_SUFFIX = ".metadata.jsonl";

export interface SequenceMetadataManifest {
  version: 1;
  suffix: typeof METADATA_SUFFIX;
  rows: number;
  bytes: number;
  sha256: string;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);

  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, canonical(item)]),
    );
  }

  return value;
}

export function sequenceMetadata(
  record: CorpusRecord,
  shard: PerRecord,
  row: number,
  population: "R" | "H" | "E",
) {
  const speechAct = SPEECH_ACTS[shard.speech[0]!]!;
  const template = record.template ?? `${record.family}/unspecified`;
  const construction = record.provenance?.constructionId ?? template;

  const roles = Object.keys(shard.labelCounts)
    .filter((label) => label !== "O" && shard.labelCounts[label]! > 0)
    .sort();

  const content = record.spans.filter((span) => isScheduleLabel(span.label));
  const clauses = content.filter((span) => span.clauseStart).length;

  const kernel = content.map((span) => [
    span.label,
    record.text.slice(span.start, span.end),
    span.clauseStart,
  ]);

  const capabilities = [
    `family:${record.family}`,
    `kind:${template.split("/")[0]}`,
    `speech:${speechAct}`,
    `clauses:${clauses}`,
    ...roles.map((role) => `role:${role}`),
    ...(record.intent === null
      ? ["intent:none"]
      : Object.keys(record.intent)
          .sort()
          .map((key) => `intent:${key}`)),
  ];

  const valueTags = [
    ...new Set(
      content
        .filter((span) => span.label.endsWith("_LIT"))
        .map((span) => `value:${span.label}:${record.text.slice(span.start, span.end)}`),
    ),
  ].sort();

  let supervisedTokens = 0;
  let actionableTokens = 0;

  for (let index = 0; index < shard.tokenCount; index++) {
    if (shard.kinds[index] === 3) continue;
    supervisedTokens++;
    actionableTokens += Number(shard.actionable[index] === 1);
  }

  return {
    version: 1,
    row,
    split: record.split,
    id: record.id,
    textSha256: sha256Hex(corpusTextKey(record.text)),
    fingerprint: record.fingerprint,
    source: record.provenance?.sourceId ?? record.source,
    provenance: record.provenance ?? null,
    family: record.family,
    template,
    construction,
    population,
    speechAct,
    // Literal values stay in the kernel. Speech/actionability stay in the stratum.
    kernel: sha256Hex(JSON.stringify(canonical({ intent: record.intent, content: kernel }))),
    structure: sha256Hex(
      JSON.stringify(content.map((span) => [span.label, span.clauseStart, span.actionable])),
    ),
    stratum: sha256Hex(
      JSON.stringify([
        population,
        record.provenance?.sourceId ?? record.source,
        template.split("/")[0],
        speechAct,
        roles,
        clauses,
        actionableTokens > 0,
      ]),
    ),
    capabilities,
    valueTags,
    inputTokens: shard.tokenCount,
    supervisedTokens,
    actionableTokens,
    labelCounts: Object.fromEntries(
      Object.entries(shard.labelCounts).filter(([, count]) => count > 0),
    ),
  };
}

export type SequenceMetadata = ReturnType<typeof sequenceMetadata>;

/** Buffer metadata; the featurizer owns opening and closing the supplied file. */
export class SequenceMetadataWriter {
  readonly #fd: number;
  readonly #hash = createHash("sha256");
  #pending = "";
  #rows = 0;
  #bytes = 0;

  constructor(fd: number) {
    this.#fd = fd;
  }

  append(record: SequenceMetadata): void {
    if (record.row !== this.#rows) throw new Error("metadata row order differs from shard order");
    this.#pending += `${JSON.stringify(record)}\n`;
    this.#rows++;
    if (this.#pending.length >= 1024 * 1024) this.#flush();
  }

  #flush(): void {
    const buffer = Buffer.from(this.#pending, "utf8");
    let offset = 0;

    while (offset < buffer.length) {
      offset += writeSync(this.#fd, buffer, offset, buffer.length - offset);
    }

    this.#hash.update(buffer);
    this.#bytes += buffer.length;
    this.#pending = "";
  }

  finish(): SequenceMetadataManifest {
    this.#flush();

    return {
      version: 1,
      suffix: METADATA_SUFFIX,
      rows: this.#rows,
      bytes: this.#bytes,
      sha256: this.#hash.digest("hex"),
    };
  }
}
