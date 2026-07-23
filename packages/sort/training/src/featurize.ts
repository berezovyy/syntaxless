import { writeShards } from "../../../shared/training/binary.js";
import type { Sequence } from "../../../shared/training/binary.js";
import { LABELS } from "../../src/labels.js";
import {
  EMBEDDING_ROWS,
  FEATURE_COUNT,
  featureRows,
  PADDING_ROW,
  tokenize,
} from "../../src/model/tokenizer.js";
import { prepare } from "../../src/prepare.js";
import type { Row } from "./corpus.js";

function* sequences(records: readonly Row[]): Generator<Sequence> {
  for (const record of records) {
    if (record.trainingEligible === false) throw Error("evaluation-only rows cannot be featurized");

    const prepared = prepare(record.text, record.context);
    const tokens = tokenize(prepared.text);

    if (tokens.length > 128) throw Error("training sequence exceeds 128 tokens");

    const rows = new Uint16Array(tokens.length * FEATURE_COUNT).fill(PADDING_ROW);
    const labels = new Uint8Array(tokens.length).fill(255);
    const boundaries = new Uint8Array(tokens.length);
    const kinds = new Uint8Array(tokens.length);
    const neighbors = new Int16Array(tokens.length * 2).fill(-1);

    const nonspace = tokens.map((token, i) => (token.kind === 3 ? -1 : i)).filter((i) => i >= 0);
    const lookup = new Map(prepared.lexemes.map((lexeme, i) => [lexeme.start, i]));

    for (const [index, token] of tokens.entries()) {
      rows.set(featureRows(token), index * FEATURE_COUNT);
      const role = token.kind === 3 ? undefined : record.roles[lookup.get(token.start)!];

      if (token.kind !== 3 && role === undefined) throw Error("unaligned role supervision");

      if (role !== undefined) {
        const label = LABELS.indexOf(role);

        if (label < 0) throw Error("unknown training role");
        labels[index] = label;
      }

      boundaries[index] = Number(role === "FIELD");
      kinds[index] = token.kind;
      neighbors[index * 2] = nonspace.findLast((i) => i < index) ?? -1;
      neighbors[index * 2 + 1] = nonspace.find((i) => i > index) ?? -1;
    }

    yield {
      rows,
      labels,
      boundaries,
      kinds,
      neighbors,
      fingerprint: record.fingerprint,
      frame: record.family,
    };
  }
}

export function featurize(records: readonly Row[], prefix: string, split?: Row["split"]): void {
  const owners = new Map<string, Row["split"]>();

  for (const record of records) {
    if (!["train", "validation", "heldout"].includes(record.split)) {
      throw Error("invalid corpus split");
    }

    const owner = owners.get(record.fingerprint);

    if (owner !== undefined && owner !== record.split) {
      throw Error("language group crosses corpus splits");
    }

    owners.set(record.fingerprint, record.split);
  }

  writeShards(prefix, sequences(split ? records.filter((row) => row.split === split) : records), {
    labels: LABELS,
    paddingLabel: 255,
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
  });
}
