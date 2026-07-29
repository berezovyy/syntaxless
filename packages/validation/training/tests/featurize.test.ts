import { expect, test } from "bun:test";

import { labelFromId } from "../../src/model/labels.ts";
import { tokenize } from "../../src/model/tokenizer.ts";
import { featurizeRecord, PADDING_LABEL } from "../src/featurize.ts";
import type { CorpusRecord } from "../src/generate.ts";
import { renderField } from "../src/renderer.ts";

test("featurize assigns gold labels to a rendered email constraint", () => {
  const rendered = renderField({ type: "email", required: true, maxLength: 100 }, undefined, {
    order: "presence-type-glue-comparison",
    typeSurface: "email",
    presenceSurface: "required",
    maxSurface: "max",
    unitSurface: "characters",
    glueSurface: "with",
    numberStyle: "digits",
    template: "email-required-maxlength/manifesto",
  });

  const record: CorpusRecord = {
    id: "train-000000",
    split: "train",
    seed: 1,
    family: "constraint",
    template: rendered.template,
    fingerprint: "fp",
    text: rendered.text,
    intent: { type: "email", required: true, maxLength: 100 },
    spans: rendered.spans,
  };

  const shard = featurizeRecord(record, tokenize(record.text));
  const roles = [...shard.labels].filter((id) => id !== PADDING_LABEL);

  expect(roles.map(labelFromId)).toEqual([
    "REQUIRED",
    "TYPE_EMAIL",
    "GLUE",
    "LTE",
    "NUMBER",
    "UNIT_CHARACTER",
  ]);
});
