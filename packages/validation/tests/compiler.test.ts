import { expect, test } from "bun:test";

import { compile } from "../src/index.ts";
import type { FieldIntent, JsonSchema } from "../src/index.ts";

const supported: Array<{ name: string; intent: FieldIntent; schema: JsonSchema }> = [
  {
    name: "integer exclusive min 10 widens to minimum 11",
    intent: {
      type: "integer",
      required: true,
      minimum: { value: 10, inclusive: false },
    },
    schema: { type: "integer", minimum: 11 },
  },
  {
    name: "number exclusive min 10 emits exclusiveMinimum 10",
    intent: {
      type: "number",
      required: true,
      minimum: { value: 10, inclusive: false },
    },
    schema: { type: "number", exclusiveMinimum: 10 },
  },
  {
    name: "integer exclusive max 10 widens to maximum 9",
    intent: {
      type: "integer",
      required: true,
      maximum: { value: 10, inclusive: false },
    },
    schema: { type: "integer", maximum: 9 },
  },
  {
    name: "number exclusive max 10 emits exclusiveMaximum 10",
    intent: {
      type: "number",
      required: true,
      maximum: { value: 10, inclusive: false },
    },
    schema: { type: "number", exclusiveMaximum: 10 },
  },
  {
    name: "uuid uses format uuid",
    intent: { type: "uuid", required: false },
    schema: { type: "string", format: "uuid" },
  },
  {
    name: "url uses the standard uri format",
    intent: { type: "url", required: true },
    schema: { type: "string", format: "uri", minLength: 1 },
  },
];

for (const { name, intent, schema } of supported) {
  test(name, () => {
    expect(compile(intent)).toEqual({ supported: true, schema });
  });
}
