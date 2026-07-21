import { expect, test } from "bun:test";

import { validateComparison, validateField } from "../src/compiler/validate.ts";
import type { FieldIntent, FieldType } from "../src/index.ts";

test("min exclusive and max inclusive at the same value is an empty interval", () => {
  const failure = validateComparison({
    min: { value: 10, inclusive: false },
    max: { value: 10, inclusive: true },
  });

  expect(failure).toMatchObject({ supported: false, code: "conflicting-bounds" });
});

test("exact inclusive min=max is valid", () => {
  expect(
    validateComparison({
      min: { value: 10, inclusive: true },
      max: { value: 10, inclusive: true },
    }),
  ).toBeNull();
});

const lengthOnWrongType: Array<{ type: FieldType; extra: Partial<FieldIntent> }> = [
  { type: "boolean", extra: { minLength: 1 } },
];

for (const { type, extra } of lengthOnWrongType) {
  test(`length on ${type} is wrong-slot`, () => {
    const failure = validateField({ type, required: true, ...extra } as FieldIntent);

    expect(failure).toMatchObject({ supported: false, code: "wrong-slot" });
  });
}

const valueOnWrongType: Array<{ type: FieldType }> = [{ type: "string" }];

for (const { type } of valueOnWrongType) {
  test(`value bounds on ${type} are wrong-slot`, () => {
    const failure = validateField({
      type,
      required: true,
      minimum: { value: 1, inclusive: true },
    });

    expect(failure).toMatchObject({ supported: false, code: "wrong-slot" });
  });
}

test("minLength greater than maxLength is conflicting-bounds", () => {
  const failure = validateField({
    type: "string",
    required: true,
    minLength: 20,
    maxLength: 5,
  });

  expect(failure).toMatchObject({ supported: false, code: "conflicting-bounds" });
});

test("unknown type fails", () => {
  const failure = validateField({ type: "date" as FieldType, required: true });

  expect(failure).toMatchObject({ supported: false, code: "invalid-type" });
});

test("non-integer and negative lengths are invalid-length", () => {
  expect(validateField({ type: "string", required: true, minLength: 1.5 })).toMatchObject({
    supported: false,
    code: "invalid-length",
  });

  expect(validateField({ type: "string", required: true, maxLength: -1 })).toMatchObject({
    supported: false,
    code: "invalid-length",
  });
});

test("non-finite bound values are invalid-bound", () => {
  expect(validateComparison({ min: { value: Number.NaN, inclusive: true } })).toMatchObject({
    supported: false,
    code: "invalid-bound",
  });

  expect(
    validateComparison({ max: { value: Number.POSITIVE_INFINITY, inclusive: true } }),
  ).toMatchObject({
    supported: false,
    code: "invalid-bound",
  });
});
