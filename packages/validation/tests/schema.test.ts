import { expect, test } from "bun:test";

import { schema } from "../src/index.ts";

test("schema returns the compiled schema and its intent", async () => {
  const result = await schema("required email with max 100 characters");

  expect(result.supported).toBe(true);

  if (result.supported) {
    expect(result.intent).toEqual({ type: "email", required: true, maxLength: 100 });

    expect(result.schema).toEqual({
      type: "string",
      format: "email",
      minLength: 1,
      maxLength: 100,
    });
  }
});

test("schema refuses an off-domain schedule sentence with a null intent", async () => {
  const result = await schema("every weekday at 9");

  expect(result.supported).toBe(false);

  if (!result.supported) {
    expect(result.intent).toBeNull();
    expect(result.code).toBe("unrecognized");
  }
});

test("schema preserves a valid intent when nullable compilation is unsupported", async () => {
  const result = await schema("required nullable email", { source: "rules" });

  expect(result).toMatchObject({
    supported: false,
    code: "nullable",
    intent: { type: "email", required: true, nullable: true },
  });
});
