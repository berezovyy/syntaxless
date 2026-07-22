import { expect, test } from "bun:test";

import { compile, compileObject, explain, parse, schema } from "../src/index.ts";
import { dispose, getEngine } from "../src/model/engine.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { fixtureArtifact } from "./helpers/artifact.ts";

test("required properties are enforced by an enclosing object, including hostile names", () => {
  const fields = JSON.parse(
    '{"email":{"type":"email","required":true},"__proto__":{"type":"boolean","required":false}}',
  );

  const result = compileObject(fields);

  expect(result).toEqual({
    supported: true,
    schema: {
      type: "object",
      properties: {
        email: { type: "string", format: "email", minLength: 1 },
        ["__proto__"]: { type: "boolean" },
      },
      required: ["email"],
    },
  });

  expect(compileObject({ bad: { type: "boolean", required: true, minLength: 1 } })).toMatchObject({
    supported: false,
    code: "wrong-slot",
    field: "bad",
  });
});

test("compiler rejects empty required strings and discrete integer intervals", () => {
  expect(
    compile({
      type: "integer",
      required: true,
      minimum: { value: Number.MAX_SAFE_INTEGER, inclusive: false },
    }),
  ).toMatchObject({ supported: false, code: "invalid-bound" });

  expect(compile({ type: "string", required: false, maxLength: 0 })).toEqual({
    supported: true,
    schema: { type: "string", maxLength: 0 },
  });

  expect(compile({ type: "string", required: true, maxLength: 0 })).toMatchObject({
    supported: false,
    code: "conflicting-bounds",
  });

  expect(
    compile({
      type: "integer",
      required: true,
      minimum: { value: -1.5, inclusive: false },
      maximum: { value: 2.5, inclusive: true },
    }),
  ).toEqual({ supported: true, schema: { type: "integer", minimum: -1, maximum: 2 } });

  expect(
    compile({
      type: "integer",
      required: true,
      minimum: { value: 0.1, inclusive: true },
      maximum: { value: 0.9, inclusive: true },
    }),
  ).toMatchObject({ supported: false, code: "conflicting-bounds" });
});

test("signed decimals and strict limits preserve exact meaning, questions refuse", async () => {
  const result = await parse("required number more than -2.5 and at most 3.25");

  expect(result).toMatchObject({
    ok: true,
    intent: {
      type: "number",
      required: true,
      minimum: { value: -2.5, inclusive: false },
      maximum: { value: 3.25, inclusive: true },
    },
  });

  expect(await parse("required string exactly 0 characters")).toMatchObject({
    ok: false,
    code: "conflicting-bounds",
  });

  expect(await parse("optional string exactly 0 characters")).toMatchObject({
    ok: true,
    intent: { type: "string", required: false, minLength: 0, maxLength: 0 },
  });

  expect(await parse("required email?")).toMatchObject({ ok: false });

  expect(await parse("required number at least 9007199254740993")).toMatchObject({
    ok: false,
  });

  expect(
    await schema("required email", { source: "model", weights: new Uint8Array([0]) }),
  ).toMatchObject({ supported: false, code: "incompatible-model", intent: null });
});

test("explanation preserves every bound and labels the nonempty default", () => {
  expect(explain({ type: "email", required: true, maxLength: 100 })).toBe(
    "Required email field; at least 1 character (required-string default); at most 100 characters.",
  );

  expect(
    explain({
      type: "number",
      required: false,
      minimum: { value: -3, inclusive: false },
      maximum: { value: 10, inclusive: true },
    }),
  ).toBe("Optional number field; greater than -3; at most 10.");
});

test(
  "custom engines use their own tensors, " + "reject corrupt geometry, and survive disposal",
  async () => {
    dispose();
    const artifact = JSON.parse(JSON.stringify(fixtureArtifact()));

    const first = await getEngine({
      weights: new TextEncoder().encode(JSON.stringify(artifact)),
    });

    const bias = artifact.segments.find((s: { name: string }) => s.name === "role_head.bias");
    const q = artifact.q as string;

    const old = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".indexOf(
      q[bias.offset]!,
    );

    artifact.q = q.slice(0, bias.offset) + (old === 0 ? "B" : "A") + q.slice(bias.offset + 1);
    artifact.confidenceFloor = 0.75;
    const bytes = new TextEncoder().encode(JSON.stringify(artifact));
    const pending = getEngine({ weights: bytes });

    expect(getEngine({ weights: bytes })).toBe(pending);
    const second = await pending;
    const tokens = tokenize("required email");
    const before = first.inferTokens(tokens, true).logits!;
    const after = second.inferTokens(tokens, true).logits!;

    expect(second.confidenceFloor).toBe(0.75);
    expect(Math.abs(before[0]! - after[0]!)).toBeGreaterThan(0.001);
    expect(first.inferTokens(tokens, true).logits).toEqual(before);
    const nodeBytes = Buffer.from(JSON.stringify(artifact));
    const snapshot = getEngine({ weights: nodeBytes });

    nodeBytes.fill(0);
    expect((await snapshot).inferTokens(tokens, true).logits).toEqual(after);
    dispose();
    expect(second.inferTokens(tokens, true).logits).toEqual(after);
    artifact.segments[0].shape[1] += 1;

    await expect(
      getEngine({ weights: new TextEncoder().encode(JSON.stringify(artifact)) }),
    ).rejects.toThrow();
  },
);
