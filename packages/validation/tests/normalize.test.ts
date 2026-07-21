import { expect, test } from "bun:test";

import { normalize } from "../src/parser/normalize.ts";

test('normalize("Required E-mail") folds case and keeps the hyphen', () => {
  expect(normalize("Required E-mail")).toBe("required e-mail");
});

test(
  "normalize is NFKC, collapses whitespace, " + "and strips trailing .! and preserves questions",
  () => {
    expect(normalize("Café")).toBe("café");
    expect(normalize("max   100")).toBe("max 100");
    expect(normalize("required email.")).toBe("required email");
    expect(normalize("required email!")).toBe("required email");
    expect(normalize("required email?")).toBe("required email?");
  },
);
