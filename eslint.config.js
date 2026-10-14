// Layout rules the formatter can't express: bun fmt keeps blank lines but never adds them, never
// adds braces and never splits strings. Everything here is about reading the code, not about
// behaviour, so `bun run lint:fix` settles all of it except long lines, which need rewriting by hand.
import { fileURLToPath } from "node:url";

import { includeIgnoreFile } from "@eslint/compat";
import stylistic from "@stylistic/eslint-plugin";
import tseslint from "typescript-eslint";

const multiline = [
  "multiline-const",
  "multiline-let",
  "multiline-var",
  "multiline-expression",
  "multiline-export",
  "multiline-type",
  "block-like",
  "function",
  "class",
  "interface",
  "enum",
];
const declaration = ["singleline-const", "singleline-let", "singleline-var"];

export default [
  includeIgnoreFile(fileURLToPath(new URL(".gitignore", import.meta.url))),
  { ignores: ["packages/*/src/model/weights.gen.ts", "docs/media/**", "eslint.config.js"] },
  {
    files: ["**/*.ts"],
    languageOptions: { parser: tseslint.parser, sourceType: "module" },
    linterOptions: { reportUnusedDisableDirectives: "error" },
    plugins: { "@stylistic": stylistic },
    rules: {
      // A body on its own line gets braces, so every block reads as a block. One-line guards such
      // as `if (!ok) return null;` stay as they are.
      curly: ["error", "multi-line", "consistent"],
      // One declaration per statement: `let low = 0n, high = 1n` wraps into a hard-to-scan column.
      "one-var": ["error", "never"],

      // Later entries win. Multi-line statements, blocks and declarations get a blank line on both
      // sides; runs of one-line declarations, imports, exports and type aliases stay together.
      "@stylistic/padding-line-between-statements": [
        "error",
        { blankLine: "always", prev: "*", next: [...multiline, "return"] },
        { blankLine: "always", prev: ["import", "directive", ...multiline], next: "*" },
        { blankLine: "always", prev: ["const", "let", "var", "using"], next: "*" },
        { blankLine: "any", prev: declaration, next: declaration },
        { blankLine: "any", prev: "import", next: "import" },
        { blankLine: "any", prev: "directive", next: "directive" },
        { blankLine: "any", prev: "singleline-export", next: "singleline-export" },
        { blankLine: "any", prev: "singleline-type", next: "singleline-type" },
      ],
      "@stylistic/lines-between-class-members": [
        "error",
        "always",
        { exceptAfterSingleLine: true },
      ],

      // bun fmt wraps code at 100 columns but can't split strings, templates or comments.
      "@stylistic/max-len": [
        "error",
        { code: 100, tabWidth: 2, ignoreUrls: true, ignoreRegExpLiterals: true },
      ],
    },
  },
];
