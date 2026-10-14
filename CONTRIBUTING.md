# Contributing

Thanks for helping improve Syntaxless. This guide covers setup, the checks every
change must pass, commit conventions and the rules for evaluation data.

## Setup

Requirements: Bun 1.4.3 or newer. Training and Python tests also need Python 3.13
and [uv](https://docs.astral.sh/uv/).

Formatting needs `bun fmt`, which so far ships only in canary builds
(`1.4.4-canary.1` or newer). `bun run format` and `bun run check` fetch that
pinned canary with `bunx`, so they work on stable Bun. To run `bun fmt`
directly, install a canary with `bun upgrade --canary`.

With [Nix](https://nixos.org/), `nix develop` (or `direnv allow`) gives you all of
this, pinned in `flake.nix`: the same Bun canary, Node.js 22, Python 3.13 and uv.
Linux and Apple Silicon macOS are supported.

```sh
bun install --frozen-lockfile
uv sync --project training/cron --locked   # only for Python tests or training
```

## Checks

Before opening a pull request, run:

```sh
bun run check         # formatting, lint, types, TypeScript tests, builds and size gates
bun run test:python   # when you touch training/ or a package's training workspace
```

When you change a published package, also run its packed-tarball and browser
checks:

```sh
bun run --cwd packages/<package> check:package
bun run --cwd packages/<package> test:browser
```

Format code and docs with `bun run format`, and fix layout lint (blank lines between
steps, braces on multi-line `if` bodies) with `bun run lint:fix`; lines over 100
columns have to be rewritten by hand. Test expectations follow
[TESTING.md](TESTING.md): independent expected values, a positive control for
every refusal, and no tests that accept any outcome.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) with the package
as the scope:

```
feat(filter): support "between" ranges
fix(retry): cap jitter at the configured maximum
docs(format): document currency display options
test(sort): cover descending nulls
refactor(training): share the int6 exporter
```

Use the `-training` suffix for a package's training workspace (for example
`feat(sort-training): …`), and no scope for repository-wide changes. Mark
breaking changes with `!` (for example `fix(filter)!: …`). Keep each commit
focused and passing on its own.

## Models and generated files

- Never hand-edit `src/model/weights.gen.ts`, consumption archives or other
  generated artifacts. Produce them through `bun run train:<package>` and the
  package's release gate.
- Corpora, run directories and build outputs are ignored by Git. Don't commit
  them, or one-off local reports.
- Every promotion gate hashes the training sources, including `training/shared`
  and `packages/shared`. Editing them invalidates any pending freeze, and the
  next promotion needs a new freeze.

## Evaluation data

Reserved evaluation sets measure fresh language acceptance, so they follow
strict rules:

- Never use reserved or consumed wording for training, checkpoint selection,
  calibration or tuning. Corpus generation excludes it by hash.
- Author a new set only after the candidate is frozen, using the package's
  `evaluation/author.ts`, and never edit a written set. Changed wording is a new
  set with the next number.
- A set is consumed by exactly one gate run. Consumed sets are regression
  evidence only and can't be reused as acceptance.
- Don't edit dataset bytes, `.meta.json` digests or `consumed/` records. Records
  store repository-relative paths, and archives are hash-bound.
- Report results as they are: don't weaken thresholds or relabel failures after
  seeing results.

See each package's `evaluation/README.md` for its protocol.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
