# Code quality checks

Run from the repository root with Python 3.10+, uv, and Node 22.18+:

```sh
npm --prefix extensions/vscode-sepalith ci
npm --prefix extensions/vscode-sepalith run lint
python3 scripts/check_quality.py
```

The Python command uses the pinned Ruff and ty environment in `tools/quality`.
It does not install the training stack or change the root uv environment.
The default Python gate covers `packages/sepalith`, including tests and examples.
The TypeScript gate covers the extension's `src` and `scripts` directories.

Run the existing offline checks, including the TypeScript lint rule tests:

```sh
python3 scripts/check_core.py
python3 scripts/check_product.py
```

Both CI workflows enforce their respective gates. A change to either Python
configuration or the quality tool lockfile also triggers the core workflow.

## TypeScript policy

The [upstream anti-slop rules](https://github.com/dmmulroy/anti-slop) are vendored
under `extensions/vscode-sepalith/tools/anti-slop`. Oxlint and `@oxlint/plugins`
are pinned to the same version. TypeScript's existing strict compiler check
remains part of product validation.

The rules reject chained assertions, widening known values, broad dictionaries,
reflection that bypasses typed access, unknown return contracts, module mocking,
and assertions without a nearby `SAFETY:` explanation. Built-in rules also reject
explicit `any`, unused code, `@ts-ignore`, and `@ts-nocheck`; `@ts-expect-error`
requires a description.

Local differences from upstream:

- Runtime `typeof` checks remain legal. This package validates HTTP JSON and LSP
  responses directly, without a schema library.
- Names containing `shape` remain legal; naming alone is not a correctness test.
- Unknown parameters remain legal in `runtime.ts`, `context_build.ts`, and test
  scripts, where parsing and deliberately malformed fixtures need them. The
  completion parser has a single boundary exception. Other source files enforce
  the upstream rule; an error formatter uses the supported `cause` convention.
- The LSP record predicate has one documented dictionary exception. It leaves
  fields unknown until the normalizer checks them.
- Effect rules are omitted because this package does not use Effect.

Do not add a safety comment merely to silence a warning. Parse external data,
keep inferred types, or state the actual invariant that makes an assertion valid.
The completion parser replaces the previous unchecked server-response casts.

## Python policy

[Ruff rules](https://docs.astral.sh/ruff/rules/) enforce valid names and syntax,
annotated function contracts, no explicit `Any` parameters or returns, common
bug patterns, selected redundant branches, exception handling, and specific,
non-stale suppressions. [ty](https://docs.astral.sh/ty/reference/rules/) checks
type consistency and treats redundant casts, blanket ignores, stale ignores,
and possibly unresolved references as errors. Warnings fail the gate too.

This is a related policy, not a port of the TypeScript plugin. Ruff and ty do not
reject every use of `Any` inside containers, all untyped functions, reflection,
or module mocking. Python's `isinstance` and boundary inputs typed as `object`
remain valid.

Existing exceptions are explicit:

- The runner and unittest helpers are exempt from missing-annotation rules.
  The runner has a separate migration in progress; its existing dynamic API is
  not rewritten by this tooling change. ty and the other Ruff rules still run.
- `_json_copy` validates arbitrary JSON metadata recursively and retains its
  existing `Any` return contract with one `ANN401` suppression.
- Negative tests that intentionally violate typed contracts use specific
  `ty: ignore[invalid-argument-type]` comments with explanations.

New package modules inherit the full annotation policy. Remove exceptions as
those modules gain precise contracts; do not replace errors with broad casts.

## Research audit

```sh
python3 scripts/check_quality.py --research
```

This applies the same rules to `experiments` and `scripts`, runs both tools even
if Ruff fails, and exits nonzero for findings. It is an opt-in audit, not a CI gate.
Expect existing annotation and lint findings. The isolated tool environment lacks
training dependencies, so ty also reports unresolved third-party imports. For an
import-complete investigation, run the pinned ty binary with `--python` pointing
to the relevant research environment. Do not suppress those imports globally.

## Updating the tools

Update `oxlint` and `@oxlint/plugins` together, using exact matching versions,
and regenerate the npm lockfile. Update the pinned Python dependencies in
`tools/quality/pyproject.toml`, then run `uv lock --project tools/quality`.
Run the quality, core, and product checks after an upgrade. Review the vendored
upstream revision and run `npm --prefix extensions/vscode-sepalith run test:lint`
when changing the plugin.
