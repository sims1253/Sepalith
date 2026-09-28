# DAT-10 provider parse-unavailable preparation

This packet prepares two CPU-only actions for the lead:

1. A narrow provider policy change for malformed or unfinished R editor
   snapshots.
2. A fresh, rich-diagnostic retry command for Semantic9534's failed shard
   range 0027--0040.

No campaign renderer, training job, model, optimizer, CUDA device, or existing
campaign output root was launched or modified by this packet.

## Parse-unavailable policy

The exact Noop TRAIN/editor snapshot `e677ee6a8da38436f4bdb6b6` fails in R's
`parse()` at the source helper. A parse failure is legitimate editor input. The
copied helper at `source/source_imports.R` catches only that parse exception and
writes this explicit source-import record:

```text
status: source_import_evidence_unavailable
source_sha256: <hash of the raw UTF-8 snapshot>
target_name: empty
target_span: empty
import_dependencies: []
unresolved_nonimport: []
```

It invents no imported names, target information, or error-derived semantic
evidence. `source/namespace_runtime.ts` validates that status, the source hash,
and the empty dependency list. It returns `status: "unavailable"` while
retaining the package `NAMESPACE` path and SHA. `full_document_policy.ts` can
therefore use its existing context path. Missing Rscript, missing paths,
helper I/O, invalid helper JSON, and invalid namespace evidence remain
infrastructure failures. The renderer carries the explicit unavailable field
only for this new status; valid and no-following-function outputs retain the
existing shape.

The copied source closure is independent of the frozen packet. The only source
behavior edits are the parse-status branch, the validated runtime union, the
local closure imports, and the renderer's unavailable-status pass-through.

## Evidence and tests

The packet retains only the two approved target-free row fixtures:

| fixture | source | SHA256 |
| --- | --- | --- |
| e677 pre-edit | `inputs/e677-preedit.jsonl` | `1c1cb54909b08296cf09c0980de1cd18210e86d3d7b8e2abbe3808e650531b53` |
| valid control | `inputs/valid-control.jsonl` | `299c4ed67dee4ffb354821a4e621335de2975bf583844a43c206fc04a5a5a41a` |

`tests/test_parse_unavailable.py` ran the real copied R helper and renderer and
passed:

- exact e677 structured source-evidence absence, including source SHA and
  empty dependencies;
- exact e677 full-document fallback with `supported` output;
- missing Rscript remaining infrastructure-fatal;
- valid-control output parity with the existing Noop wrapper;
- cursor geometry and `source_import_inventory_no_following_function` byte
  parity with the frozen helper.

The local TypeScript runtime modules also parsed successfully under Node's
strip-types loader. A TypeScript compiler is not installed in this environment,
so `npx --no-install tsc` was unavailable; the real renderer executions cover
module loading and the exercised runtime paths.

The e677 fallback result records `source_inventory_status` as
`source_import_evidence_unavailable`, `observed_dependencies` as `[]`, and
`selection_target_or_gold_used` as `false`. Its retained package namespace SHA
is `db83b08bde02edd95882df56e00a7c8665f59cc988907e188a986154ec333e69`.

## Semantic9534 retry preparation

`semantic9534-retry/` contains the exact Semantic9535 renderer/module closure,
the rich copied namespace runtime, and a source manifest:

```text
source-manifest.json SHA256:
9725d968382c98fe5313985f5407fd6aa01a3e34063393b88c3bfb45193572e8
```

The launch-ready command is
`semantic9534-retry/commands/retry_14_shards.sh`. It uses the same input root,
tokenizer bridge, tokenizer JSON, context size 16,384, and generation reserve
2,048 as the failed run, but writes only to this fresh root:

```text
/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1
```

It assigns 0027, 0029, 0031, 0033, 0035, 0037, 0039 to core 8 and 0028,
0030, 0032, 0034, 0036, 0038, 0040 to core 10. Each shard writes a terminal
record with helper phase, error name/message, code, signal, killed flag,
stdout, and stderr. A failed shard remains failed; the lane continues so all
14 terminal records are available, and the aggregate command still exits
nonzero. The script refuses the preserved old root
`render-16k-v1` and requires fresh per-shard output paths.

The earlier isolated Semantic9535 0027 and 0028 full-render probes both passed
with one supported row each. That makes a transient or concurrency-sensitive
failure plausible, but does not prove the 14-shard retry will pass. The retry
has not been launched here.

## Train/eval/serving parity

For new data and any model trained on it, training, evaluation, and serving
must share this provider status contract and source closure. In particular,
all three must preserve the raw source SHA and package `NAMESPACE` evidence,
represent parse-unavailable imports as an empty set, and use the same context
fallback. Mixing this provider with a fatal-parse evaluator or a serving path
that guesses imports changes the input distribution and invalidates direct
comparisons. Existing valid and no-following-function behavior remains bound
to the original helper outputs.

## Scope and remaining uncertainty

The preparation is CPU-only and uses no labels, completions, target text, or
gold fields. The parse fix is a static candidate verified on the exact malformed
fixture and controls. It changes hold accounting for future malformed
snapshots, so the lead must bind its source/runtime hashes before admitting new
data. The Semantic retry's actual failure distribution, wall time, and merged
row count remain unknown until the lead launches the fresh 14-shard command.
