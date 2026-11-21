# Sealed-final protocol v2

**The run is user-gated.** This protocol replaces the lost evaluator for
[C17 step 4](cards/C17-release-prep.md#4-recover-and-freeze-the-sealed-final-protocol).
The user must approve the plan and freeze the final weights and harness before
anyone opens the sealed panel. The results may document release quality.
No training, checkpoint selection, quantization selection, or tuning decision
may depend on them.

## Code and provenance

The recovered [admission guard](../../campaign/work/final-admission-guards/) is
byte-identical to commit `d7f2aa94981d57cd51f5dfd6573e4407e083ff94` on
`origin/t3code/best-token-model-plan`. The
[preparation receipt](../../campaign/receipts/DAT-08-admission-guards-preparation.json)
pins the guard, its tests, and its README. The receipt also records the lost
v1 code pins.

| Lost v1 file | SHA-256 |
| --- | --- |
| `final_constructor.py` | `de9e7303ebcd572b50015f9a90d352de31a3379f75f7b6540e58757e51173d97` |
| `raw_source_case_builder.py` | `2e617f5c14b04a3a329362665d12f18ff836f88a4f029719a9a5d0ab0e93999b` |
| `raw_source_tokenization.py` | `81821054e3c22a9d8f80cdc07d95b7350b4e35eefa3713fd69ac368aa1304c39` |
| `test_final_constructor.py` | `9587ab04802f0de7138748564145d8262b04c9d236d8f22c4370fad0ce53a701` |

The clean-room implementation lives in
[`docs/campaign/work/final-protocol-v2/`](../../campaign/work/final-protocol-v2/).

| v2 file | SHA-256 |
| --- | --- |
| `protocol.py` | `4ea4572f5865c7d47ecc4282a09e8a6003de73c036925a9f39bc19f268fd789d` |
| `runtime.py` | `587f65bbd48eb23089a0c7a3ebf614bd8ac0a6c24f6e6542df6a141a99a2892c` |
| `test_protocol.py` | `7b9c757221f598feea3b6116a7069b387380d62a4a26f8ce50089cd3dd14b28f` |

The evaluator and PRM03 renderer come from
`523efc07bcb86ebba7fe1030c0a2dc6356f9d111`. The launcher checks the package
source tree against that commit. It uses the renderer and the evaluator's
capacity, greedy generation, stop, and classification functions. The harness
SHA-256 is `8675cdb517f8a4b28ca8a8b7713caf9d3c3adeb34a0fa195b8004195c610734a`.
It binds that commit, both v2 runtime files, and
`final-admission-guards/admission_guard.py`. From the repository root, compute
it with:

```bash
python -c 'import sys; sys.path.insert(0, "docs/campaign/work/final-protocol-v2"); import protocol; print(protocol.harness_hash())'
```

GitHub stores both code directories in this change. Keep a second copy of the
code under `/mnt/nas/bulk/backups/sepalith/`. The preparation copy is
`/mnt/nas/bulk/backups/sepalith/sealed-final-v2-8675cdb517f8/`. Backups contain
code and synthetic tests only.

## Panel and audit

The runner consumes a frozen metadata manifest and separate context and target
payloads. The manifest must identify `final_locked_v1` and carry the guard's
`SourceCaseSpec` fields plus `function_ids` for each case. Freeze the case
count and manifest hash. Bind the same manifest hash as `source_lock_sha256`
in the guard's semantic receipt. That receipt must attest to the split and
bind every requested row and all three hashes. Its file hash and the producer
identity in the receipt are part of the approved plan.

Before freeze, establish the provenance of the panel and the three expected
hashes from approved metadata. The runner implements the v2 handoff shape.
The custodian must verify the mapping from the locked registry and prepare
that handoff under the user's access authorization. Resolve any ambiguity
before a run. A historic basename is insufficient evidence of split identity.
For example, the [historical DEV75 receipt](../../campaign/work/quant-quality/RUN-09-native-dev-quant-quality.md)
describes `DAT-07-final-evaluator-cases.jsonl` as a DEV panel.

The membership input has eight inventories. Each covers the full named source
and contains `row_ids`, `package_ids`, `group_ids`, `split_identities`, and
`function_ids`. Supply TRAIN, DEV, the full CPT corpus, the full C14 pool, SYNTH
seeds, and the exec pool's train, validation, and hold-out. Include all exec
packages and function IDs. Use the frozen partition metadata or an existing
identity-only export for the exec hold-out. Keep all frozen exec hold-out rows
closed. Record provenance and coverage for each inventory before freeze.
Use the same identities as the case manifest, including aliases when the
sources use different names. The audit treats `exec:<package>` as
`<package>` in package comparisons. A case with no applicable function uses
an empty `function_ids` list.

The launcher reuses the v1 guard's freeze, semantic binding, path, and
source-read checks. V2 compares memberships across all eight inventories.
If any case intersects an inventory, the launcher flags its whole group.
A group that intersects several inventories counts once toward the ceiling.
Above **20% of all frozen groups**, the launcher writes the aggregate audit
and stops before opening source files or context and target payloads.

After the overlap check passes, the constructor verifies the raw source bytes and
the canonical JSON of the supplied context and target against the case pins.
Use the guard's `sha256_json` for the PRM03 context map and the target object
`{"operation": "...", "region_new": [...]}`. Exclude a case if any of its three
hashes differs. Count it once in the case total and once under each differing
field. Above **5% of all frozen cases**, the launcher writes the aggregate
audit and stops before generation. The constructor leaves the inputs intact.
Duplicate identities, broken metadata bindings, unsafe paths, and read
errors stop the run. An empty clean subset also stops the run.

## Models and scoring

| Model key | Frozen artifact | Runtime |
| --- | --- | --- |
| `arm-a-803` | Arm A SFT checkpoint 803 | Local Transformers, bf16 |
| `c19-100-bf16` | C19 full checkpoint 100 | Local Transformers, bf16 |
| `c19-100-q6_k` | C17 default Q6_K GGUF, SHA-256 `6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261` | llama.cpp `3cb7ffb1a` |
| `c20-adopted` | C20 checkpoint named in a pinned adoption receipt, if adopted before freeze | Local Transformers, bf16 |

Run serially on CPU with eight threads, context capacity 8,192, and seed 3407.
Use `zeta2-prm03-v1`, the pinned campaign tokenizer, and one manual BOS token.
Use greedy decoding, a cap of 1,024 new tokens, and stop IDs `1` and `130073`,
as required by [RULES.md](RULES.md). The campaign classifier requires canonical
EOS `1` for an accepted prediction. A cap hit or a stop on `130073` can end
generation but does not produce a correct prediction.

The primary metric is the fraction of admitted clean cases with an exact edit
or a correct no-op. An edit must match the entire expected region. A no-op
must classify as a no-op. Keep every hash-valid case in its denominator,
including invalid outputs and cap hits. Report the overlap subset separately,
using the same metric. Also report edit and no-op counts, false suggestions
on no-ops, protocol validity, cap hits, and completions that reached a stop.

Estimate 95% percentile intervals with 10,000 bootstrap replicates and seed
3407. Within each subset, sort packages by ID and resample that many packages
with replacement. Keep every case in a sampled package and divide total
correct cases by total sampled cases. Use the same package draws for all
models. Report paired differences for C19 bf16 minus Arm A, Q6_K minus C19
bf16, and adopted C20 minus C19 bf16. Use the 250th and 9,750th ordered
replicates as the interval endpoints. The lost v1 release cutoffs are
unavailable. This protocol reports the pre-registered estimates and audit
status for the user's C17 review.

## Freeze and launch

Keep the inputs and run-state directory local and access-controlled. Use a
filesystem with directory locks, hard links, and fsync. Create one empty
run-state directory. Its absolute path is part of the plan. Use stable,
read-only paths for the weight, tokenizer, runtime, and panel artifacts.

The plan is a JSON object with the following keys. A file pin is an object
with absolute `path` and the SHA-256 of the file's exact bytes.

| Key | Value |
| --- | --- |
| `schema` | `sepalith.sealed-final.v2.plan` |
| `models` | The model keys above. Use a file pin for the GGUF and `{"path": "/absolute/checkpoint", "files": {"relative/file": "<sha256>"}}` for each bf16 checkpoint. List every file. This binds weights, config, and any bundled tokenizer files. |
| `c20_adoption` | Required if C20 is in `models`. Pin a JSON object with `status: "adopted"` and a `model` entry equal to its checkpoint record. |
| `tokenizer` | A directory record in the same shape as a bf16 checkpoint. Use the campaign tokenizer, including its pinned `tokenizer.json` and `tokenizer_config.json`. |
| `runtime` | `python_sha256` for the resolved interpreter, exact `distributions` versions for torch, transformers, tokenizers, safetensors, and accelerate, and a `llama_server` file pin with `commit: "3cb7ffb1a"`. Use the reviewed CPU builds. |
| `cases`, `semantic_receipt`, `memberships` | File pins for the three metadata JSON objects. The case object contains `split` and `cases`. Membership keys are `train`, `dev`, `cpt`, `c14`, `synth`, `exec_train`, `exec_validation`, and `exec_holdout`. |
| `case_count` | The total number of locked cases. |
| `source_roots` | Absolute real directories from which the guard may read the pinned sources. |
| `payloads_path` | Absolute path to a local JSON object keyed by row ID. Each value contains the pinned `context` and `target` objects. |
| `state_dir` | Absolute path to the dedicated empty run-state directory. |

The user issues this receipt after reviewing the plan. The launcher checks the
receipt's exact SHA-256 against the digest supplied with the user's written
authorization. Keep the approved digest separate from the receipt.

```json
{
  "schema": "sepalith.sealed-final.v2.freeze",
  "status": "frozen",
  "weights_frozen": true,
  "harness_frozen": true,
  "final_access_unlocked": true,
  "weights_frozen_at": "<UTC ISO 8601>",
  "harness_frozen_at": "<UTC ISO 8601>",
  "plan_sha256": "<sha256 of exact plan file>",
  "weights_sha256": "<guard sha256_json of the plan models object>",
  "harness_sha256": "8675cdb517f8a4b28ca8a8b7713caf9d3c3adeb34a0fa195b8004195c610734a"
}
```

After the user approves the receipt and C17 release criteria, use the frozen
interpreter and plan from the repository root:

```bash
python docs/campaign/work/final-protocol-v2/protocol.py \
  --plan /absolute/local/plan.json \
  --freeze-receipt /absolute/local/freeze.json \
  --freeze-sha256 "$approved_freeze_sha256"
```

## Run state and publication

The launcher locks the run-state directory and writes `started.json` once
before opening the panel. It then writes one immutable `result.json` with
schema `sepalith.sealed-final.v2`. A complete result contains the plan and
freeze digests, aggregate audit counts, and the clean and overlapping
summaries. A threshold abort records the reason and the counts available
at that gate. Prompts, targets, predictions, case IDs, package IDs, and
per-case results stay out of the output and logs.

A crash leaves the started marker in place. Resume requires separate written
user authorization with the digest of a `sepalith.sealed-final.v2.resume`
receipt. It must contain `authorization: "user-approved"`, the original
`plan_sha256`, the `started_sha256` of the marker, and `completions` file
pins for every frozen model. Each completion artifact must have schema
`sepalith.sealed-final.v2.completions`, the model key under `model`, the same
`plan_sha256`, and a `completions` object with generated token IDs for every
admitted case. It must come from the original frozen run. The launcher
records one resume attempt and scores those pinned artifacts without
generating again. This runner writes aggregate results only, so if a crash
leaves no independently retained, approved completion artifacts, the run
stays stopped. A completed or threshold-aborted run cannot resume.

Keep the aggregate audit and result local for C17 review. Release publication
needs separate user authorization.
