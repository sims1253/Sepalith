# RL-02 train context sidecar materialization

Observed 2026-09-12 in the Tuesday execution worktree. This CPU-only packet
materializes the target-free RL-02 context sidecar from the admitted DAT-05
global registry. It remains candidate-only data preparation and does not grant
RL admission.

The maker now accepts an explicit `--split train` projection while validating
the original global report, registry/provenance paths and hashes, global split
counts, admitted IDs, candidate-spec identities, and provenance candidate-file
hash references. The DAT-05 candidate spec is a bare JSON list and is accepted
as such. The two train candidate files were fully hashed and streamed. The
75-row `DAT-07-final-dev-candidates.jsonl` file was present in the explicit
spec but had no selected train provenance lines, so its contents were not
opened or parsed.

The global registry contains 11,839 admitted rows: 11,764 train and 75 dev.
The complete stored geometry was used without truncation: the prompt is
`input_ids[:target_start]` including manual BOS and is capped at 2,048 IDs; the
completion is `len(input_ids) - target_start`, including the protocol EOS, and
is capped at 192 IDs. This leaves 8,440 eligible train rows and excludes 3,324
train rows by length. The sidecar and filtered rows preserve global registry
order, and the selected-ID file is the exact same ordered projection.

Artifacts:

| artifact | rows | bytes | SHA256 |
| --- | ---: | ---: | --- |
| `eligible-train-rows.jsonl` | 8,440 | 77,278,568 | `e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602` |
| `context-sidecar.jsonl` | 8,440 | 83,927,520 | `6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f` |
| `selected-train-ids.json` | 8,440 | 227,966 | `d1437d677c06f3cfe42cc7af88e888010a2f4bf4d281d81ba62a4f6c61a2a45c` |
| `materialization.json` | — | 5,576 | `7e2f5b8d13e4b79f0d95b341e0f3b3def105d0731de28b8c646c20692c2c8df3` |

The ordered selected-ID digest is
`32aef4f180bc5a86d623d75eede8d06890057ef13597803c16e45f6b1b7af5d7`.
The first ID is `000a6aaa48aee4291dbcb0cb` and the last is
`360b7afbc780bf434b377f0e`. `campaign_rl_data.load_training_records` loaded
all 8,440 records from the three fresh artifacts, returned
`context_snapshot_sha256=None` for the per-row sidecar shape, and computed the
ordered row identity digest
`de512fc747e5a78e2ba431ece4ec348484b40929774acc7e9b437e857d74fa3e`.

Focused CPU verification passed:

```text
PYTHONPATH=packages/sepalith/src:experiments/training python3 -m unittest -q experiments/training/test_campaign_rl_contexts.py
Ran 7 tests ... OK
python3 -m py_compile experiments/training/campaign_rl_contexts.py experiments/training/test_campaign_rl_contexts.py
PASS
```

The tests cover global-registry train filtering, malformed sealed dev input,
candidate-spec list loading, exact selected-ID/order joining, complete
prompt/completion length eligibility, static source/geometry checks, and
recursive target/reward-key exclusion from `context`. No model, framework,
CUDA, network, serving, install, final-set read, registry write, or admission
write occurred. The root agent must still bind these artifacts in a lead-owned
RL input manifest and perform the separate parent/admission gates before any
RL launch.
