# RL main preflight memory audit

This is a source and artifact audit for the two failed first-update attempts.
It does not load a model, import a framework, parse a large JSON artifact, or
attribute the host guard stop to CUDA.

## Observed run facts

The frozen RL-02 inputs contain 8,440 admitted train rows. The known artifact
sizes are:

| Artifact | Bytes |
| --- | ---: |
| `eligible-train-rows.jsonl` | 77,278,568 |
| `context-sidecar.jsonl` | 83,927,520 |
| `selected-train-ids.json` | 227,966 |
| `primary-b.recipe.json` | 74,015,461 |

The main `primary-a` and `primary-b` attempts both reached a verified
`load-audit.json` and then stopped before generation. The b guard stopped at
7,469 MiB free with `host_free_memory_below_floor`; its terminal is
`stopped_or_failed`, child exit `-15`, and there were no generation, reward,
gradient, or telemetry records. The b admission receipt records no page-out or
new NVIDIA driver event. These facts establish a pre-update host-resource
failure, not a CUDA or data-quality root cause.

## Source path that scales with the 8,440-row sidecar

The current saved source snapshot is:

`/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/15fb2080278022b076223c2a823413fc5cb9c014d50dab42dc9df6d57d3f2f41/source/experiments/training/`

The relevant source lines are:

* `campaign_rl_data.py:240-259` implements `_jsonl` by reading every line
  into a Python list. `load_training_records` then retains `rows_by_id` from
  the full 77 MB row file and `raw_contexts` from the full 84 MB sidecar
  (`campaign_rl_data.py:552-584`). For the sidecar shape it additionally keeps
  `sidecar_by_id`, validated `RLTrainRecord` objects, and one full
  `source_identity`/`selection_geometry` object per selected row
  (`campaign_rl_data.py:573-614`).
* `RLDataManifest.to_identity()` materializes another selected-ID list and a
  full `row_identities` list (`campaign_rl_data.py:196-209`). The selected IDs
  and row identities are therefore deliberately present in the recipe identity
  as well as in the live records.
* `preflight_rl_recipe` loads the complete rows and sidecar and returns the
  identity, data identity, selected IDs, and source draw sequence
  (`campaign_rl_train.py:1759-1797`). `campaign_rl_entry.preflight_entry` calls
  that function, then calls `load_training_records` a second time
  (`campaign_rl_entry.py:729-745`).
* After the model load, `run` calls `load_training_records` a third time while
  retaining the admission identity and live records (`campaign_rl_entry.py:1002-1025`).
  `build_live_trainer` validates the recipe and loads the same rows and sidecar
  a fourth time (`campaign_rl_train.py:1829-1839`) before constructing the
  dataset (`campaign_rl_train.py:1875-1879`). The run/build-trainer pair can
  therefore overlap in full parsed copies while the 5 GB merged weight model
  is resident. This is the strongest direct explanation for the high RSS at
  the transition from model load to trainer construction. The first two loads
  are repeated sequential validation passes; their retained identity/data
  objects still increase the next peak.

The launcher does one earlier recipe/preflight pass
(`campaign_rl_launch.py:81-86`), but then `execve`s the supervised child
(`campaign_rl_launch.py:108-115`), so that pass's Python heap is discarded.
The child parses the 74 MB recipe again (`campaign_rl_entry.py:1153-1156`).
`write_json` streams through the object (`campaign_checkpoint.py:50-65`), so
the 74 MB `admitted-recipe.json` and 145 MB `entry-preflight.json` files are
evidence of the identity payload size; the repeated parsed data structures,
rather than the file names themselves, are the likely RSS multiplier.

## 75-case evaluator comparison

`campaign_eval.development_evaluator` reads every declared DEV case into a
persistent `cases` list at trainer construction (`campaign_eval.py:121-140`).
The main recipe retains 75 cases, while the earlier smoke retained 14. During
an evaluation it creates all prepared rows and then appends full generated IDs,
raw output, and loss objects to `results`, rewriting a growing partial JSON
artifact after each case (`campaign_eval.py:145-193`). Thus the 75-case panel
adds a persistent and later evaluation-time multiplier over the 14-case smoke.
It cannot explain `primary-b`'s observed stop by itself: b stopped before the
first update, and no evaluator output exists. The dominant pre-update risk is
the repeated 8,440-row load above; the full 75-case evaluator remains a later
evaluation memory risk that should be measured after a successful first update.

## Boundaries for the next root-owned retry

The RL-05 verifier records the first-update result only. A retry should keep
the frozen 2048/192/2240 rollout contract and the 4096 evaluator/model-load
contract, and should capture RSS/free-memory points around each known load
boundary: child recipe parse, `preflight_entry`, model load, adapter attach,
and `build_live_trainer`/dataset construction. This will distinguish host
cache pressure from the repeated Python-heap peak without changing the data or
quality protocol. No model or full identity receipt was loaded for this audit.
