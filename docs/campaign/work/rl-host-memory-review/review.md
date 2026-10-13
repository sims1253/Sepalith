# RL-06 host-memory source review

Status: source-only diagnosis and patch proposal. The active RL attempt, CUDA state, operating-system state, model bytes, data bytes, and checkpoint payloads were not opened or changed. The review read the five frozen RL modules and the imported `campaign_rl_data.py` hashing/loader helper only to resolve call sites.

## Frozen source identity

The reviewed source root is:

`/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/15fb2080278022b076223c2a823413fc5cb9c014d50dab42dc9df6d57d3f2f41/source/experiments/training`

| file | bytes | SHA256 |
| --- | ---: | --- |
| `campaign_rl_launch.py` | 5,384 | `fa245326052bfae2997d91d80bbf8f45c055d35ffcd0b4b2b7d4440045ffb36a` |
| `campaign_rl_entry.py` | 58,617 | `786356e4c325c1445327c12c5ff27b3a23c6736d2d1475e338d03b84ef7e60af` |
| `campaign_rl_train.py` | 100,780 | `78d27aa98cebc80292d1871a39821eee5a1a705b5a8f412ce270d1707d724443` |
| `campaign_control.py` | 5,024 | `9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac` |
| `campaign_checkpoint.py` | 11,665 | `64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba` |
| supporting import `campaign_rl_data.py` | not in the five-file set | `86ebc3fe80a976e3c35a29057234b341c9e06060135a87382bbf8c915b1db91b` |

## Confirmed duplicate work

The launcher parses the recipe at `campaign_rl_launch.py:81`, then calls `preflight_file` at `:86`. `campaign_rl_entry.py:1131-1136` parses the same recipe again and calls `preflight_entry`. That entry preflight calls `preflight_rl_recipe` at `:729` and `verify_merged_parent_manifest` at `:733`.

`verify_merged_parent_manifest` defaults to `verify_weights=True` (`campaign_rl_entry.py:240-244`). For every inventory item it calls `sha256_file(weight_path)` at `:336`; the helper streams 8 MiB blocks (`campaign_rl_data.py:217-225`). A single large merged weight therefore receives one complete sequential hash pass in the launcher.

The launcher then hashes the small recipe again at `campaign_rl_launch.py:97` and hashes `/usr/bin/timeout` at `:63`. Those are extra reads, but they cannot explain a 5 GB read. The launcher replaces itself with GNU timeout at `:115`, so the child has no in-memory preflight object to reuse.

The child parses the recipe at `campaign_rl_entry.py:1154`, enters `run`, and calls `preflight_entry` again at `:1002`. That second entry preflight reaches the same default `verify_merged_parent_manifest` call at `:733`, including the full weight hash at `:336`. The source therefore proves two full parent-weight hash passes before model loading: one in the launcher and one in the child. It does not prove the exact runtime byte counter; the supplied observation of about 6.3 GB read is consistent with these passes plus the other inputs.

Each pass also hashes the manifest (`:250`), config and generation config (`:269`), and tokenizer files (`:294`). It validates file sizes and the complete inventory (`:297-347`) without retaining weight bytes. The Python hash buffer is bounded, while the sequential reads can populate the WSL/Linux clean page cache. That cache can reduce Windows host free memory even when Linux `MemAvailable` remains high. The source cannot distinguish page-cache pressure from the live process allocation; both paths must be measured at the controlled retry boundary.

## Confirmed object lifetime and copy pressure

The largest avoidable overlap is in `campaign_rl_entry.run`:

* `preflight_entry` loads all selected rows and sidecar records once through `preflight_rl_recipe` (`campaign_rl_train.py:1759-1795`). It returns the full identity and `manifest.to_identity()` data, including inline `row_identities`.
* The same `preflight_entry` then loads rows and sidecar records again at `campaign_rl_entry.py:739-745`, and calls `manifest.to_identity()` again at `:745`, `:759`, and `:780`.
* After the admission packet is written, `run` loads records a third time at `:1020-1025` so `_load_live_model` can inspect prompt rows at `:1027-1031`.
* `build_live_trainer` validates the recipe and loads those records again at `campaign_rl_train.py:1829-1839`, then constructs independent dataset envelopes at `:1878` with `records_to_dataset_rows(records)`.

The row loader intentionally materializes complete JSONL lists (`campaign_rl_data.py:240-259`), a `rows_by_id` map and sidecar map (`:552-614`), and per-row source/geometry copies (`:395` and `:444`). This is correct for the admitted data contract, but repeated calls create overlapping Python object graphs. `RLDataManifest.to_identity()` makes a new identity dictionary and row-identity list (`campaign_rl_data.py:183-214`), and the row-identity hash temporarily serializes that list (`:184-189`).

The recipe identity has another explicit deep-copy point: `validate_rl_recipe` first makes a top-level `dict` and then calls `check_identity` (`campaign_rl_train.py:1533-1539`); `check_identity` performs a JSON dump and parse (`campaign_checkpoint.py:68-73`). The live builder repeats that validation at `:1829`, and `checkpoint_callback` deep-copies the identity again at `campaign_checkpoint.py:170-182`. These copies preserve canonical content but temporarily coexist with the original 74 MB recipe identity.

`run` writes the full recipe and full admission packet at `campaign_rl_entry.py:1012-1015`, while retaining both in memory. The supplied 145 MB preflight and 148 MB checkpoint sizes are therefore plausible manifestations of inline identity, but file size alone is not a Python peak measurement. `live_recipe = dict(recipe)` at `:1061` is shallow; it is not the main duplicate. The list comprehensions at `:1028` and `campaign_rl_train.py:1966` hold row references, while `records_to_dataset_rows` creates the actual independent envelopes.

Checkpoint identity is intentionally inline. `seal_checkpoint` canonicalizes identity and writes it into both `campaign-state.json` and `campaign-manifest.json` (`campaign_checkpoint.py:89-110`); `verify_checkpoint` canonicalizes the requested identity and rehashes the complete checkpoint inventory (`:114-123`). `archive_checkpoint` verifies, copies, and verifies again (`:126-145`). This explains repeated checkpoint I/O and the large campaign-state file, but it is after the model/data preparation boundary and should remain unchanged for resume safety.

## Ready low-risk lifetime patch

`campaign_rl_entry-lifetime.patch` in this directory is a dry-run-checked patch for `campaign_rl_entry.py` only. It makes no recipe, identity, data, reward, model, or checkpoint change:

1. After `admitted-recipe.json`, `entry-preflight.json`, and `parent-audit.json` are durably written, it retains only `parent_audit` and the three scalar values used later (`model_load_max_seq_length`, `cuda_memory_fraction`, and `max_steps`), then deletes the full `admission` packet. The full packet remains available on disk for audit, and `recipe["identity"]` remains unchanged for trainer/checkpoint identity.
2. Immediately after `_load_live_model` returns, it deletes the third-load `records` and `manifest`. The prompt row references have completed their only use; `build_live_trainer` still performs its existing canonical load and validation.
3. It replaces later reads of the discarded packet with those captured values, preserving all current gates and output fields.

The proposal was checked with `patch --dry-run` against a copy of the exact frozen `campaign_rl_entry.py`; no source file was modified. This is the first patch to apply on a controlled retry because it reduces overlap at precisely the transition where the supplied run stopped.

## Hash and cache boundary recommendation

The smallest identity-preserving way to remove the duplicate 5 GB pass is to expose the existing `verify_weights` switch through `preflight_entry` and `preflight_file`, then have only the launcher call `preflight_file(recipe, verify_weights=False)`. The launcher would still validate recipe schema, data, geometry, manifest bytes, file presence/sizes, tokenizer/config hashes, complete inventory, and aggregate inventory identity. It must label its receipt as a structural dispatch precheck. The child keeps the default `verify_weights=True` and performs the mandatory full byte check at `campaign_rl_entry.py:733` immediately before `run` reaches `_load_live_model` at `:1027`; a stale or changed weight therefore still fails closed. The recipe and checkpoint identity are byte-for-byte unchanged. This should be introduced only at a later controlled boundary and tested with a deliberately changed weight that the child rejects.

For host cache pressure, the attached `campaign_rl_entry-cache-advisory.patch` shows a best-effort `posix_fadvise(POSIX_FADV_DONTNEED)` after a completed verified weight hash. It is advisory, does not alter bytes or validation, and must be measured rather than treated as a memory guarantee. The more effective placement for the observed run is the root-owned supervision boundary after the logged model load and LoRA attachment complete: release clean pages for the exact admitted parent weight files, then run compaction. Refuse that release if the owned training process still has an active mapping/open use of those files; do not scan or evict unrelated model, data, or checkpoint files. The observed root-side release of the exact inactive set reclaimed about 6.7 GB Linux cache and raised Windows free memory from about 11.6 GB to 17.8 GB, which supports cache pressure as a contributor but does not establish that the Python heap was safe.

Do not apply `DONTNEED` blindly after model load if the loader retains CPU-backed mappings that future training can fault. Validate the controlled arm with host free memory, Linux `Anon`/`Cached`, model-load completion, first trainer construction, and unchanged prompt/output behavior. The hint may be unsupported and should then be reported as an advisory failure, never treated as an admission failure.

## Later validation gates

Use a fresh output/receipt boundary and the same frozen recipe/source identity. Before enabling the optimization, record:

* one full current preflight with parent weights verified and its parent/data/recipe hashes;
* one run with the lifetime patch and a peak-memory trace covering post-load, adapter attachment, dataset construction, and trainer construction;
* byte equality of `admitted-recipe.json`, `entry-preflight.json` identity, `parent-audit.json`, and the terminal checkpoint identity against an unpatched fixture;
* the existing full checkpoint verification and interruption/resume test, including exact sampler cursor and `campaign-state.identity` equality;
* a negative test with one changed parent weight proving the child full verifier still rejects it when the launcher uses structural preflight;
* cache accounting showing only the exact admitted parent weight files were advised, with no data/checkpoint eviction and no active mapping/open-file violation.

The source review recommends applying the lifetime patch first, retaining the full child weight gate, and treating cache release as a separately measured supervision action. No quality, throughput, or training-success claim follows from this preparation.
