# RL-03 independent preparation review

Observed at `2026-09-12T16:28:02Z`. This is a CPU-only, read-only review of
the RL-03 two-update preparation against the frozen RL snapshot. It did not
load a model, reserve CUDA, start a launcher, create campaign state, or change
the generator, snapshot, data, parent, or admission receipt.

## Decision

The generator's emitted identity and recipe shape are compatible with the
frozen entry, train, launcher, control, and checkpoint interfaces. The live
smoke is **blocked** until root supplies both an accepted merged-SFT parent
with the canonical tokenizer bytes and a positive RL-02 admission receipt.
The generated default timeout is also too large for the current approximately
one-hour gate allocation. Root must materialize an explicitly shorter derived
recipe before launching; the generator has no timeout override option.

## Findings

### F-01: RL-02 admission is still a launch blocker

The current receipt
`docs/campaign/receipts/RL-02-v4-pool-lead-review.json` has SHA256
`5c78b4b79e6a9e933d168cc32ecef12e5951a463afe4dd5e382a09536970cddf`,
`status=candidate_pool_and_source_draw_schedule_reviewed`, and
`launch_admitted=false`. The source schedule itself is marked
`status=frozen_candidate_only`. Supplying that receipt to the generator is
correctly rejected by `_verify_rl02_admission` at lines 358-373; no local
status edit is an admission.

The future receipt must be a fresh positive admission and recursively bind all
of these exact values: rows
`e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602`, sidecar
`6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f`, selected
IDs `24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d`,
schedule `892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48`,
and sequence `d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2`.
The generator and the frozen train loader then independently validate 24,000
draws, the selected-ID order, and the source geometry.

### F-02: the currently merged parent has the wrong tokenizer bytes

Root's audit found that the step-1000 CPU merge serialized normalized tokenizer
files instead of the inspected canonical files. The required parent tokenizer
hashes are exactly:

```text
tokenizer.json        3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81
tokenizer_config.json e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b
```

The generator compares the accepted parent's tokenizer hashes with the
tokenizer used for DEV inspection at lines 663-669. The frozen entry also
verifies every tokenizer byte and the native IDs in
`verify_merged_parent_manifest` (lines 223-358). Root's proposed fresh parent
packaging is the correct repair: preserve the merged model/config and
generation config, copy the canonical tokenizer files, then independently
verify the full parent manifest and prompt fixture before issuing the parent
wrapper. Do not weaken this byte identity check.

### F-03: the default wall budget exceeds the gate allocation

The generator constants at lines 51-56 emit `max_attempt_seconds=1800`,
`termination_grace_seconds=60`, and `checkpoint_reserve_seconds=120` for all
three recipes. `campaign_rl_launch.supervised_command` (snapshot lines 32-69)
uses `max_attempt_seconds` directly to set the hard GNU `timeout` cutoff; it
has no command-line duration override. Three attempts can therefore consume
`3 * 1800 = 5400` seconds (90 minutes), before the separate theta0 allowance
of up to 1200 seconds. This must not be used under the current approximately
one-hour total gate.

For a concrete root-owned bound, 600 seconds per attempt gives three hard
windows of 1800 seconds plus the 1200-second theta0 ceiling, leaving 600
seconds for preflight and orchestration. The 600-second value is a scheduling
bound, not evidence that two updates will finish in 420 seconds of usable
training time after the 60-second grace and 120-second reserve. If root chooses
another budget, record it explicitly and rerun preflight; do not silently use
the generated 1800-second field.

### F-04: source-group parity needs the reward stream as well as generation telemetry

The frozen generation telemetry records `prompt_ids_sha256` and generated IDs,
but does not include the source row ID. The reward sink records `id` and is
therefore the authoritative per-row source evidence. For each two-update arm,
root should check the first 64 reward records as 8 source IDs repeated across
G=4 candidates per update, using schedule positions 0-7 and 8-15. Generation
records should then be compared by the step-2 `prompt_ids_sha256` and exact
`generated_ids`; timing fields are measured metadata and should not be used as
parity keys. This is an operational comparison requirement, not a reason to
weaken the identity binding.

## Verified contract

The v2 tokenizer-only inspection result is intentionally
`status=inspection_complete_recipes_not_emitted`; no parent or positive
admission was available. Its exact artifacts are:

```text
development-panel-small.jsonl  9192bd278c0342ce94d029ee5b1b9964b3fce0048ed6b99e4bcc0d97aaf06b59  14 rows
development-panel-audit.json   bd71d36deb9686621029329f7c570809ed02a4bfe50e0ed4cbde1e3f8a7f790d
source-prefix-audit.json       95cbfa441fdf766badfdcc371dfdbd206d1f25713cd2a55cb45e4ed5793a32ab
preparation-result.json         020a4918b6cb91b25dbf12661b4bf2062cbf8463f05048bf30ca60ee27dfc33c
```

The panel audit confirms the 2,619-token maximum case is already in the
frozen 14-case bytes and that target fields were not used to form requests.
The source-prefix audit confirms the full schedule hash and sequence above,
16 audited draws in two groups of eight, G=4, 32 completions per update, 128
sampler rows per update, and 256 rows over two updates. The source schedule
file hash is `892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48`.

The generator binds one identity with exactly these schedule fields:

```text
seed=3407, sampler_id=campaign-repeat-manifest-order-v1
candidate_count=4, rollout_rows_per_update=32
per_device_train_batch_size=8, gradient_accumulation_steps=4
generation_batch_size=32, steps_per_generation=4, num_iterations=1
source_draws=24000, source_draws_per_update=8, buffer_reuse=4
max_steps=2, full_save_steps=1, light_save_steps=1, evaluation_steps=[1,2]
```

The snapshot's `resolve_trl_geometry`, `load_source_draw_schedule`,
`validate_rl_recipe`, and `CampaignRepeatSampler` agree with these values.
At full checkpoint boundaries the sampler state must be:

```text
step 1: consumed_rows=128, consumed_prompt_copies=32, source_draw_cursor=8,
        selected_id_index=the 9th source sequence index
step 2: consumed_rows=256, consumed_prompt_copies=64, source_draw_cursor=16,
        selected_id_index=the 17th source sequence index
```

The exact parent/entry contract is `status=accepted`, `kind=merged_sft`, a
complete sorted weight inventory, matching aggregate weight identity, model
config and generation-config hashes, and the tokenizer bytes/IDs above. The
admission contract is `admission_status=admitted` for the 8,440 selected train
rows. No parent or admission has been claimed here.

The inspected call signatures match the generator's calls:

```text
campaign_rl_data.load_training_records(rows_path, rows_sha256, context_path,
  context_sha256, selected_ids_path, selected_ids_sha256, *, admission_status,
  expected_context_snapshot_sha256)
campaign_rl_train.load_source_draw_schedule(path, expected_sha256, *,
  selected_ids, selected_ids_sha256, ordered_ids_sha256, row_identity_sha256,
  candidate_count, source_draws_per_update, buffer_reuse)
campaign_rl_entry.preflight_entry(recipe)
campaign_rl_entry.preflight_file(recipe_path)
campaign_rl_launch.supervised_command(recipe, command, *, now=None)
campaign_checkpoint.verify_checkpoint(directory, identity=None, *, require_full=False)
campaign_control.control_callback(*, telemetry_path, identity, deadline,
  reserve_seconds, stop_steps=(), clock, monotonic, resource_probe=None)
```

`build_live_trainer` also accepts the generator's evaluator, checkpoint,
sampler, parent, deadline, and tokenizer contract plumbing. The current EXEC
copies of entry, launch, train, checkpoint, control, data, evaluator, and
protocol equal the frozen snapshot hashes; the key source hashes are recorded
in the receipt.

## Root materialization and bounded launch recipe

After the theta0 gate, accepted parent, and positive RL-02 receipt exist, run
the generator command from the preparation guide with fresh `OUT` and
`RUN_ROOT`, using the exact input hashes above. Then derive a root-owned
600-second recipe copy before preflight. This changes only the launcher budget;
the seven-field training identity remains unchanged. Keep the derived recipe
hashes in the root launch receipt.

```bash
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
EXEC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
GEN="$PLAN/docs/campaign/work/rl-smoke-preparation/prepare_rl_two_update_smoke.py"
OUT="$PLAN/docs/campaign/work/rl-smoke-preparation/generated-rl03-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=/home/m0hawk/.local/state/sepalith/campaign-20260915/rl03-two-update-$(date -u +%Y%m%dT%H%M%SZ)
PARENT=/absolute/accepted/merged-sft-parent.json
ADMISSION=/absolute/root/receipts/RL-02-admission.json

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$GEN" \
  --execution-root "$EXEC" --output-dir "$OUT" \
  --parent-manifest "$PARENT" --parent-manifest-sha256 '<PARENT_SHA256>' \
  --rl02-admission-receipt "$ADMISSION" \
  --rl02-admission-receipt-sha256 '<RL02_ADMISSION_SHA256>' \
  --run-root "$RUN_ROOT" --deadline '<UTC_DEADLINE>' \
  --protocol-sha256 5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156 \
  --trainer-sha256 77bcc7fbe4ef6b6a8ba497d22f6948215f3849b23806dfa6a64f6d3a7b1e1e5f \
  --tokenizer-revision 8dc5f6055b90fe4b9422340810b270b9569f37f3

DERIVED="$OUT/bounded-600"
mkdir "$DERIVED"
python3 - "$OUT" "$DERIVED" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:])
for name in ("uninterrupted", "split-first", "split-resume"):
    value = json.loads((src / f"rl-two-update-{name}.recipe.json").read_text())
    value["max_attempt_seconds"] = 600
    (dst / f"rl-two-update-{name}.recipe.json").write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
PY
sha256sum "$DERIVED"/*.recipe.json
```

The derived recipes must be CPU-preflighted with fresh result receipts. The
split-resume preflight is intentionally deferred until split-first has sealed
`archive/full/checkpoint-1`.

```bash
mkdir -p "$RUN_ROOT"
for name in uninterrupted split-first; do
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
    "$EXEC/experiments/training/campaign_rl_entry.py" \
    "$DERIVED/rl-two-update-$name.recipe.json" --preflight \
    --receipt "$RUN_ROOT/$name-preflight.entry-result.json"
done

"$PY" "$EXEC/experiments/training/campaign_rl_launch.py" \
  "$DERIVED/rl-two-update-uninterrupted.recipe.json" \
  --receipt "$RUN_ROOT/uninterrupted-supervision.json"
"$PY" "$EXEC/experiments/training/campaign_rl_launch.py" \
  "$DERIVED/rl-two-update-split-first.recipe.json" \
  --receipt "$RUN_ROOT/split-first-supervision.json"

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
  "$EXEC/experiments/training/campaign_rl_entry.py" \
  "$DERIVED/rl-two-update-split-resume.recipe.json" --preflight \
  --receipt "$RUN_ROOT/split-resume-preflight.entry-result.json"
"$PY" "$EXEC/experiments/training/campaign_rl_launch.py" \
  "$DERIVED/rl-two-update-split-resume.recipe.json" \
  --receipt "$RUN_ROOT/split-resume-supervision.json"
```

The launcher must be used for every live attempt. Direct entry execution lacks
the required `SEPALITH_CAMPAIGN_SOFT_DEADLINE` and
`SEPALITH_CAMPAIGN_HARD_DEADLINE` environment variables and is rejected.

## Resume acceptance comparison

Root should accept the smoke only after all three attempts have durable
terminal or failure evidence and the following checks pass:

1. The uninterrupted and split-first recipes have byte-identical seven-field
   identities. Split-first terminates at `lead_decision`, with full checkpoint
   1; split-resume starts only from that full checkpoint and reaches step 2.
2. Checkpoint-1 and checkpoint-2 campaign state report the sampler values
   above. `campaign-manifest.json` inventories include adapter/config,
   optimizer, scheduler, RNG, trainer state, and campaign state. Verify each
   with `verify_checkpoint(..., require_full=True)` and compare identity,
   geometry, source schedule hashes, optimizer/scheduler step, and
   `trainer_state.global_step`.
3. Compare `rng_state.pth` and the step-1 full checkpoint state between
   uninterrupted and split-first. Compare the step-2 generation records and
   reward rows between uninterrupted and split-resume. Any source-group or
   generated-ID difference is a resume parity failure even if both reach step
   2.
4. Verify the first 64 reward IDs against the two schedule groups, each source
   ID expanded to four candidates. Verify each step's evaluator artifact has
   the exact 14-case denominator and finite NLL/quality fields. The future
   long run still requires the full 75-case DEV panel.
5. Verify the immutable supervision receipts remain byte/hash stable and are
   distinct from entry-result receipts. A timeout, missing full checkpoint,
   pending evaluator, nonfinite gradient/reward, tokenizer drift, or changed
   source schedule blocks admission.

## Source and artifact hashes

```text
generator prepare_rl_two_update_smoke.py
  eda26a1027c074c70c73a6755369ea4b88d8fa0944d4b5870d1f8f2b5eec614e
design RL-03-two-update-smoke-preparation.md
  48b19e80a226074e477537d4c2cc68f71dfa5a8b647b9bada56106e70c8d6855
snapshot campaign_rl_entry.py
  fd3d77cee6acd424c3653bff85bc29e5f03e039c81b2088ac99e1b26ed3d2bb1
snapshot campaign_rl_launch.py
  fa245326052bfae2997d91d80bbf8f45c055d35ffcd0b4b2b7d4440045ffb36a
snapshot campaign_rl_train.py
  77bcc7fbe4ef6b6a8ba497d22f6948215f3849b23806dfa6a64f6d3a7b1e1e5f
snapshot campaign_checkpoint.py
  64202f4e5b88c94a74794442be3a4955f07d195079c9d64bbcff2850dbadd0ba
snapshot campaign_control.py
  9a5b0614f134ecf6b804d2f0d5e37a4fecf2b1dc53589b92f17580678f9dbdac
snapshot campaign_rl_data.py
  86ebc3fe80a976e3c35a29057234b341c9e06060135a87382bbf8c915b1db91b
snapshot campaign_eval.py
  7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f
canonical campaign_protocol.py
  5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156
```

Current EXEC hashes for these files equal the corresponding frozen snapshot
hashes. The review did not run a model or live RL attempt, so all theta0,
parent quality, CUDA occupancy, policy-logit memory, trainer loss, gradient,
evaluator, and interruption results remain root-owned live gates.
