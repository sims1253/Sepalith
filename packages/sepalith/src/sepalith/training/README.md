**Full-weight training runtime**

This subpackage consolidates the full-weight CPT, editing-SFT, DEV-gate and RL
code that ran the September 2026 campaign. The code was copied from five packets
in `docs/campaign/work/lead/`. Only imports and module paths changed, apart from
the graceful-stop fix and small lint cleanups. [`PROVENANCE.json`](PROVENANCE.json)
maps every module to its source file, original sha256, the packets that bind it,
and the exact changes made. The snapshot in `docs/campaign/` stays the frozen
record.

**Modules.** `optim` holds the dense Aurora/Muon full-weight optimizer mixin.
`checkpoint` seals, verifies and publishes `full_weights` checkpoints and checks
the 381-tensor dense layout and saved precision. `contracts` pins the tokenizer,
trainer tokenizer alignment, and the CPT stage-transition, native-runtime and
RL update contracts. `sft` holds the editing-SFT trainer, its binder, the
target-only collator and the draw-schedule builders. `eval` holds the DEV
generation evaluator, the gate files, and CPT holdout loss. `cpt` holds the
finished CPT trainer and its native staging and attestation. `rl` holds the
GRPO production driver, reward v2, rollout data and the profiling and theta0
gates. `guards` holds the WSL host-memory and GPU-reset supervisor. `supervisors`
holds the graceful-stop request and the page-cache helper. `paths` defines the
configurable roots: `CHECKPOINT_ROOT` (default `/mnt/e`, env
`SEPALITH_CHECKPOINT_ROOT`) and `CAMPAIGN_ROOT` (default `docs/campaign`, env
`SEPALITH_CAMPAIGN_ROOT`). The checkpoint root must be an absolute directory
below `/`. Trainer, archive and bulk-cache outputs must lie strictly inside it:
`under_checkpoint_root` compares normalized path components, so a sibling such
as `/mnt/e-other` is refused. The shared prompt contract stays at
`sepalith.campaign_protocol`.

**Lint scope.** `ruff.toml` and `ty.toml` exempt the ported modules and packet
tests file by file, listing only the rules each file violates as ported. New
modules under `training/` get the full rule set, and so do the files written
for C01 (`paths`, `request_graceful_stop`, the test fixtures).

**Entry points for later cards.** Run them as modules with `packages/sepalith/src`
on `PYTHONPATH` and the training interpreter
`.venv-sft/bin/python` at the repository root (built from `requirements-sft.txt`):

| Step | Command |
|---|---|
| SFT bind | `python -m sepalith.training.sft.bind_full_weight_edit_sft --template T --admission A --output BOUND` |
| SFT train | `python -m sepalith.training.sft.full_weight_edit_sft {preflight-template,preflight,run} --recipe BOUND [--resume CHECKPOINT]` |
| DEV generation | `python -m sepalith.training.eval.run_dev_generation --bound-recipe BOUND --checkpoint C --dev-admission A --output RESULT [--panel {dev75,dev250,both}] [--dev250-output RESULT250] [--dtype {bf16,fp16,fp32}] [--cuda-device N]`. Defaults: DEV75, bf16, the single leased GPU; this is the original call. `--panel both` evaluates DEV75 and DEV250 in one model load, and its admission also binds `dev250_panel_sha256` and `dev250_output`. Non-bf16 results carry `generation_runtime` and cannot bind a gate. |
| Gate check | `python -m sepalith.training.eval.prepare_dev_gate ...` writes `dev-gate-step-N.json`; `sepalith.training.eval.milestone_gate.continuation_plan` checks it before a resume |
| Save and stop | `python -m sepalith.training.supervisors.request_graceful_stop --recipe BOUND` (SFT or CPT recipe) |
| RL driver | `python -m sepalith.training.rl.preflight_full_weight_rl --recipe R --binding B --receipt OUT`; the trainer is built by `sepalith.training.rl.full_weight_rl_production.build_production_trainer` |

Before new recipes can use this code, three things change:

- Trainers verify `recipe["source"]` against a source manifest. New recipes
  must bind a manifest of these package files, not the packet manifests.
- Factory strings in recipes, such as `evaluator_factory`, are resolved with
  `importlib`. They must name package modules, for example
  `sepalith.training.eval.campaign_eval:...`.
- The theta0 gate's default model loader imports `campaign_rl_entry`, which
  no packet bound. Pass your own loader.

**Editing SFT v2 (card C04).** `sft/schedule_v2` builds the 13-edit plus
3-no-op schedule. `sft/recipe_v2` holds the pre-registered template, the
arm admission and the binder; it binds a source manifest of the trainer-path
package files. `sft/night_arm` is the night-queue driver. It trains to the
next gate, runs DEV75 and DEV250 generation in separate processes, applies R1
(`sft/gate_rule`), writes the decision and the gate evidence, and resumes.
`sft/night_controls` holds the deadline stop and the pre-update masking gate,
and `sft/parent_baseline` evaluates the parent at step 0. The bound recipes,
the job files and the rebuild script are in
`docs/training/plan-20260927/c04/`.

| Step | Command |
|---|---|
| Arm (night job) | `python -m sepalith.training.sft.night_arm arm --recipe BOUND [--stop-after-gate 268]` |
| Parent baseline | `python -m sepalith.training.sft.night_arm baseline --recipe BOUND` |
| Smoke | `python -m sepalith.training.sft.night_arm smoke --recipe BOUND --first-stop 10 --second-stop 20` |
| Override a gate | write `<evidence_directory>/override-step-N.json` (`sepalith.sft-v2.gate-override.v1`: `step`, `outcome`, `bound_recipe_sha256`, `reason`, optional `author` and `checkpoint_manifest_sha256`), then `night_queue retry ID` |
| Rebind after a code change | `docs/training/plan-20260927/c04/build_c04.sh bind` |

**Tests.** `python3 scripts/check_training.py` runs `packages/sepalith/tests/training/`
on CPU with the training interpreter. The tests read the frozen packet templates
through `campaign_fixtures.py`. It rebases their planning-worktree paths onto
`CAMPAIGN_ROOT`, and onto a local overlay (`SEPALITH_CAMPAIGN_OVERLAY`, default
`/mnt/e/sepalith/campaign-20260915/campaign-overlay`) for the large artifacts
the snapshot excludes. `PROVENANCE.json` lists those artifacts with their hashes.
The stdlib core suite (`scripts/check_core.py`) does not collect these tests.
