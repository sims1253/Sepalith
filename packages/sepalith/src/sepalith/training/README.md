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
configurable roots: `CHECKPOINT_ROOT` (default `/mnt/e/`, env
`SEPALITH_CHECKPOINT_ROOT`) and `CAMPAIGN_ROOT` (default `docs/campaign`, env
`SEPALITH_CAMPAIGN_ROOT`). The shared prompt contract stays at
`sepalith.campaign_protocol`.

**Entry points for later cards.** Run them as modules with `packages/sepalith/src`
on `PYTHONPATH` and the training interpreter
`/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python`:

| Step | Command |
|---|---|
| SFT bind | `python -m sepalith.training.sft.bind_full_weight_edit_sft --template T --admission A --output BOUND` |
| SFT train | `python -m sepalith.training.sft.full_weight_edit_sft {preflight-template,preflight,run} --recipe BOUND [--resume CHECKPOINT]` |
| DEV generation | `python -m sepalith.training.eval.run_dev_generation --bound-recipe BOUND --checkpoint C --dev-admission A --output RESULT` |
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

**Tests.** `python3 scripts/check_training.py` runs `packages/sepalith/tests/training/`
on CPU with the training interpreter. The tests read the frozen packet templates
through `campaign_fixtures.py`. It rebases their planning-worktree paths onto
`CAMPAIGN_ROOT`, and onto a local overlay (`SEPALITH_CAMPAIGN_OVERLAY`, default
`/mnt/e/sepalith/campaign-20260915/campaign-overlay`) for the large artifacts
the snapshot excludes. `PROVENANCE.json` lists those artifacts with their hashes.
The stdlib core suite (`scripts/check_core.py`) does not collect these tests.
