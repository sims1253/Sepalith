# C01. Consolidate the full-weight runtime into `packages/sepalith`

- **Where:** PC, daytime, CPU only
- **Needs:** C00, meaning the snapshot is in `docs/campaign/` on main
- **Produces:** `packages/sepalith/src/sepalith/training/` with tests, plus a
  provenance map
- **Effort:** one to two agent-days

## Goal

Later cards need one importable, tested copy of the runtime, not 48 packet
snapshots. The snapshot in `docs/campaign/` stays as the frozen record.

## Read first

- CONTEXT.md, section Code
- `packages/sepalith/README.md` and its existing `campaign_protocol` module
- The `source-manifest.json` files in the five packets listed in CONTEXT.md.
  They pin the exact file hashes each packet ran with.

## Steps

1. For each packet in CONTEXT.md, read `source-manifest.json` and collect
   `(relative path, sha256)`. Where the same module name appears with
   different hashes, keep the version the newest packet binds and list the
   others in the map. Newest means editing SFT and the eval gate for SFT
   code, driver v2 for RL code, and the cadence-64 continuation for CPT code.
2. Create the subpackages `training/{optim,sft,cpt,eval,rl,checkpoint,contracts,guards,supervisors}`
   under `packages/sepalith/src/sepalith/`. Copy modules in and change only
   imports and module paths, never logic. Keep each module's original sha256 in
   `training/PROVENANCE.json`, along with the new path and a note on what
   changed.
3. Remove hardcoded references to the planning worktree
   (`/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/...`). Replace them
   with parameters or with paths relative to the package. Keep the `/mnt/e`
   checkpoint-root requirement, but make it configurable with the same default.
4. Port the packet tests (`tests/test_*.py` in each packet) into
   `packages/sepalith/tests/training/`. They must pass on CPU with the
   training interpreter, without CUDA.
5. Fix `request_graceful_stop.py` so it works for the SFT trainer (it imports
   the CPT trainer today). Add a test.
6. Add `packages/sepalith/src/sepalith/training/README.md`. It should give a
   one-paragraph map of the modules and say which entry points later cards
   use: SFT train, DEV generation, gate check, RL driver.
7. Mark `experiments/training/rl_smoke.py` and `experiments/training/rl/` as
   historical in their README or docstring. They are the MiniCPM LoRA track
   and are superseded by the campaign RL driver.
8. Run `python3 scripts/check_core.py` and `python3 scripts/check_quality.py`.
   Fix new lint issues in the new package only.

## Acceptance

- Every module the five packets bind is present, or listed as intentionally
  dropped with a reason.
- Ported tests pass on CPU.
- No reference to the planning worktree remains in the package.
- The core and quality checks pass.

## Stop and ask if

- Two versions of a module differ in logic and both are bound by packets this
  plan still needs.
- A test fails in a way that changing an import cannot fix.
