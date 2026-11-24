# Final panel v1

This directory holds the replacement constructor for the sealed final (C17
step 4). The design is in
[final-panel-v1.md](../../../training/plan-20260927/final-panel-v1.md).

| File | Purpose |
| --- | --- |
| `final_panel.py` | The constructor. Its entry point is `build_panel()`, and its CLI runs rehearsal builds of `dev_group` and `train_group`. A `final_candidate_group` build runs only with a user build authorization. |
| `recipe.json` | Pre-registered quotas, floors, filters, input pins and module pins |
| `dev_equivalence.py` | Reproduces DEV250 and DEV75 cases from raw dev parents. It writes `receipts/C17-final-panel-dev-equivalence.json`. |
| `inventories.py` | Writes the eight membership inventories with a coverage receipt, and the model, tokenizer and CUDA runtime pins (run `pins` with `.venv-sft`). Output goes to `~/.local/state/sepalith/sealed-final-v2/` only. |
| `holdback.py`, `holdback-split.json` | The pre-registered holdback split of the 193 final groups (group IDs only): 105 build, 88 sealed. `holdback.py check` recomputes it from the pinned DAT-02. |
| `dev250_reference.py` | DEV250 rates of the plan roster reweighted to the final family mix. It writes `receipts/C17-final-panel-dev250-reference.json`. |
| `dress_rehearsal.py` | The dev dress rehearsal of A2 to A5: rehearsal authorization, authorized build, plan and freeze, the v2 launcher. Since 2026-11-24 it scores on the GPU through the night job `final-protocol-v2/night_job.py rehearsal` (`assemble --panel-root` plans a fresh root over an earlier built panel). |
| `test_final_panel.py` | Synthetic tests. They need no `/mnt` data and open no DAT-02 or final group. |

Use an interpreter that has `tree_sitter_r` and `transformers`, such as the
repository `.venv`. Commands, run from the repository root:

```bash
export PYTHONPATH=packages/sepalith/src
(cd docs/campaign/work/final-panel-v1 && python -m unittest -v test_final_panel)
python docs/campaign/work/final-panel-v1/final_panel.py --split dev_group \
  --out /abs/fresh/dir --tokenizer ~/.local/state/sepalith/sealed-final-v2/tokenizer
python docs/campaign/work/final-panel-v1/dev_equivalence.py /abs/receipt.json
python docs/campaign/work/final-panel-v1/inventories.py inventories
```

Do not run the final build without the user's written freeze and build
authorization (design, step A2). It reads final content once.
