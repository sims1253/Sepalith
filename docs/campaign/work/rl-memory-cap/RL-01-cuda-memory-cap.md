# RL-01 CUDA allocator cap preparation

This preparation adds the selected `0.75` per-process CUDA allocator fraction to the RL entry boundary. It does not claim that the SFT TDR/reset was caused by allocator pressure or that the cap fixes the failure. The cap is a bounded first live comparison selected by root after the SFT-10 audit.

The contract is required in both locations:

```json
{
  "cuda_memory_fraction": 0.75,
  "identity": {
    "policy": {
      "cuda_memory_fraction": 0.75
    }
  }
}
```

CPU preflight rejects a missing value, booleans and non-numeric values, non-finite values, values outside the open/closed interval `(0, 0.80]`, and a recipe value that differs from `identity.policy.cuda_memory_fraction`. The checked value is copied into the runtime admission packet so the live loader cannot silently choose a different fraction.

`_load_live_model` keeps its existing direct-call compatibility: `cuda_memory_fraction` has an explicit default of `0.75`. In the admitted run path, `run()` passes the preflight-validated runtime value. After the existing CUDA availability/device-count check and before `FastLanguageModel.from_pretrained`, the loader calls `torch.cuda.set_per_process_memory_fraction(fraction, 0)`. It then records the device total and the calculated cap in the tokenizer/load audit. `load-audit.json` exposes the same allocator audit under `cuda_allocator`:

```json
{
  "cuda_allocator": {
    "cuda_allocator_fraction": 0.75,
    "cuda_allocator_total_bytes": 17179869184,
    "cuda_allocator_cap_bytes": 12884901888,
    "device_index": 0
  }
}
```

The setter and cap observation happen before model allocation. A setter, device-property, or invalid total-memory failure raises `RLEntryError` and prevents model loading. No framework import is introduced into CPU preflight; the setter helper is reached only from the live model-loading path.

The focused test suite uses a fake CUDA module and fake Unsloth loader. It verifies that setter, device-property observation, and model allocation occur in that order, that the default direct loader call uses `0.75`, and that recipe/policy omissions, non-finite/out-of-range values, and mismatches fail closed.

Preserved pre-edit bytes are in `pre-edit-campaign_rl_entry.py` and `pre-edit-test_campaign_rl_entry.py` in this directory. The current execution source remains dirty by design for root's final immutable refreeze; no shared source, recipe, launcher, state, model, host, or running SFT/evaluation surface was edited.

## Evidence

- Execution commit before this preparation: `a7345e35219ceb624957115c39b869101026a817`.
- Changed execution files: `experiments/training/campaign_rl_entry.py` and `experiments/training/test_campaign_rl_entry.py`.
- Entry contract and fraction validation: `campaign_rl_entry.py:72-80`, `119-130`, and `490-520`.
- Allocator setter and audit: `campaign_rl_entry.py:856-873` and `913-943`.
- Runtime call and load-audit exposure: `campaign_rl_entry.py:1027-1040`.
- Fraction rejection and default/call-order tests: `test_campaign_rl_entry.py:218-250` and `311-371`.
- Focused command:

  ```text
  source /home/m0hawk/Documents/Sepalith/.venv-sft/bin/activate && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest -v test_campaign_rl_entry.py
  ```

  Result: 20 tests passed in 3.437 seconds. This is CPU/mock validation only; no CUDA device or model was loaded.

The combined canonical RL entry/train/telemetry/theta0/checkpoint suite also passed: 71 tests in 7.712 seconds under `CUDA_VISIBLE_DEVICES=''`, one CPU thread, and the pinned `.venv-sft` interpreter. Its framework warnings and mock trainer output did not produce failures.

The live gate remains root-owned: refreeze both changed files, bind `0.75` into each derived smoke/main recipe and identity, verify one quiet CUDA owner and host/TDR monitoring, then run the separate theta0/load and RL smoke gates. The cap remains a diagnostic condition in the resulting receipt, including any cap-induced OOM, rather than a scientific or root-cause claim.
