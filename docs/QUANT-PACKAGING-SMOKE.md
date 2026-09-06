# Packaging and quant smoke — 2026-09-06

Historical local smoke record, preserved during cleanup. It predates the manifest
profile/cache hardening and updated check counts. The referenced raw
15-row files were not present at their documented paths during cleanup, so these
results were not independently reconfirmed. No models or native runtime downloads
were rerun.
Current offline checks are `python3 scripts/check_product.py`.

The serving/packaging implementation passes the local pipeline smoke. Q4_K_M
reduces B4's GGUF size by 30.5% with the Q8 embedding/output policy. The small
paired scenario test found no exact-match loss and no imatrix benefit. This is
not a release-quality verdict.

## Artifacts

Source: banked B4 LoRA merge at `/mnt/h/sepalith/runs/pft1_b4_merged`.
Converter and quantizer: llama.cpp b10453, commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`.
The new Q8 baseline matches the banked B4 exact outcomes for all 15 sampled IDs.

Calibration text contains 1,870 sampled rows from 32 training families, seed 3407,
up to 64 rows per family, retaining the assembled rendering. Source split:
`/mnt/h/sepalith/datasets/sft_v7/train.jsonl`, SHA-256
`683dcc24434dbf585a05dcdc486799df4fba3f604cae44f783cca6387de698b4`.
The smoke imatrix processed eight 512-token chunks on CPU, with special-token
parsing. It contains 186 importance-matrix entries. This samples only a small
part of the prepared text; it does not establish calibration coverage.

| Export | Size, MiB | Exact / 15 | Validator pass / 15 |
|---|---:|---:|---:|
| Q8_0 | 1918.8 | 9 | 10 |
| Q6_K + imatrix | 1601.8 | Not evaluated | Not evaluated |
| Q4_K_M, uncalibrated | 1332.8 | 9 | 11 |
| Q4_K_M + imatrix | 1332.8 | 9 | 10 |
| IQ4_XS + imatrix | 1258.0 | Not evaluated | Not evaluated |

Both Q4 arms use the same Q8_0 output and token-embedding overrides. Each tier
was quantized directly from f16; no Q8-to-Q4 requantization occurred. Both Q4
arms preserve every baseline exact pass/fail outcome on the sampled IDs.
The uncalibrated arm gains one validator pass; the calibrated arm has the same
validator outcomes as Q8. No request errors occurred.

Evaluation: existing `eval_scenarios.py`, three held-out rows from each of five
families, context 8192, temperature 0, maximum 640 output tokens, eight CPU
threads. The scenario holdout and validators were unchanged. No latency claim
is made because other work was active on the machine.

The orchestration shell reported an EOF parsing error after the third evaluation
completed because its script was edited while it was running. All three final
JSON aggregates contain 15 rows, the paired comparison passes, and the owned
server exited. The saved script passes `bash -n`; the error did not truncate an
evaluation.

GGUFs and receipts: `experiments/models/packaging_b4*.gguf{,.json}`.
Calibration text, matrix, hashes, logs, and paired summary:
`experiments/models/quant-calibration/`.
Per-example results: `experiments/eval/results_scenarios_packaging_b4*.jsonl`.
These large/local artifacts remain gitignored.

## Packaging checks

- VS Code TypeScript build and bundle pass.
- Four Python exporter/calibration tests pass, including mismatched imatrix
  provenance, policy flags, deterministic sampling, and Q4 calibration opt-in.
- Two Zed launcher tests pass, including a native shell launch with mocked HTTPS
  downloads, hash verification, cache reuse, and automatic platform dispatch.
- TypeScript runtime checks pass for manifest validation, backend selection,
  checksum failure, cache repair, and cancellation.
- A downloaded, normalized upstream Linux CPU runtime completed a real health
  check and OpenAI completion request using the 0.8B Q8 GGUF. The normalized
  runtime contains 40 files and occupies about 63 MB unpacked.
- GitHub attachment filenames and generated launcher shell syntax were checked.
- Installed Zed reports 1.15.0. Personal settings were not changed, and UI
  prediction/acceptance was not tested.

## Release gates

Run a larger calibration and the full paired scenario/noop battery before choosing
a release quant. Validate Q6/IQ4 quality if those tiers will be published. Check
native Windows/macOS and GPU runtime startup, CPU recovery, and both editors'
prediction acceptance. Select the HF model repository and immutable revision,
assemble the final manifest, and publish reviewed runtime assets to GitHub.

First-release targets are VS Code and Zed. Runtime hosting is GitHub Releases;
weights are Hugging Face model files pinned by commit and SHA-256. No artifacts
have been published. See [the runbook](SERVING-PACKAGING.md) for commands.
