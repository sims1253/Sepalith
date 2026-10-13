# Audit of the plans prepared before September 27

On September 27, four read-only reviews checked the prepared plans. Each
covered one area: the editing-SFT packet, the DEV evaluation gate, the RL
preparation, and runtime portability plus scheduling. Findings marked
**verified** were rechecked by reading the cited file or recomputing the
number. The last column names the card that handles each finding.

The reviews asked one question: were these plans made for the model and settings we
actually have? Several were not. Older material assumes the Qwen3.5
gated-DeltaNet LoRA recipe, 192-token completion caps, or a Monday cutoff.

## Model lineage

| Finding | Status | Card |
| --- | --- | --- |
| The production checkpoint is `LlamaForCausalLM` (MiniCPM5-2B lineage). `SYSTEMS.md`, the Kaggle and Anyscale research notes and `scripts/cloud/` describe the Qwen3.5 GDN LoRA recipe. | verified; handoff and training docs corrected in this change | C03 rewrites the cloud scripts |
| The Qwen3.5-specific Kaggle dtype crash does not apply. A T4 still cannot hold full-weight training state (about 22 GB), and the trainer is single-process bf16 with no sharding. | verified | placement in README |

## Editing SFT packet (`r2-full-weight-edit-sft-preparation-v1`)

| Finding | Severity | Status | Card |
| --- | --- | --- | --- |
| Target masking and termination are correct: prompt masked, body, terminal and EOS 1 supervised. This fixes the SFT-06 collator defect. | OK | verified | none |
| The 1,160-update schedule keeps 25% no-ops by repeating the only 1,094 no-op rows: 830 rows appear 4 times and 264 rows appear 5 times. | major | verified (`exposure_histogram {1: 13904, 2: 8, 4: 830, 5: 264}`) | C04 |
| The 938-update one-pass alternative drops no-ops to about 7% of draws, with no per-batch ratio control. | major | verified (`{1: 15004, 2: 2}`) | C04 |
| The learning-rate ceiling of 3e-5 is enforced in `bind_full_weight_edit_sft.py:31`, with no documented rationale. CPT used a hidden LR of 3e-6. The template's example value is 1e-5. | major | verified | C04, C08 (pilot 3e-6, 1e-5, 3e-5) |
| `root-admission.template.json` carries concrete-looking hyperparameters next to `FILL` placeholders. | minor | reported | C04 |
| `request_graceful_stop.py` imports `full_weight_cpt_trainer`, which is not in the SFT packet. Using it would crash. | blocker for graceful stop | verified | C04 |
| The pre-update masking gate designed after SFT-06 is not wired into the full-weight packet. | minor | reported | C04 |
| 3,503 of 7,785 finish_block rows went through repair. Nobody has checked that the repair fixed the earlier finish-quality defects. | open | reported | C06 |
| `na_rm_propagation` has only 136 rows. | minor | verified (family counts) | accepted limitation |
| Of the 10,017 roxygen candidates, 3,903 are recommended for context admission and 6,114 are held. Admission still needs license, dedup and semantic review plus a fresh cohort. | open | reported | C10 |
| The trainer writes only under `/mnt/e/` and the commands pin `.venv-sft`. | by design | verified | none |

## DEV evaluation gate (`r2-full-weight-edit-sft-eval-gate-v1`)

| Finding | Severity | Status | Card |
| --- | --- | --- | --- |
| Scoring is exact whole-region equality under PRM03, with separate no-op scoring and teacher-forced loss as a diagnostic only. Decoding is greedy with stop ids 1 and 130073. | OK | reported | none |
| `max_new_tokens` 1024 is ample. The longest DEV75 target is 402 tokens; the TRAIN p99 is 536. | OK | reported | none |
| The DEV75 mix differs from TRAIN: 32 of 75 are no-op (43%) and 6 are finish_block (8%). finish_block is 52% of TRAIN. | major | verified | C05 (DEV250) |
| With 43 edit and 32 no-op cases, only large swings reach significance. | major | reported | RULES (count bands) |
| The DEV split has 169 identity groups and 2,846 candidate rows, enough for a larger panel. | OK | reported | C05 |
| No check exists that DEV packages are absent from the CPT corpus. | open | reported | C05 |
| `run_dev_generation.py` hardcodes bf16 and exactly one CUDA device, which blocks Kaggle T4. | blocker for Kaggle | verified | C05, C03 |
| Every gate requires a manual root decision, and no automatic rule exists. | blocker for unattended nights | verified (`milestone_gate.py`) | C04, RULES |
| `docs/PROMPT-CONTRACT.md` documents only the older `zeta2-v1` render, not PRM03. | minor | reported | C05 |

## RL preparation (RL-11, RL-12)

| Finding | Severity | Status | Card |
| --- | --- | --- | --- |
| Reward v2's R-parse probe treats every nonzero exit, including signals, as a syntax failure. The fix was assigned but never applied. | blocker | verified (receipt status open) | C11 |
| No memory profile exists for full-weight GRPO. The only RL GPU profile is a 25M-parameter LoRA run. | blocker for local GRPO | reported | C14 |
| The length policy (3,072 prompt, 1,024 completion, 4,096 context) is reviewed but not integrated. The driver still hardcodes 2,048 and 192, which rejects 4,106 of 15,006 rows. | major | verified (`campaign_rl_data.py:50-51`) | C11 |
| The 80-group census (58 all-correct, 16 all-zero, 6 mixed) came from the old LoRA "R2" model with 192-token caps. It does not predict the new model. | major | reported | C12 |
| Reward v2 is exact match plus binary penalties. It has no partial credit and no `air` canonicalization. | major | reported | C11 |
| Two RL tracks exist: the campaign RL-11 driver, and `experiments/training/rl_smoke.py` plus `rl/` from the LoRA era. | major | reported | C11 (mark the LoRA track historical) |
| No DPO code exists. RFT can reuse the SFT runtime once verified samples become TRAIN-shaped rows. | gap | reported | C13 |
| No vLLM or census harness exists. | gap | verified (repo-wide search) | C03, C12 |
| RL-12 still cites a Monday cutoff and the "theta0 Q8 and b4" identities. | minor | reported | superseded by this plan |
| The RL prompt pool is the same 15,006 TRAIN rows as SFT, under the same tokenizer. TRL is pinned at 0.24.0 and the mixins assert their MRO. | OK | reported | none |
| The 8,246-prompt pool size quoted in older notes is superseded by 15,006. | note | reported | none |

## Runtime, portability and scheduling

| Finding | Severity | Status | Card |
| --- | --- | --- | --- |
| The full-weight runtime was not in git on any branch. It existed only as untracked packet `source/` copies and state-dir scripts. | critical | verified | C00 (snapshot in git), C01 |
| The live supervisor hardcodes the planning worktree as its source. Cleaning that worktree would break it. | critical | reported | C01 |
| The packet sources hold 1,551 Python files but only 317 distinct contents. | high | verified (hash count) | C01 |
| A deadline-based graceful stop with full-state save and exact resume already exists and is tested. | OK | reported | C02, C04 reuse it |
| No unattended nightly scheduling exists. WSL has systemd enabled, and cron only runs while WSL is up. | high | reported | C02 |
| One observed checkpoint, evaluation and resume cycle took about 7 minutes 46 seconds. Resume to first update took about 101 seconds. | sizing | reported | C02 |
| No real secrets were found in the code snapshot. Literal checks covered all ten keys in `~/.zshrc`, the old Hermes key and the Kaggle key; pattern checks covered HF, GitHub, AWS, Kaggle and private keys. One token-shaped test fixture returned 401 from Hugging Face and was left out anyway. | OK | verified | C00 |
| `scripts/cloud/` targets the Qwen3.5 LoRA recipe. | high | verified | C03 |

## Placement conclusions

- Full-weight training, including SFT, RFT, DPO and GRPO, runs only on the PC
  in the nightly window. Kaggle T4 has no bf16, and the trainer would need
  sharding across two 16 GB cards, which would not match the reviewed recipe.
  A Kaggle TPU port could work in principle, but it would be a porting project
  outside this plan.
- Inference-heavy work runs on Kaggle once the fp16 parity check passes. That
  covers the pass-rate census, sample generation for RFT and DPO, and optional
  DEV250 repeats.
- Gate evaluations run on the PC inside the same night as training. This keeps
  them bf16 and authoritative, and lets gates auto-continue without waiting a
  day.
- CPU work runs on the PC during the day at low priority. Larger batch jobs can
  use Kaggle CPU sessions. The notebook is for llama.cpp latency checks on
  target-class hardware.
