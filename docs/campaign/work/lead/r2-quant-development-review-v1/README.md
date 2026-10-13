# REL-07 five-arm quant development review

Review time: 2026-09-14T06:51:32.607610+00:00. This is an independent, read-only review of the prepared packet at `docs/campaign/work/lead/r2-quant-development-v1`. No model tensors were read, no server was launched, no CUDA or cloud access was used, and no file in the running packet was edited.

## Decision

The context-aware DEV scorer and the TRAIN panel identity pass independent checks. The five-arm packet itself remains **HOLD** for scientific comparison: its admitted run stopped on the host memory guard before IQ3 completed and before IQ2 started, and its recorded launch uses CUDA with six model threads despite the CPU-only two-thread review constraint. Partial arm outputs must not rank or promote an IQ arm.

## Independent DEV scoring check

The pinned label artifact is `corrected-dev75-v1/dev75-corrected-finish-v1.jsonl` (SHA-256 `7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035`). It has 75 unique DEV IDs: 43 `replace` rows and 32 `no_op` rows. The packet's `r2-quant-development-v1/dev-cases.json` preserves ID order and matches those labels on family, package, operation, target body, and context for all 75 rows. The request file has 225 unique `(row_id, cap)` pairs, exactly 75 rows at each cap (192, 384, 512), all `cold`, `rep=1`; there are no context-budget failures.

I independently replayed the scoring expression in `r2-dev-cap-sweep-v1/analyze_context.py` against the recorded raw text and the pinned `campaign_protocol.PromptContext`. Every one of its 225 row-level records matches. The context-aware parser treats a replacement body equal to `context.region_old` as `no_op`; this corrects the preliminary `summary.json`, whose token-only parser counted those unchanged bodies as suggestions.

| cap | rows | protocol-valid | exact edits / 43 | correct no-op / 32 | false suggestion / 32 | finish valid / 6 | finish exact | median wall ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 75 | 69 | 26 | 26 | 4 | 3 | 0 | 244.775 |
| 384 | 75 | 72 | 26 | 26 | 6 | 4 | 0 | 246.007 |
| 512 | 75 | 72 | 26 | 26 | 6 | 4 | 0 | 228.749 |

The cap-192 response and returned-token IDs have 75/75 parity with the recorded native-Q8 baseline. These are exact token/region and protocol measures; they do not establish semantic R correctness. Caps 384 and 512 are diagnostic according to their manifest and are not serving-promotion evidence.

## TRAIN panel identity and denominators

`r2-quant-development-v1/panel.jsonl` is 45 unique rows, byte-identical to `r2-q4-pair-v1/panel.jsonl`, all `split=train`, with 41 `replace` rows and 4 `no_op` rows. Family counts are `format_propagation=22`, `na_rm_propagation=3`, `pipe_rewrite=10`, `rename_propagation=6`, and `no_op=4`. All 45 manifest source-line references and row hashes match the exact source JSONL line plus newline in `finish-corrected-train-v1/train-token-rows.jsonl`; the source and panel SHA-256 is `e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe` for the source and `6739323b8a8d5d94dd0e386f8fdedc35a6e09595907908d04a43d0851ac1c197` for the panel. The manifest's `selection.edit_rows=4` describes four edit families and must not be used as the row denominator; the data-derived edit denominator is 41.

The selected arm manifest binds five model path/hash pairs and declares 45 TRAIN rows. Completed Q8, ordinary Q4, and calibrated Q4 outputs each contain 90 requests (45 cold plus 45 warm), all with accepted protocol status. The TRAIN records are protocol/latency evidence only; this panel copy does not carry the DEV context labels used for context-aware exact-edit scoring.

## Execution and cleanup findings

The admission record declares 5 arms, 75 DEV cases and 90 TRAIN requests per arm, or 825 requests. In this packet, the observable outputs are 165 each for Q8, ordinary Q4, and calibrated Q4; 27 DEV rows for calibrated IQ3; and no IQ2 output. That is 522 recorded requests, with no IQ3 TRAIN result and no IQ2 result. The guard terminal record is `stopped_or_failed` at `2026-09-14T06:39:48.081427Z`, child exit `-15`, reason `host_free_memory_below_soft_floor_persisted`; its last memory sample was 15,837 MiB available against a 16,384 MiB soft floor, with zero swap and no driver events. IQ3's server log shows cleanup, but no IQ3 terminal record was written; IQ2 was never launched in this attempt.

The three completed arm terminal records say `exit_code=0` and `pid_absent=true`. A current process scan found no surviving `llama-server` or packet client process, so no live process was observed at review time. The cleanup implementation still terminates only the direct server PID in `run.py`/`run.sh`; the packet does not record a post-interruption process-group audit or IQ3 `pid_absent` result. The outer guard claims process-group ownership, but absence of a terminal record limits the cleanup evidence.

The launch source hashes in `launch.json` match the seven admitted files. The runtime contract is not exact: both probes send `temperature=0` but omit the declared `seed=0`, and `run.py` omits the declared `-ngld 99` argument while using `-ngl 99` and `-t 6`/`-tb 6`. The observed commands set `CUDA_VISIBLE_DEVICES=0`, `-ngl 99`, and six model threads. This is a launch-contract and reproducibility limitation even if greedy temperature zero usually makes ties rare.

## Acceptance boundary

Accept the DEV context-aware reclassification and the 45-row TRAIN panel identity for reuse. Keep the five-arm packet in HOLD until a fresh, fully recorded run has all five arms, all 75 DEV rows and 90 TRAIN requests per arm, exact runtime arguments/seed, and terminal process cleanup evidence. Do not infer IQ3 or IQ2 quality from the partial IQ3 prefix or from missing records. Exact-edit/no-op results remain a structural measurement, not a semantic correctness or package-test result.
