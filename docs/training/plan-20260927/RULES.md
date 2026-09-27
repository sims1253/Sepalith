# Pre-registered decision rules

These rules let agents continue without waiting for the user. Agents apply
them mechanically and record the inputs and the outcome in their status file.
Only the user changes a rule, and only before the run it governs starts.

Every rule has three outcomes:

- **continue**: proceed automatically.
- **stop-for-user**: pause and set the status file to `needs_user`, with the
  numbers and a one-line recommendation.
- **rollback**: stop the arm and name the last good checkpoint as its result.

## Metrics

All metrics come from the existing campaign evaluator (`campaign_eval.py`),
using greedy decoding, `max_new_tokens` 1024, and stop ids 1 and 130073.

| Name | Panel | Meaning |
| --- | --- | --- |
| `valid` | DEV75 | Cases whose output parses under the PRM03 contract (of 75) |
| `cap` | DEV75 | Cases that hit the generation cap (of 75) |
| `edit` | DEV75 | Edit cases whose predicted region exactly equals the target (of 43) |
| `nofp` | DEV75 | No-op cases where the model suggested an edit anyway (of 32; lower is better) |
| `score250` | DEV250 | Exact edits plus correct no-ops on DEV250 (built in card C05) |

DEV75 is a restraint-stress panel: 43% no-ops and few finish_block cases.
DEV250 matches the TRAIN family mix and carries most of the weight.
With 75 cases, a McNemar test needs about 8 or more one-sided discordant pairs
to reach p < 0.05. The bands below use raw counts calibrated to that noise
floor, not significance tests.

## R1. SFT gates

**First gate of an arm**, compared with the step-0 parent baseline:

- continue if `valid >= 60`, `cap <= 2`, the loss is finite, and `edit` beats
  the parent's `edit`.
- stop-for-user otherwise.

**Later gates**, compared with the previous gate of the same arm:

| Change | continue | stop-for-user | rollback |
| --- | --- | --- | --- |
| `edit` drop | 0 to 2 | 3 to 7 | 8 or more |
| `nofp` rise | 0 to 1 | 2 to 3 | 4 or more |
| `valid` drop | 0 to 1 | 2 to 4 | 5 or more, or `valid < 68` |
| `cap` (absolute) | 0 to 1 | 2 to 4 | 5 or more |
| `score250` drop | 0 to 4 | 5 to 9 | 10 or more |

The worst outcome across rows wins. Any NaN, a crash in the evaluator, or a
wrong panel or checkpoint hash is always stop-for-user.

## R2. Learning-rate pilot selection

Arms run to the first gate from the same parent with identical data order.

1. Discard arms that fail the first-gate floor in R1.
2. Pick the arm with the highest `score250`.
3. If the top two are within 5, pick the lower learning rate. Its forgetting
   risk is lower.
4. If they are still tied, pick the lower CPT 2K validation loss.
5. If no arm survives step 1, stop-for-user.

## R3. Roxygen arm B versus arm A

Adopt arm B only if all of these hold:

- `score250(B) >= score250(A) - 3`
- B beats A by 3 or more exact cases on the DEV roxygen subset
- `nofp(B) <= nofp(A) + 1`

Otherwise keep arm A. Borderline cases, meaning one condition missed by 1,
are stop-for-user.

## R4. Post-SFT route

Run the pass-rate census (card C12): k = 8 samples per TRAIN prompt at
temperature 1.0, scored by reward v2.1. Let `M` be the share of prompts with
0 < exact pass rate < 1.

1. Always try the offline route first (card C13): RFT on verified samples,
   then DPO. It needs at least 2,000 verified-correct samples and at least
   1,000 preference pairs. Otherwise stop-for-user.
2. After an adopted offline step, run the census again on the new model.
3. Go to online GRPO (cards C14 and C15) only if `M >= 15%` on the latest
   census and the memory probe passes.
4. If `M < 15%`, stop-for-user. The recommended options are a second offline
   iteration at temperature 1.2, or ending post-training.

## R5. Offline step adoption (RFT, DPO)

Compared with the model the step started from:

- adopt if `score250` rises by 5 or more, `nofp` rises by at most 1, and
  `valid` drops by at most 1.
- stop-for-user if `score250` rises by 1 to 4 with the other two conditions met.
- discard otherwise.

## R6. GRPO runs

Evaluate DEV75 every 50 updates and DEV250 every 150 updates.

- Apply the R1 later-gate table against the previous evaluation.
- Stop-for-user if the share of zero-variance groups after filtering stays
  above 50% for three evaluations in a row, which means the signal is used up.
- Stop-for-user if the mean completion length changes by more than a factor
  of 2 from the start.
- Rollback if the reward rises while `score250` falls by 10 or more. That
  pattern points to reward hacking.

## R7. Resources

- GPU work runs only in the nightly window (01:00 to 09:00 Europe/Berlin) on
  the PC, or on Kaggle. No paid compute.
- Kaggle GPU use above 20 hours in a quota week is stop-for-user.
- If `/mnt/e` drops below 150 GB free, stop-for-user before writing checkpoints.
- Deleting any checkpoint, dataset or published artifact is always
  stop-for-user.
