# RL-05 first-update independent review: `RL-primary-p2-mixed-mb4-a`

Observed at the first valid update boundary while the bounded run continued. This review is scoped to update 1 only; it makes no terminal, resume, or quality claim.

## Result

**First update: pass.** The 4x8 arm used the admitted geometry shown in the process log:

- per-device batch 4, gradient accumulation 8, total batch 32;
- `G=4`, P2 (`generation_groups_per_call=2`), four calls of eight rows;
- eight logical groups and 32 generated rows/rewards;
- source schedule SHA `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132`.

The first generation prefix has 32 rows at `global_step=0` and exactly four calls/eight G4 groups. All 32 rows reached canonical EOS, with no cap hits. The first reward prefix has the required family mix: format propagation 8, rename propagation 8, no-op 8, finish block 4, and pipe rewrite 4. Operations are replace 24/no-op 8.

The first gradient record is finite and nonzero: 588 trainable LoRA tensors present, 294 nonzero, norm `0.08439055626126028`, `global_step=0`, `step=0`. The process log reached progress `1/3000`, providing the first optimizer-step evidence. No first-update allocator failure occurred. Root observed subsequent progress through update 3 during this review; those later updates are outside this receipt.

## Exact failed-arm comparison

The first 32 generation rows from the 4x8 run were compared with the first 32 rows from failed `RL-primary-p2-mixed-a`. Ordered prompt hashes, generated-token hashes, prompt token counts, generated token counts, and terminal reasons matched all 32 rows. The only generation-record field differences were expected `elapsed_sec` and the v4/v6 `source_schedule_sha256`. The ordered first 32 reward records matched byte-for-byte, including source IDs, output hashes, rewards, family, operation, canonical EOS, and cap flags.

This verifies rollout/reward reproducibility for this theta0 and source prefix. It does not establish gradient, optimizer, or model-state equivalence between 4x8 and 8x4; the failed arm emitted no gradient.

## Host and terminal scope

The child and host guard were still running at the observation; no terminal artifact was used to close the review. The latest observed guard samples had no driver events and no page output; available host memory remained above the 8 GiB floor, although committed memory and transient page reads were monitored by the guard. The 52 MiB first telemetry record and large identity/preflight graphs were not parsed.

## Acceptance evidence

| check | evidence | result |
|---|---|---|
| 4x8 runtime geometry | process log and generation `group_geometry` | pass |
| G4/P2 shape | 4 calls × 2 groups/call × 4 candidates | pass |
| generation prefix | 32 rows, `global_step=0`, 4 calls, 8 groups | pass |
| terminal/cap contract | 32 canonical EOS, 0 cap hits | pass |
| family denominator | 8/8/8/4/4 expected mix | pass |
| reward join | ordered first 32 IDs/output hashes/rewards exact vs failed prefix | pass |
| gradient | finite, 588 present/294 nonzero, norm 0.0843905563 | pass |
| optimizer boundary | progress bar `1/3000`, first gradient step 0 | pass |
| terminal/resume/full checkpoint | run remained active; no evidence collected | pending |

No framework or model was loaded by this review process. It used only bounded reads of the known generation, reward, gradient, and host process artifacts.
