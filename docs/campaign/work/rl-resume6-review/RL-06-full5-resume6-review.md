# RL-06 full5 resume-6 discrepancy review

This CPU-only review compares the known continuous `RL-primary-p2-mixed-mb4-a` run with the resumed `RL-primary-p2-mb4-full5-b` run. Both records are the rollout immediately before optimizer step 6 (`global_step=5`); the old run's last 32 rows are compared with the resumed run's first 32 rows. No model, CUDA framework, optimizer tensor, or sealed final artifact was loaded.

The verifier is [`verify_rl06_resume6.py`](verify_rl06_resume6.py). It passed syntax checking and produced the receipt [`RL-06-full5-resume6-independent-review.json`](RL-06-full5-resume6-independent-review.json). The result is `semantic_match_gradient_scalar_drift`.

## Restore evidence

The resumed entry preflight records `resume_audit.status=verified`, `complete_optimizer_boundary=true`, `step=5`, `consumed_rows=1280`, `source_draw_cursor=40`, and `source_schedule_bound=true`. Its resume inventory binds the full checkpoint's adapter, optimizer, scheduler, RNG, trainer state, tokenizer, and campaign state. The parent full checkpoint manifest is `full=true`, `step=5`; its trainer state is global step 5/max steps 3000. The resumed train-begin tokenizer contract is verified at step 5 with BOS 0, EOS/PAD 1, vocabulary 130,560.

The shared source, parent, policy, renderer, geometry, and tokenizer fields match. The old identity is `dee63a052efeb9dd1ddeb2fbbf7e623c0b3cc8dc61cefcd8d5526df7ddbe8384`; the resumed identity is `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2`. They are different contracts because the old run is `primary-grpo-p2-2048x192-v4-mb4` with full-save cadence 25 and no recorded runtime environment, while the resumed run is `primary-grpo-p2-2048x192-v5-full5` with full-save cadence 5, first DEV update 5, and explicit `OMP_NUM_THREADS=2`, `TORCHINDUCTOR_COMPILE_THREADS=2`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, and related environment bindings. Therefore this comparison cannot establish cross-identity byte-exact resume.

## Update-6 comparison

All 32 prompt hashes, generated ID arrays and hashes, generated token counts, terminal reasons, right-padding counts, source schedule hashes, group geometry, accounting fields, and schema values match exactly. The old last-32 reward records equal the resumed first-32 reward records as complete JSON records. Their eight collapsed source IDs are exactly v6 sequence draws 40 through 47:

```text
2444bb3fa384d2983534cc4d
08fbe1cca9ef899427c1a3e2
25073dcc01264693929381cd
1ded34cc93a918309f73b166
ae782c83ad0e008b67d34998
9dc6f473da4cdfaa7265859d
33f2fa0ca8482ebe022fe04d
0dd89c550556c3fcd6499cfb
```

Stable trainer metrics also match: loss `0.08832169324159622`, 214,180 tokens, reward `0.6643569469451904`, and reward standard deviation `0.08358980715274811`.

The only generation metadata differences are run-specific elapsed time and `trl_microstep`/`update`: the continuous run reports `trl_microstep=40`, while the resumed invocation reports `trl_microstep=0`. The frozen source code records this field from TRL's trainer `_step` in `campaign_rl_train.py`; the sampler resume cursor is tracked independently. Since all source/output/reward rows match, this is a known invocation metadata discontinuity rather than evidence of a changed rollout. It should remain visible in future telemetry reviews.

## Gradient discrepancy

Both update-6 gradient records are finite, report 588 present and 588 nonzero LoRA gradients, and have no invariant field differences. The old norm is `0.13306185510899274`; the resumed norm is `0.133078204166254`. The difference is `+0.000016349057261250133`, or `+0.01228680995606021%`. The logged loss and reward metrics remain exact despite the norm difference.

The evidence supports a small floating-point/reduction or compiled-kernel variation as plausible. It does not prove that explanation: the old continuous and resumed runs have different identity contracts, and the old artifact has no full optimizer checkpoint at step 5 for tensor-semantic comparison. The difference is finite and small, with exact rollout/reward joins and a verified full restore boundary, so it does not by itself block root's continuation. Do not record this as a byte-exact resume proof.

The resumed run was still root-owned and active during this audit. This receipt makes no terminal or quality claim and does not authorize process control.
