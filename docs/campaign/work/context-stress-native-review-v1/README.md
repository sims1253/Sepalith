# RUN-05 native stress review

This directory contains an independent CPU-only review of the completed native traces in
`/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-stress-a/`.
The reviewer did not launch a client or server. `review-stress-run.py` loads the immutable
synthetic source fixture and the local tokenizer at
`/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain` with
`add_special_tokens=False, split_special_tokens=True`, then checks each captured
`/tokenize` response.

Run the review from the PLAN worktree with:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  docs/campaign/work/context-stress-native-review-v1/review-stress-run.py
```

The review passes 18/18 exact HF prompt tokenizations and 12/12 completion outputs. The
completion denominator includes baseline and unchanged repeat controls:

| native context | completions | input `context_budget` rejects | accepted cases |
| ---: | ---: | ---: | --- |
| 2048 | 2 | 4 | near-2k |
| 4096 | 4 | 2 | near-2k, near-4k |
| 8192 | 6 | 0 | near-2k, near-4k, near-8k |

Every captured completion has `stop_type=eos`, `truncated=false`, parser `accepted`,
operation `no_op`, and plan mode `none`. The generic `protocolRejected` denominator equals
the input overflow count in each profile; it contains no malformed model output. The raw
response hash is
`2a5d98f5909c8c7e367ca851a776b0976af2cd2665e5048ce12fc0f65a3d26f9`, and generated token
hash is
`edd8cf99f6d73ca4f5150d59156104f1cbd000c72b9e379bb6fa9565801e79f2` for every accepted
case. Same-case output signatures match across every allocation where the case fits and
across repeat controls.

Fresh and duplicate completion timings are recorded separately in `review-report.json`:

| native context | fresh elapsed ms | duplicate elapsed ms |
| ---: | --- | --- |
| 2048 | 245.137766 | 49.706916 |
| 4096 | 253.263152, 420.944803 | 52.428615, 55.957948 |
| 8192 | 247.865694, 410.058312, 856.083450 | 49.927476, 53.648917, 64.380724 |

All three terminals report client and server exit code 0 with no remaining processes.
Each server has `GGML_CUDA_GRAPH_OPT=0`; logs show `USE_GRAPHS=1`, 1308 graph nodes,
two graph splits, graph reuse, and cleanup before exit.

The 8192 profile remains `stress_only_8192`. Stress selector budgets are diagnostic
geometry (5750, 11000, and 23000 UTF-16 units) around the 6000-unit production default;
this review makes no quality or promotion claim.

Key hashes:

* fixture: `016cf26a5a7923b35d298749c30e01bb1ec0ad3defcdc2e06157f1e9b1c1d2c7`
* input manifest: `23c60b9c2e0d6b41571b5c0cab28ec2d2e8adf160e7bd2e7fd355ecda76a9076`
* review report: `90d1f3ca9447fd1376be1d76dc01e722513e59f894eb1c37907acd12218fa654`
* review script: `7ba9fcdb85d7da6b1e5c14c864b98bd81ab45907428e1ee9d0a4fdff665e999b`

