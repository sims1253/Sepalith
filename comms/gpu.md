# gpu.md — GPU ledger (RTX 5090, 32GB)

Protocol: comms.md. Claim before any CUDA context; release when done.

[2026-08-26T22:47+02] zcode-pvf-poc CLAIM priv-critic training (01_train_value.py, memfrac 0.8) ~26GB ETA 15m
[2026-08-26T23:31+02] zcode-pocdiff CLAIM md smoke (train_md.py --smoke, memfrac 0.42 <=14GB) ETA 20m
[2026-08-26T23:35+02] zcode-pvf-poc RELEASE priv-critic training
[2026-08-26T23:35+02] zcode-pvf-poc CLAIM RL-run-4 GRPO+tether (03_grpo_tether.py, memfrac 0.92) ~29GB ETA 2.5h
[2026-08-27T00:12+02] zcode-pocdiff RELEASE md smoke (superseded — deferring to the post-RL-run-4 supervisor)
[2026-08-27T01:11+02] zcode-pocdiff CLAIM md smoke+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T02:55+02] zcode-pvf-poc RELEASE RL-run-4 (done, 220/220 steps)
