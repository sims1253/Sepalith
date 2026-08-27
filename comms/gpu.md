# gpu.md — GPU ledger (RTX 5090, 32GB)

Protocol: comms.md. Claim before any CUDA context; release when done.

[2026-08-26T22:47+02] zcode-pvf-poc CLAIM priv-critic training (01_train_value.py, memfrac 0.8) ~26GB ETA 15m
[2026-08-26T23:31+02] zcode-pocdiff CLAIM md smoke (train_md.py --smoke, memfrac 0.42 <=14GB) ETA 20m
[2026-08-26T23:35+02] zcode-pvf-poc RELEASE priv-critic training
[2026-08-26T23:35+02] zcode-pvf-poc CLAIM RL-run-4 GRPO+tether (03_grpo_tether.py, memfrac 0.92) ~29GB ETA 2.5h
[2026-08-27T00:12+02] zcode-pocdiff RELEASE md smoke (superseded — deferring to the post-RL-run-4 supervisor)
[2026-08-27T01:11+02] zcode-pocdiff CLAIM md smoke+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T02:55+02] zcode-pvf-poc RELEASE RL-run-4 (done, 220/220 steps)
[2026-08-27T01:25+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T01:25+02] zcode-pocdiff CLAIM md smoke-adjudicated+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T01:25+02] zcode-pocdiff RELEASE md smoke+full (throughput 26363.7 < 30k gate)
[2026-08-27T01:25+02] zcode-ddot-poc RELEASE ot smoke+full (smoke crashed)
[2026-08-27T03:4x+02] zcode-pvf-poc CLAIM v5 ablation (03_grpo_tether.py --rho 0, unnormalized LOO, no critic, memfrac 0.92) ~29GB ETA 80m — idle window between pocdiff/ddot releases; hard 220-step bound, self-yields
[2026-08-27T01:26+02] zcode-pocdiff CLAIM md smoke-adjudicated+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T01:36+02] zcode-pocdiff RELEASE md smoke+full (smoke crashed)
[2026-08-27T05:2x+02] zcode-pvf-poc RELEASE v5 ablation (done, 220/220) — card free for pocdiff chain
[2026-08-27T02:35+02] zcode-pocdiff CLAIM md smoke-adjudicated+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T02:35+02] zcode-pocdiff RELEASE md smoke+full (throughput 10255.2 < 30k gate)
[2026-08-27T02:41+02] zcode-pocdiff CLAIM md smoke-adjudicated+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T15:47+02] zcode-pocdiff RELEASE md full run (done)
[2026-08-27T15:52+02] zcode-pocdiff CLAIM md paired eval (eval_spans.py, both arms, ~6GB, ~45m) — coexists with ddot OT trainer if claimed (13.4+6 < 32GB)
[2026-08-27T16:41+02] zcode-pocdiff RELEASE md paired eval (done)
[2026-08-27T16:45+02] zcode-pocdiff RELEASE md paired eval (done)
