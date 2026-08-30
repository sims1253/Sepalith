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
[2026-08-27T21:01+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T21:04+02] zcode-ddot-poc RELEASE ot smoke+full (smoke restarted — gate/log-cadence bug caught before gate eval; no GPU work lost)
[2026-08-27T21:04+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T21:24+02] zcode-ddot-poc RELEASE ot smoke+full (smoke restarted — Sinkhorn batching fix; throughput was launch-bound)
[2026-08-27T21:27+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T21:33+02] zcode-ddot-poc RELEASE ot smoke+full (restarted — packed-coords throughput fix; prior smoke measured 4.5k tok/s pre-fix)
[2026-08-27T21:33+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T21:45+02] zcode-ddot-poc RELEASE ot smoke+full (restarted — top-k routing fix; diffuse early plans kept O(N^2) CE pairs)
[2026-08-27T21:47+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T21:53+02] zcode-ddot-poc RELEASE ot smoke+full (restarted — pair_topk 8->3: 15.2k tok/s measured at k=8, CE-work is k-linear)
[2026-08-27T21:54+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T22:00+02] zcode-ddot-poc RELEASE ot smoke+full (restarted — compiled smoke; eager ceiling was 15.5k)
[2026-08-27T22:00+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T22:07+02] zcode-ddot-poc RELEASE ot smoke+full (restarted — ot_step bypassed the compiled trunk; plumbing fix)
[2026-08-27T22:07+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T22:41+02] zcode-ddot-poc RELEASE ot full run (restarted from 0 — CUDA-graph Sinkhorn + GPU data path: 62.5x on the launch-bound coupling, projected ~36k vs 22k tok/s; ~3h of uncheckpointed progress traded for ~10h saved)
[2026-08-27T22:41+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T23:14+0200] zcode-ddot-graph CLAIM benchmarks for full-graph capture (coexists with ot run, <3GB, intermittent)
[2026-08-27T23:44+0200] zcode-ddot-graph RELEASE benchmarks for full-graph capture (done, committed 7bf3df3; bench peaked ~6.7GB not <3GB — the real 206M config needs 2.5GB for params+grads alone, GPU never pressured: 18+GB stayed free, live run untouched)
[2026-08-27T23:45+02] zcode-ddot-poc RELEASE ot full run (restarted from 0 with --full-graph: whole-step CUDA capture, 1.7x/micro measured, projected 50-60k; only step-100/17-min of progress redone)
[2026-08-27T23:45+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 16h
[2026-08-27T23:46+02] zcode-ddot-poc note: the 23:45 claim text said 0.42 — actual is POC_MEM_FRACTION=0.7 for full-graph memory (script text now fixed)
[2026-08-28T00:35+02] zcode-ddot-poc RELEASE ot full run (restarted — bucket constants tightened; 26.3k@98%util was pad-bound, not launch-bound; ~step-130/25min redone)
[2026-08-28T00:35+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.7 full-graph, <=23GB) ETA 11h
[2026-08-28T01:18+02] zcode-ddot-poc RELEASE ot full run (reverted to non-graphed config per decision rule: graphed 30.4k < 35k; optimization frozen)
[2026-08-28T01:18+02] zcode-ddot-poc CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 17h
[2026-08-28T16:52+02] zcode-ddot-poc RELEASE ot full run (done)
[2026-08-29T00:49+02] zcode-ddot-poc CLAIM three-way eval (eval_ot.py, md+ot arms, ~10GB, ~30m)
[2026-08-29T01:39+02] zcode-ddot-poc RELEASE three-way eval (crashed: unbatched 216-row sampling spiked ~29GB via WSL sysmem fallback + a decode bug; fixing)
[2026-08-29T01:41+02] zcode-ddot-poc CLAIM three-way eval rerun (batched, memfrac 0.6, ~30m)
[2026-08-29T02:34+02] zcode-ddot-poc RELEASE three-way eval rerun (crashed: cross-batch cat pad mismatch; fixed per-row) — CLAIM three-way eval v3 (per-row collection, memfrac 0.6)
[2026-08-29T02:50+02] zcode-ddot-poc CLAIM three-way eval v4 (import fix; memfrac 0.6, ~30m)
[2026-08-29T02:57+02] zcode-ddot-poc RELEASE three-way eval (done — verdict KILLED)
[2026-08-29T14:13+0200] zcode-stabtok CLAIM p1 muon-hygiene chain (4x480 steps, 206M ladder, memfrac 0.55 <=18GB, no coexist) ETA 8h
[2026-08-29T20:21+0200] zcode-stabtok RELEASE p1 muon-hygiene chain (done: 4/4 arms 480 steps + bpb_eval; verdict NEGATIVE, posted)
[2026-08-29T20:23+0200] zcode-cma-poc CLAIM decay/CMA POC chain (Task 2 scorer 480 steps + scoring pass, then Task 4 arms C/D/K/KT/H @ 1BT each; train.py 206M TinyGQA vocab 32768, memfrac 0.42 <=14GB, no coexist) ETA ~35h (chained, heartbeat q30min)
[2026-08-30T12:24+0200] zcode-cma-poc RELEASE decay/CMA POC chain (matrix DONE 12:21: scorer + C/D/K/KT/H all trained, rsynced)
[2026-08-30T12:24+0200] zcode-cma-poc CLAIM cma readout battery (eval_arms per arm + SMA/weighted averages + canary; evals ~10GB, sequential, no coexist) ETA ~2h
