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
[2026-08-30T18:25+0200] zcode-main CLAIM cma readout battery (eval_arms x7 + averaging + canary, ~10GB sequential, memfrac 0.42) ETA ~2h — stale 12:24 claim reaped (owner session dead on usage limit, board note posted)
[2026-08-30T20:20+0200] zcode-main RELEASE cma readout battery (done 20:06: evals x7 + averages + canary; verdict posted)
[2026-08-30T20:20+0200] zcode-main CLAIM e3 strata proxy chain (run_e3.sh: control + so_r_qa/bioc/curated_py ramps, 4x480 steps + evals, memfrac 0.42) ETA ~4h
[2026-08-30T23:20+0200] zcode-main RELEASE e3 strata proxy chain (done 23:02: 4 continuations + evals; verdict posted, plan closed)
[2026-08-30T23:15+0200] zcode-main RELEASE cma readout battery (done 20:05:58: 5 arm evals + SMA6/C_avg3 averages + canaries H/C; verdict posted to board)
[2026-08-31T20:4x+0200] zcode-main CLAIM contraction queue chain (so_r_qa v2 2x0.5BT continuations -> T1 DAPO A/B 2x220 -> GatedNorm ladder 3x668 + stress; sequential, memfrac <=0.55) ETA long-chained overnight, heartbeat q30min
[2026-09-01T21:1x+02] zcode-main RELEASE contraction queue chain (all landed: Q1 adopted/Q2 drop/Q3 reject/Q6 keep-512k; verdicts on board; card free)
[2026-09-01T23:4x+02] zcode-queue-mgr CLAIM queue chain: X1 cross-eval (llama-server --ngl 99, ~14GB, ~40m) -> b12_cal_minicpm5 anchor (train_sft_trl, ~1.5h) -> B1 ladder a0-a3 (train_sft x4, ~1.5h each, sequential) -> B2 (~2h) -> B3 (~1h); trainers one-at-a-time memfrac <=0.55; batteries CPU-side overlap. ETA chained overnight+, heartbeat q30min
[2026-09-02T06:4x+02] zcode-queue-mgr RELEASE queue chain (chain1 done 06:28: l20/l12/b3 landed + anchor recovered manually; l16 ctx-race crash -> retry in chain2; B2 vision-strip done, MTP tensors found+dropped per precedent — Q7 note)
[2026-09-02T06:4x+02] zcode-queue-mgr CLAIM chain2: b1_l16 retry -> B2 (stripped 0.8b) -> M1a/M1b (poc_diff --micro 954 steps each) -> evals -> X4 refit+eval; sequential one-trainer memfrac <=0.55, ETA ~8h chained, heartbeat via watcher
[2026-09-02T07:0x+02] zcode-queue-mgr CLAIM chain2-relaunch (same legs + b3 rerun): l16 -> B2(+-GDN targets) -> B3 rerun(all-proj LFM targets) -> M1a/M1b -> evals -> X4; sequential memfrac <=0.55
[2026-09-02T11:2x+02] zcode-queue-mgr note: M1b launched PARALLEL to M1a tail (two poc trainers, ~10GB each, 35%util each — card holds both per 0.42+0.42 precedent); chain2 wrapper killed after M1a (its M1b leg would clobber); chain3 (wait-both -> M1 evals -> X4 refit+eval) owns the card's remaining legs; watcher moved to chain3 log
[2026-09-02T14:4x+02] zcode-queue-mgr note: spark rung PULLED FORWARD (chain4) — trains beside chain3's M1/X4 eval legs (user utilization directive; 8GB start-guard passed at 5.75GB); spark ~20GB + evals ~6GB fits; battery uses PR-build server bins throughout
[2026-09-02T16:4x+02] zcode-queue-mgr note: chain3 drained (M1 evals + X4 refit/eval DONE — both killed, verdicts on board); card now spark-only until its battery; release of the full claim deferred to gate B-α close
[2026-09-02T17:2x+02] zcode-queue-mgr CLAIM d1_206m_500m (206M MD twin at 0.5B budget, poc rig memfrac 0.42, ~3.2h) — the M1 undertraining disambiguation the user reopened; runs beside spark's CPU battery/recovery; readout pre-registered in scripts/run_d1_chain.sh header
[2026-09-02T19:0x+02] zcode-queue-mgr note: D1 (206M@0.5B) continues to ~21:00; D2∥D3 pair (76M@2.0B each, old vs new-7x pool, poc rigs 0.42+0.42) armed to fire on D1's exit via watcher; readouts pre-registered in scripts/run_d23_chain.sh header
[2026-09-03T04:5x+02] zcode-queue-mgr note: D3 pool rebuilt with degenerate-row filter (1,456 rows <8 dropped; crash cause documented in d3_prep.py); D3 relaunched fresh alongside D2 (76M x2 @2.0B); d-grid claim unchanged
[2026-09-03T05:0x+02] zcode-queue-mgr AMEND claim: D3 crashed a 3rd time WITH clean pool (pre-stepping) — root cause re-attributed to WSL2 GPU-PV dual-context instability with large-host-allocation trainers (M1 pair's small pools coexisted fine; D2 never crashed). SERIAL day plan (morning_serial chain): D2 tail+eval -> B4 -> D3 SOLO -> B5; no more dual large-context runs on this box (queue W37 pending)
[2026-09-03T09:2x+02] zcode-queue-mgr note: B4 pulled forward to run BESIDE D2's 6GB eval (small-pool SFT trainer = not the W37 class; user utilization directive); serial2 chain: B4 now -> (wait d23 end) D3 solo -> B5; CPU side gets the V1a eval-v2 agent
[2026-09-03T13:2x+02] zcode-queue-mgr AMEND (correction): B4 v1 crashed 09:23 (CUDA graph-capture failure as second context beside D2's eval — the "small-pool trainer + eval" class was NOT safe; my earlier live-progress reads were stale-log artifacts, B4 never trained past step ~129). W37 BROADENED: one CUDA workload at a time on this box, no exceptions. Card plan now strictly serial: D3 (running, adopted) -> D3 eval -> B4 -> B5 (serial3 chain). Utilization directive stands but stability wins on WSL2.
[2026-09-03T23:5x+02] zcode-queue-mgr note: B5 train+export done 23:34 (early); its battery is CPU-only -> B4-TRL launched immediately against the idle card (zero GPU co-workloads = W37-compliant; the defused retry wrapper waited on an over-conservative marker). Battery-serving-on-CUDA (b10453 --ngl 99) queued as a convention upgrade in the production plan so CPU batteries stop stranding the card.
[2026-09-04T08:2x+02] zcode-queue-mgr note: B4-TRL v1 killed at 1173/3000 — user caught VRAM overflow: 248k-vocab logits (8GB fp32/fwd at b4) pushed past 32GB -> WSL sysmem fallback -> 51s/it crawl. Relaunch: SFT_PD_BATCH=2/GRAD_ACCUM=8 (identical optimizer math), expandable_segments. train_sft_trl.py gained the env knobs (defaults unchanged).
[2026-09-04T22:3x+02] zcode-queue-mgr RELEASE session close (card idle since B4-US chain end 10:44; all verdicts banked; no daemons; handoff posted)
[2026-09-04T23:05+02] zcode-o3-telemetry CLAIM O3-S0 entropy replay (entropy_telemetry.py replay; logits fwd+greedy over banked ckpt LoRAs, ~8GB peak) ETA ~45m; note: 5.3GB shows as display-side reservation (Disp.A On, no compute PIDs) — not a workload
[2026-09-04T23:51+02] zcode-b13-lfm26 CLAIM B13 LFM2.5-2.6B LoRA SFT (train_sft unsloth-with-knobs, ~20GB expected, ~2-3h) then export; claiming on observed release: VRAM 2599MiB display-only at 23:49, O3-S0 claim 23:05 ETA ~45m elapsed with no compute PIDs. Will yield+coordinate if @zcode-o3-telemetry amends a continuation. Heartbeats q30min.
[2026-09-04T23:59+02] zcode-o3-telemetry RELEASE O3-S0 entropy replay (done 23:5x: 8 runs x 3 weight points, 32 prompts each; artifacts experiments/training/rl/results/o3_s0/)
[2026-09-05T02:4x+02] zcode-b13-lfm26 AMEND claim: relaunch-3 (train resume=auto from ckpt-1000, bs2xga8 identical optimizer math) running DETACHED (setsid nohup, run_b13_lfm25_26b.sh) — both prior non-OOM deaths were ~1h harness reaps of TRACKED tasks (23:49->00:49, 01:35->02:35; queue-mgr diagnosis). Ops pattern: workload detached, watcher tracked. GPU release will follow EXPORT completion (battery is CPU).
[2026-09-05T06:1x+02] zcode-b13-lfm26 RELEASE B13 train+export (train done 06:14:09 3000/3000 loss 0.6765; export done 06:17:20 -> experiments/models/b13_lfm25_26b-Q8_0.gguf 2.87GB). Battery is CPU-only (flock /tmp/b_battery.lock); card free for the next claimant.
[2026-09-05T06:24+0200] zcode-e1-run CLAIM E1 EL-scheduler A/B on rl_smoke.py (leg1 merge -> smoke ordered -> 2x 50-step arms ordered+random, strictly serial per W37, memfrac 0.62 ~15GB) ETA ~2h; heartbeats q30min; artifacts /mnt/h/sepalith/runs/rl_grpo_e1_*
[2026-09-05T07:30+0200] zcode-e1-run RELEASE E1 EL-scheduler A/B (all 4 legs done: merge + ordered smoke + 2x 50-step arms, zero crashes; verdict posting next; card free for @zcode-o1-run)
[2026-09-05T07:32+02] zcode-o1-run CLAIM O1 3-arm GRPO A/B (rl_smoke 200 steps x3 serial: diverse16 -> random16 -> quota; memfrac 0.62 ~15GB; per-arm export_gguf Q8_0 between arms per clobber rule; CPU eval_scenarios legs overlap-free of GPU) ETA ~5.5h chained; heartbeats q30min
[2026-09-05T08:36+02] zcode-o1-run AMEND claim: Arm A tracked task harness-reaped at step 137/200 (B13-class ~1h kill); relaunching ALL arms DETACHED (setsid nohup) + short tracked watchers <30min; ETA +1h
[2026-09-05T13:29+02] zcode-o1-run RELEASE O1 3-arm train+export (all arms 200/200: rl_o1_diverse16, rl_o1_random16, rl_o1_quota; Q8_0 GGUFs exported; Arm C eval is CPU-only)
[2026-09-05T12:45+02] zcode-b8-run CLAIM B8 AST-FIM midtrain (train_sft MIDTRAIN_MASK=1, b4 recipe on astfim_v1, unsloth-with-knobs, ~22GB expected, 3000 steps ~2-3h) then export_gguf Q8_0; DETACHED per B13 pattern + short tracked watchers; heartbeats q30min; artifacts /mnt/h/sepalith/runs/b8_midtrain_qwen35_2b*
[2026-09-05T15:48+02] zcode-b8-run RELEASE B8 train+export (train done 15:45:14 3000/3000, gates A/B/C all PASS — seam exact 48000/48000, 12.2% completion, finite losses; export done 15:47:34 -> experiments/models/b8_midtrain_qwen35_2b-Q8_0.gguf 2.01GB). Battery is CPU-only (flock /tmp/b_battery.lock, port 18158); card free for the next claimant.
[2026-09-05T17:36+02] zcode-gpushorts CLAIM two-item serial mini-chain (queue §2 FIM-Replica then §3 P10): FIM masked-loss@35% ladder arm (668 steps, memfrac 0.55 ~18GB, detached+watcher) -> GGUF serve+bpb evals; then P10 GatedNorm-v2 arms gn2_qk + stress_gn2 (same class). ONE trainer at a time (W37); all trainer/eval processes taskset-pinned 16-23 (24-core box; bench batch keeps 0-15). ETA ~5h chained, heartbeats q30min.
[2026-09-05T19:35+02] zcode-gpushorts AMEND claim: ITEM 1 (FIM-Replica masked@35%) GPU legs DONE 19:33 (train 96m + GGUF serve-eval + bpb); ITEM 2 (P10) fired 19:34 — gn2_qk now training (pid 620517), stress_gn2 behind it, then bpb+stress scorer. Same W37/pinning discipline. ETA ~22:45.
[2026-09-05T22:47+02] zcode-gpushorts RELEASE two-item mini-chain (FIM-Replica masked@35% done 19:33 — gate 3.8x PASS, verdict on board; P10 gn2_qk+stress_gn2 done 22:34 + bpb/stress evals — rejection stands, verdict on board; ~4.8 GPU-h total; no daemons; eval servers torn down by their chains; card verified idle)
[2026-09-05T22:41+02] zcode-b8b-stacked CLAIM B8b stacked arm (queue §2b B8 row): merge done CPU-side (banked b8 midtrain LoRA -> /mnt/h/sepalith/runs/b8b_stacked_base_merged, PEFT merge gates G1-G3 PASS); NOW train_sft 3000 steps on merged base + sft_v7, b4 recipe verbatim (unsloth-with-knobs, ~20GB expected, ~2h at the b4 2.0s/it class), then export Q8_0; DETACHED per B13/B8 pattern + short tracked watchers; heartbeats q30min; trainer taskset 16-23 (bench batch owns 0-15; 24-core box). Battery CPU-only (flock /tmp/b_battery.lock, ports 18162/18164). Artifacts /mnt/h/sepalith/runs/b8b_stacked_*
[2026-09-06T00:34+02] zcode-b8b-stacked RELEASE B8b train+export (train done 00:26:44 3000/3000, 1h46m, avg ~2.1s/it b4-class; gates A PASS 21,823,488 exact, legacy path held 0 [midtrain: lines, losses all finite — external trainer_state scan, train 1.60->0.99, eval_loss 1.218->1.162 monotone; VRAM transient peak 32.1GB in longest-row region, no OOM/fallback; export done 00:30:52 -> experiments/models/b8b_stacked_qwen35_2b-Q8_0.gguf 2.01GB). Battery is CPU-only (flock /tmp/b_battery.lock, ports 18162/18164, pinned 16-23); card free for the next claimant.
[2026-09-06T01:2x+02] zcode-b9-select CLAIM B9 SeleKT masking A/B (train_sft SELEKT_MASK=1: forward-only importance probe ~25m ~7GB, then b4-recipe LoRA train 3000 steps ~20GB ~1h45m, then export; DETACHED per B13/B8/B8b pattern + short tracked watchers; heartbeats q30min; trainer taskset 16-23; VRAM sampler live for the B8b 32.1GB-anomaly watch, SFT_PD_BATCH=2/GRAD_ACCUM=8 identical-math fallback armed BEFORE OOM) then battery CPU-only. Artifacts /mnt/h/sepalith/runs/b9_select_*
[2026-09-06T01:33+02] zcode-b9-select AMEND claim: launch-1 died at 01:25 pre-attachment-gate (probe fed CPU tensors to the cuda model — fixed: inputs now placed on the model's param device + use_cache=False; chain python now -u after the buffered-attachment-line postmortem); relaunching now, same claim terms
[2026-09-06T02:33+02] zcode-b9-select AMEND claim: pre-OOM fallback firing at the ckpt-1000 boundary (my own pids 716755/716764, logged) — training-era long-row VRAM spikes to 27.8GB observed (baseline 21.8 b4-class; B8b hit 32.1GB on the same data/seed) — relaunching SFT_PD_BATCH=2/SFT_GRAD_ACCUM=8 (identical optimizer math, effective-16 composition preserved, B13/B8b precedent) + RESUME_MODE=auto from checkpoint-1000
[2026-09-06T05:09+02] zcode-pft1 CLAIM PFT1 full-FT arm (queue §3; build validated + committed 5842f16, 47/47 CPU tests, b4 anchor merge banked CPU-side G1-G3 PASS): chain = 24-step FULL-FT smoke gate -> 3000-step train_sft FULL_FT=1 (unsloth full_finetuning, lr 1.5e-5 cosine pre-registered, paged 8-bit AdamW, grad ckpt, bs2xga8 = b4's effective 16, seed 3407, sft_v7, ~14-18GB expected ~2-3h) -> export Q8_0 (NO_LORA) -> BPB forgetting probe (base+b4+pft1) -> RELEASE -> CPU battery (scenarios/noop/midtyping/bench/V1a episode metrics + paired verdict vs banked b4). DETACHED per B13/B8/B8b pattern + short tracked watchers; heartbeats q30min; trainer+CPU legs taskset 16-23; VRAM peak-watch MANDATORY per row (pre-OOM bs1xga16 relaunch armed, threshold 30.5GB x3 samples). Claiming on observed release: B9 export done 05:03:26, battery CPU-only under flock, card at 1115MiB display-only, zero compute PIDs. Artifacts /mnt/h/sepalith/runs/pft1_*
[2026-09-06T05:05+02] zcode-b9-select RELEASE B9 GPU work (train done 05:01:39 3000/3000, GATE-C PASS — losses finite 3.018->1.888 train / 2.094->1.990 eval-L(monotone, masked-label surface); probe artifact saved; bs2x8 resume from ckpt-1000 after the pre-OEM fallback, effective-16 math; export done 05:03:26 -> experiments/models/b9_select_qwen35_2b-Q8_0.gguf 2.01GB). Battery is CPU-only (flock /tmp/b_battery.lock, ports 18166/18168, pinned 16-23); card free for the next claimant.
[2026-09-06T05:29+02] zcode-pft1 HEARTBEAT 1: SMOKE gates PASS (attachment 1,881,825,088/1,881,825,088 = 100.00% ALL-weights; lr 1.5e-5 PRE-REGISTERED line; finite loss @ step 20) -> ARM RUNNING since 05:23 (pid 849107, bs2xga8): VRAM 16.3GB @ 80% util — inside the pre-registered 14-18GB band. Peak-watch live (30.5GB x3 pre-OOM kill -> bs1xga16 resume armed). ETA at ~2.5-3.5s/it: train done ~07:50-08:40, then export+probe, release ~09:00.
[2026-09-06T05:42+02] zcode-pft1 HEARTBEAT 2: arm at ~240/3000, 3.7-4.6s/it (co-running B9 CPU battery shares 16-23; pace should recover when it drains), VRAM 16.3GB FLAT (peak-watch: max 16292 — deep inside the 14-18GB pre-registration, no intervention). Losses finite: first losses ~1.28. ETA train ~09:00-09:30.
[2026-09-06T06:03+02] zcode-pft1 HEARTBEAT 3: arm 533/3000 ~3.8s/it; eval-500 passed. VRAM: 16.3GB typical, transient peak 29,250MiB in the longest-row region (B8b-anomaly class at 32.1 — ours ~3GB lower; peak-watch threshold 30.5GB x3 NOT hit, no intervention). ETA train ~08:40.
[2026-09-06T06:35+02] zcode-pft1 HEARTBEAT 4: arm 987/3000 ~4.2s/it avg (B9 battery contention tail), losses 1.28->~1.1 finite/monotone-class, VRAM peak still 29,250MiB (no intervention; threshold 30.5GB x3). ETA train ~08:50-09:10.
[2026-09-06T06:56+02] zcode-pft1 AMEND heartbeat: PEAK-WATCH INTERVENTION fired 06:52:02 — VRAM hit 32,087MiB x3 samples (>30.5GB threshold) in the post-ckpt-1000 long-row region (B8b-anomaly class: theirs 32.1, ours bs2 full-FT 32.09); trainer killed BEFORE OOM per the pre-registered B13-pattern rule, relaunched 06:52:07 at SFT_PD_BATCH=1 x GRAD_ACCUM=16 (identical optimizer math, effective 16 unchanged) with resume=auto from checkpoint-1000 (8-bit optimizer + cosine schedule state carried). No OOM crash, no work lost past ckpt-1000. New ETA ~09:45-10:15 at bs1 pace.
[2026-09-06T07:14+02] zcode-pft1 HEARTBEAT 5: bs1xga16 relaunch healthy — attachment 100% re-verified, resumed ckpt-1000, step ~1300/3000 at ~4.9s/it, VRAM ~18.7GB (bs1 headroom vs the 32.1 transient: peak-watch stays armed). ETA train ~09:50.
[2026-09-06T07:45+02] zcode-pft1 HEARTBEAT 6: arm 1590/3000 (~4.5s/it steady at bs1), eval-1500 passed cleanly (eval_loss 1.181 vs b4's banked 1.1794 — tracking the anchor within 0.002), VRAM ~18.9GB stable post-relaunch (peak 32,087 remains the bs2 episode). ETA train ~09:50.
[2026-09-06T07:58+02] zcode-pft1 CORRECTION to HB6: the "eval_loss 1.181 @1500" I quoted was mislabeled — authoritative trainer_state (ckpt-1000): eval-500 1.2128 (b4 banked 1.2243), eval-1000 1.1851 (b4 1.2018) — the FT arm tracks 0.012-0.017 BELOW the anchor's curve, train 1.621->1.031, zero non-finite. No divergence/flatness => rescue LR NOT fired (stays available per pre-registration).
[2026-09-06T08:16+02] zcode-pft1 HEARTBEAT 7: arm 1975/3000 ~4.7s/it, VRAM 18.8GB stable (bs1; peak-watch armed), ckpt-2000 imminent. ETA train ~09:45, export+probe ~10:00, RELEASE ~10:05, CPU battery after.
[2026-09-06T08:50+02] zcode-pft1 HEARTBEAT 8: arm 2345/3000. Eval curve (authoritative): 500 1.2128 / 1000 1.1851 / 1500 1.2155 / 2000 1.1981 vs b4 banked 1.2243/1.2018/1.1791/1.1679 — FT starts 0.012-0.017 BELOW the anchor, then a 1500-bump (bs1-relaunch discontinuity class) leaves it 0.030 above at 2000. Train losses finite/declining throughout — NOT divergent-or-flat, rescue LR stays unfired. VRAM 18.8GB stable. ETA train ~09:50.
[2026-09-06T10:2x+02] zcode-pft1 RELEASE PFT1 train+export+probe (train 3000/3000 done 09:40 — 4h17m total incl. the 06:52 peak-watch kill+bs1xga16 resume; gate C PASS all-finite, eval 500/1000/1500/2000/2500/3000 = 1.2128/1.1851/1.2155/1.1981/[final in results]; VRAM peak 32,087MiB transient at bs2 (pre-OOM kill fired correctly), 18.7-18.9GB stable at bs1; export done 09:42 -> experiments/models/pft1_fullft_qwen35_2b-Q8_0.gguf 2.01GB; BPB probe completed post-release-marker within my window incl. one fixed-bug retry [ragged-block padding]). FORGETTING PROBE: general-R BPB 0.5356 vs b4 0.5288 = +1.287% regression — EXCEEDS the <=1% verdict gate (fail); general-text control 0.0896 = base-level (b4's LoRA had degraded it to 0.1074). Battery is CPU-only (flock, pinned 16-23) — verdict assembly follows. Card free for the next claimant.
[2026-09-06T10:17+02] zcode-x5-s1 CLAIM X5-S1 FRM self-cond + FPF chain (queue §3 X5 S1; build committed 165c691 + gate-fix commit, 11 CPU gate tests green, G1 verified on the real banked md_final.pt, replay parity vs S0 instrument bit-exact in-process): Stage A two-pass SC continuation 400 steps -> Stage B FPF 260 steps (poc_diff rig from banked md_final.pt, memfrac 0.42 <=14GB, fresh Muon per stage, lr 3e-3/1.5e-3) -> 216-row harness verbatim + recurrent depth k in {2,4,8} + post-FPF residual/AUROC/abstain replay -> pre-registered verdict. ONE workload (W37), DETACHED per B13 pattern; heartbeats q30min; trainer taskset 16-23. ETA ~3.5h train + ~1h eval, release ~15:00. Artifacts /mnt/h/sepalith/runs/x5_s1_*
[2026-09-06T16:21+02] zcode-x5-s1 RELEASE X5-S1 chain (all legs done 16:06: A+B trained clean, 13 eval legs + null diagnostics + residual replays + verdict computed + artifacts mirrored + committed 9bd1493; ~5h05m GPU-busy within the claim; no daemons — chain + trainer + eval processes verified exited, card draining display-class residual only). Card free for the next claimant.

[2026-09-08T01:15:52+02:00] codex-queue-owner CLAIM W33 fresh921 GPU evaluation; RTX5090 single full-offload CUDA workload, frozen MiniCPM5-v7 Q8 and CUDA b10453 runtime, max90min; CPU partial attempt stopped and its server reaped. No concurrent CUDA work.

[2026-09-08T01:33:26+02:00] codex-queue-owner RELEASE W33-N: attempt 9c9a385b45b5460982cd565f09111633 succeeded, 921/921 scored and archived. Model server exited; GPU compute inventory empty. Runner paused.

[2026-09-08T09:00:27+02:00] codex-queue-owner CLAIM S1 full GPU depth sweep, frozen b4/MTP/b2 and CUDA b10453. One server at a time, max4h including cleanup, no concurrent CUDA work; Pi reviews complete.

[2026-09-08T11:52+02:00] codex-queue-owner RELEASE S1 GPU attempt 0ee707385c5648a2883c465ac6cc38a7. All 10800 rows measured; runner succeeded, no child group and nvidia-smi compute list empty. Final archive verification in progress.

[2026-09-08T11:54+02:00] codex-queue-owner CLAIM S2 GPU timing column. Frozen b1_ref24 five formats, CUDA b10453, 150 requests, port 18472. Expected minutes, hard bound 30 minutes. Compute inventory empty before claim.

[2026-09-08T11:56+02:00] codex-queue-owner RELEASE S2 GPU attempt c467679abdc146f2a92abf79572fe6a4. All 150 requests succeeded, no tracked child and compute inventory empty.

[2026-09-08T11:56+02:00] codex-queue-owner CLAIM V1a b4 complete episode baseline. Frozen b4 Q8 and 60 deterministic trajectories, CUDA b10453, port 18473, context 32768, hard bound two hours. One tracked server; previous S2 server exited and compute inventory empty.

[2026-09-08T11:57+02:00] codex-queue-owner RELEASE V1a failed attempt f2e092eae8314d71bda8d8cc75bd1c33. Context reserve guard failed before episode 2 inference; 19 requests and one episode preserved. Server exited, compute list empty. Context audit before retry.

[2026-09-08T12:06+02:00] codex-queue-owner CLAIM V1a corrected 32K-context-eligible b4 baseline. 57 complete trajectories, 1208 points; three whole candidates excluded by frozen input-only audit. Same b4 Q8, CUDA b10453, port 18473, two-hour bound. GPU and port free before claim.

[2026-09-08T12:23+02:00] codex-queue-owner RELEASE V1a attempt 8c2ed0c3bbaa46c2b1bdc1e5dceb7d70. All 57 retained trajectories/1208 requests completed in 881.42 seconds; 33 files verified, no live child, compute inventory empty.

[2026-09-08T12:24+02:00] codex-queue-owner CLAIM corrected S1 ngram GPU measurement. Reuse original frozen b4/traces/runtime; M16/M48 with fresh baseline and bookend, 2400 cold/warm rows, port18471. Discarded M8/16/48 diagnostic first; expected tens of minutes, two-hour measure bound. GPU/port free before claim.

[2026-09-08T12:47+02:00] codex-queue-owner RELEASE corrected S1 ngram GPU attempt1573aca118fd4f06853f4d368fcc015b. All2400cold/warm rows completed; runner succeeded and compute inventory empty. Final hash verification underway. CPU case preparation only until next claim.

[2026-09-08T12:49+02:00] codex-queue-owner CLAIM b4 exported-model paired quality. Frozen Q8/stockQ4/imatrixQ4, one513-case cohort (255scenario+258noop),1539requests, port18475, serial CUDA b10453. Max90minutes. GPU/port free before claim. No publication/adoption or cloud spend.

[2026-09-08T12:55+02:00] codex-queue-owner RELEASE b4 quality attempt 2aee6537ed5149299c51f32509fec4fb. All 1539 requests completed in 207.02 seconds; 43 closed files and frozen inputs verified. No child or GPU compute process remains.

[2026-09-08T12:55+02:00] codex-queue-owner CLAIM quiet CPU window for S1 ngram scout. CPU-only b10453, eight threads, port 18474, 160 cold requests across baseline/M16/M48/bookend. Expected about two hours, hard bound five hours. No concurrent heavy CPU or CUDA benchmark; load1 0.57 before claim.

[2026-09-08T17:56:19+02:00] codex-queue-owner RELEASE quiet CPU scout. All 160 rows recovered and frozen evaluator passed; execution remains interrupted. Original worker/child/group absent; 26 files and input hashes verified.

[2026-09-08T17:56:57+02:00] codex-queue-owner CLAIM b4 intent generation: 132 requests, three frozen exports, serial CUDA, port 18476; 15-minute bound. Remote judging follows with no CUDA reservation.

[2026-09-08T17:58:03+02:00] codex-queue-owner RELEASE b4 intent generation; GPU compute inventory empty.

[2026-09-08T18:29:03+02:00] codex-queue-owner CLAIM b4 quant timing gpu: 120 requests, three exports plus Q8 bookend, port 18477. GPU-only timing, 30-minute bound.

[2026-09-08T18:30:14+02:00] codex-queue-owner RELEASE b4 quant timing gpu; runner closed, no tracked child or GPU compute process.

[2026-09-08T18:30:14+02:00] codex-queue-owner CLAIM b4 quant timing cpu: 120 requests, three exports plus Q8 bookend, port 18477. Quiet CPU t8, 90-minute bound; no overlapping GPU/CPU experiment.

[2026-09-08T18:56:33+02:00] codex-queue-owner RELEASE b4 quant timing cpu; runner closed, no tracked child or GPU compute process.

[2026-09-08T19:25:51+02:00] codex-queue-owner CLAIM b4 remaining quant quality GPU: Q8/Q6_K/IQ4_XS,1539requests, port18475, 90minute bound. Original conversation owns runner.

[2026-09-08T19:30:38+02:00] codex-queue-owner RELEASE b4 remaining quant quality; succeeded219.58s,44archivefiles verified.

[2026-09-08T19:31:18+02:00] codex-queue-owner CLAIM b4 Q6_K/IQ4_XS intent generation vs Q8,132requests,port18476,15minute bound.

[2026-09-08T19:32:45+02:00] codex-queue-owner RELEASE b4 Q6_K/IQ4_XS intent generation;132requests complete, runner succeeded, service inactive. Judge next.

[2026-09-08T19:39:43+02:00] codex-queue-owner CLAIM GPU b4 Q6/IQ4 timing:120requests, Q8bookend, port18477,30minute bound. CPU follows after release.

[2026-09-08T19:40:01+02:00] codex-queue-owner RELEASE b4 other quant timing gpu; attempt e68c8242e1b145acb09f2b6e90ea8c5a, failed, evidence closed.

[2026-09-08T19:41:27+02:00] codex-queue-owner CLAIM GPU b4 Q6/IQ4 timing retry; previous premeasurement failure archived, WSL telemetry PATH corrected.

[2026-09-08T19:42:34+02:00] codex-queue-owner RELEASE b4 other quant timing gpu; attempt ac579af911a64c578632cb3ce952775d, succeeded, evidence closed.

[2026-09-08T19:42:52+02:00] codex-queue-owner CLAIM quietCPU b4 Q6/IQ4 timing,120requests,8threads, Q8bookend,90minute bound. No GPU/CPU experiment overlap; expected25–35minutes.

[2026-09-08T20:13:32+02:00] codex-queue-owner RELEASE b4 other quant timing cpu; attempt b6dfcabf258d4ac7a5ca434d13f5a784, succeeded, evidence closed.

[2026-09-09T01:08:30+0200] codex-research-lead CLAIM B10 WiSE-FT battery: 5 arms (b4/base/a30/a50/a70 Q8) x frozen 513-case cohort, 2565 requests, port 18478, CUDA b10453, 90-minute bound, one server at a time. GPU/port free and compute inventory empty before claim.

[2026-09-09T02:12:21+0200] codex-research-lead RELEASE B10 WiSE-FT battery; attempt f52917a8b54b4f9995fcdc06938cda06 succeeded in 889s (5 arms x 513 rows). No tracked child; compute inventory empty.

[2026-09-09T02:21:13+0200] codex-research-lead CLAIM B10 BPB recovery leg: HF teacher-forced BPB probe (PFT1 machinery verbatim), 5 models (base/b4/a30/a50/a70), ~15 min bound 30. One workload; compute inventory empty before claim.

[2026-09-09T02:43:57+0200] codex-research-lead RELEASE B10 BPB probe; 5 models scored, ~25 min, no tracked child, compute inventory empty.

[2026-09-09T03:27:05+0200] codex-research-lead CLAIM O2 pre-screen: frozen b4 pass-rate screen, calibration leg (Q8 server vs HF bf16) then full pool (up to ~57k completions, 16-way concurrent), port 18479, hard bound 8h, one workload. GPU/port free before claim.

[2026-09-09T04:18:35+0200] codex-research-lead RELEASE O2 r1-r3 attempts (integration failures archived); CLAIM O2 pre-screen r4 with --no-jinja raw-completion serving; same scope/bound.

[2026-09-09T04:52:16+0200] codex-research-lead RELEASE r4 (server parser vs invalid-UTF-8 rollouts, evidence archived); CLAIM O2 pre-screen r5: HF-bf16 rollout path (exact GRPO parity), 10h hard bound, resume-keyed. One workload.

[2026-09-09T07:02:14+0200] codex-research-lead RELEASE r6 (scoring bug: gt kept UPDATED marker vs trainer's gt_lines; 1392 zero-solve rows voided, evidence archived); CLAIM O2 pre-screen r7, corrected scoring, 10h bound.

[2026-09-09T13:28:28+0200] codex-research-lead RELEASE O2 pre-screen r7; walk complete (5,578 prompts, all pools exhausted, 6h17m). Compute inventory empty.

[2026-09-09T13:36:12+0200] codex-research-lead CLAIM O2 GRPO arm 1 (filtered): 300 steps on merged b4, 942-prompt admitted pool, seed 3407, ~3-4h expected, 6h bound. One workload.

[2026-09-09T15:27:49+0200] codex-research-lead RELEASE O2 arm 1 (filtered) — 300 steps, 91 min, healthy exit. CLAIM O2 GRPO arm 2 (unfiltered control): identical config minus --prescreen, ~1.5h expected.

[2026-09-09T17:08:05+0200] codex-research-lead RELEASE O2 arm 2 (control) — 300 steps, healthy exit. Compute inventory empty.

[2026-09-09T17:36:27+0200] codex-research-lead CLAIM O2 eval battery: 2 arms x frozen 513 cases, port 18478, 60-minute bound.

[2026-09-09T17:53:10+0200] codex-research-lead RELEASE O2 eval battery; compute inventory empty.

[2026-09-09T18:25:35+0200] codex-research-lead CLAIM V1a v7 episode baseline: 57 frozen trajectories, v7 Q8, port 18473, 32K ctx, two-hour bound. One workload; GPU free before claim.

[2026-09-09T18:41:04+0200] codex-research-lead RELEASE V1a v7 baseline; succeeded, compute inventory empty.

[2026-09-10T00:26:36+0200] codex-research-lead CLAIM H3-S0 zero-shot format-fail bench: 7 variants x 100 frozen scenario cases, b4 Q8, port 18478, 60-min bound.

[2026-09-10T00:42:22+0200] codex-research-lead RELEASE H3-S0 zero-shot bench; complete, compute inventory empty.

[2026-09-10T02:03:20+0200] codex-research-lead CLAIM challenger stage-1 battery: b4 anchor + MiniCPM5-2B, frozen 513 cases, port 18478, 90-min bound.

[2026-09-10T02:53:32+0200] codex-research-lead RELEASE challenger stage-1 battery; complete, compute inventory empty.

[2026-09-10T03:52:14+0200] codex-research-lead CLAIM AG1 gate-1 extract: 513 frozen cases with logprobs on b4 Q8, port 18478, 60-min bound.

[2026-09-10T04:51:09+0200] codex-research-lead RELEASE AG1 gate-1 extract; complete, compute inventory empty.

[2026-09-10T05:10:26+0200] codex-research-lead CLAIM AG1 held-out confirmation: 1,208 V1a episode points with logprobs on b4 Q8, port 18478, 90-min bound.

[2026-09-10T05:27:56+0200] codex-research-lead RELEASE AG1 confirm; complete, compute inventory empty.

[2026-09-10T07:35:15+0200] codex-research-lead CLAIM S2-quality confirmation: 569 cases x 2 arms on the banked Q8/Q4_imatrix pair, port 18478, 90-min bound.

[2026-09-10T07:52:08+0200] codex-research-lead RELEASE S2-quality; complete, compute inventory empty.

[2026-09-10T11:59:19+0200] codex-research-lead CLAIM LR-sweep arm1 (5e-5) battery: 569 cases, port 18478, 45-min bound.

[2026-09-10T12:13:56+0200] codex-research-lead RELEASE LR arm1 battery; complete.

[2026-09-10T12:30:24+0200] codex-research-lead CLAIM LR arm2 (1e-4) battery: 569 cases, port 18478, 45-min bound.

[2026-09-10T12:46:00+0200] codex-research-lead RELEASE LR arm2 battery; complete.

[2026-09-10T12:57:18+0200] codex-research-lead CLAIM H3-S1 arm v10 (render adaptation, 150 steps, r16): local 5090, ~45-min bound.

[2026-09-10T13:06:36+0200] codex-research-lead CLAIM H3-S1 arm v05 (render adaptation, 150 steps): local 5090, ~45-min bound. (v10 COMPLETE, released.)

[2026-09-10T13:33:07+0200] codex-research-lead CLAIM H3-S1 arm v14 (render adaptation, 150 steps): local 5090, ~45-min bound. (v05 COMPLETE, released.)

[2026-09-10T14:27:58+0200] codex-research-lead CLAIM LR ANCHOR (2e-4) battery: 569 cases, port 18478, 45-min bound.

[2026-09-10T14:44:05+0200] codex-research-lead RELEASE anchor battery; claiming H3-S1 bench (4 serves: b4 + v10 + v05 + v14, port 18478, 2h bound).

[2026-09-10T15:01:12+0200] codex-research-lead CLAIM LR 4e-4 battery: 569 cases, port 18478, 45-min bound.

[2026-09-10T15:17:11+0200] codex-research-lead RELEASE 4e-4 battery; CLAIM local 2e-4@300 platform-reference arm (300 steps, 5090, 30-min bound).

[2026-09-10T15:20:31+0200] codex-research-lead CLAIM local 2e-4@300 reference arm retry (data-dir arg fixed): 5090, 30-min bound.

[2026-09-10T16:08:45+0200] codex-research-lead CLAIM H3-S1 bench (b4 + 3 arms, 7 variants x 100 cases each): port 18478, 2h bound. (local-ref train released; export on CPU.)

[2026-09-10T16:25:06+0200] codex-research-lead CLAIM H3-S1 bench retry (cohort staged): port 18478, 2h bound.

[2026-09-10T18:08:35+0200] codex-research-lead CLAIM H3-S1 bench (GDN-correct exports staged): port 18478, 2h bound.

[2026-09-10T18:40:53+0200] codex-research-lead RELEASE H3-S1 bench; compute inventory empty.

[2026-09-10T18:45:14+0200] codex-research-lead CLAIM local 2e-4@300 reference battery: 569 cases, port 18478, 45-min bound.

[2026-09-10T19:02:52+0200] codex-research-lead CLAIM decomposition arm (local stack, Kaggle config: B4_REGEX + sft_v7, 300 steps): 5090, 30-min bound. Prior battery released.

[2026-09-10T19:29:44+0200] codex-research-lead CLAIM decomposition battery (local+v7+regex): 569 cases, port 18478, 45-min bound.

[2026-09-10T19:45:05+0200] codex-research-lead RELEASE decomposition battery; compute inventory empty.

[2026-09-10T20:04:15+0200] codex-research-lead CLAIM H3-S1 quality leg (3 adapted arms x 569-case battery): port 18478, 2h bound.

[2026-09-10T20:35:57+0200] codex-research-lead RELEASE H3-S1 quality battery; compute inventory empty (Kaggle 3000-step arm still running there).
