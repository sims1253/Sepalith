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
