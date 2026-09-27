# Resource ownership

This is a manual handoff ledger for the lead agent. Task status does not grant
GPU ownership or prove that a process has stopped. Inspect the real process
and current owner before changing this ledger. The campaign tool launches no jobs.

| Resource | Owner | Task / job identity | Observed at | Release condition |
|---|---|---|---|---|
| RTX 5090 | lead SFT-11 | Corrected ordinary fullweight continuation guard1831562 / trainer1831597 active; native allocator verified; in-process restore checks | 2026-09-14T23:20UTC | Bound579269f0...; checkpoint90 cursor384 offset66 to194; micro1GA16; guard10800s; no competingCUDA |
| Notebook CPU / editor | notebook_eval reserved DAT-10 | SSH reachable18:43Z load0.00; bounded source/parser checks permitted; no remotePID yet | 2026-09-14T18:46:55.964325+00:00 | Max2CPU20min;TRAIN necessary inputs only; record actual remote launch/terminal |
| Local CPU | lead/data_expansion DAT-10 | Replay1689900/1691118; combined16K cache1751493; semantic shards0to4 controller1777225 | 2026-09-14T22:25UTC | Semantic shard5 terminal1878rows; no training admission; bounded low-priorityCPU |
| External queue | None | Opuswarmstart review terminal; confirmed fixes tested | 2026-09-14T00:18UTC | No external job active |
| Anyscale | no active campaign job | All3 p4d probes and clusters terminal; currentusage has no p4d and no creditdelta; reservations released | 2026-09-14T23:24UTC | Balance86.2113; cumulative13.7887; ceiling60; headroom46.2113; freshbilling required before futurepaid launch |
| Azure | None | Fresh active-subscription VM and VMSS lists empty; GPU quota not rechecked | 2026-09-14T20:21UTC | No allocation; EUR100 campaign ceiling unchanged |
| Final evaluation | lead | Metadata/source locks only; content sealed | Actual freeze pending | Actual weights and harness freeze receipt; previous calendar cutoff superseded by user |


The user confirmed that the current experiment owner has been told to stop
after currently running experiments complete. The campaign lead announced
the handoff in canonical `comms/board.md`. The old owner completed SESSION WRAP at06:17:47UTC on September12. The historical battery process exited; the lead verified release before subsequent probes.
On WSL, `nvidia-smi --query-compute-apps` returned no process rows despite the
live training process and occupied GPU; inspect the process and telemetry too.

Primary campaign implementation will use the separate worktree
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`, created from
owner commit `a79d6de`. The running owner source remains unchanged. Required
dirty source is preserved in PRE-01 snapshots and must be reviewed into each
launch snapshot explicitly.

Record a new owner, task ID, actual PID/provider job ID and timestamp before a
launch. Release only after verifying completion or termination. A crashed or
disconnected agent does not release its process. GPU evaluation and speculation
benchmarks use the same lease as training. CPU serving work can run alongside
training, with contention noted in measurements.

Workers receive a file scope and resource scope in their delegation packet.
Changing either returns to the lead for scheduling. The user's existing
authorization for the campaign and budgets remains in force.

CPU DEV comparison lease: lead, SFT-08/RUN-03 b4-dev-v2, localhost18099 t2/ngl0, admitted 2026-09-12T14:12:04.888690+00:00; actual PID in native training/SFT-08-b4-dev-v2/launch.json. Server stops within1000s; clientwithin900s. No competing CUDA.

RUN-06 2026-09-12T14:20:37.014595+00:00: notebook baseline PID66359/supervisor66356 terminal and port18401 free. Lead admits primary500-ngram-a sameCPUprofile/weights, max1200s, diagnostic8requests60seach. Baseline1/8 real5s, diagnostic8/8; prefill dominates.

RTX5090 lease 2026-09-12T14:23:06.634220+00:00: primarya attemptfa4aef59b610476ab5d408f8ef6ff4d0 terminal1000, all3PIDs gone; lead admits primary3000-b fromfull1000 with same26a07c source/schedule, stop2000. RunnerPID/attempt recorded immediately afterlaunch. ReceiptSFT-03-primary-b-admission.json.

Notebook build lease 2026-09-12T14:36:41.792641+00:00: lead RUN01/06 backend-build-a; ngraṁ server67614/supervisor67613terminal and18401/2/3free. CPUaffinity0,1; sequentialOpenBLAS900s/Vulkan1500s plusconfigure, outer2760s. No models/latency testsduringbuild. ActualPID remote runs/backend-build-a/launch.json.

Current verifiedCUDA owner 2026-09-12T14:38:06.491213+00:00: primaryb attempt262f5753dd77463e9ccc9dcccc1ba237, supervisor3221054/timeout3221228/training3221229, step1186; next2000. Priorprimarya1000 andlocalb4CPUv2 areterminal. Notebookbackend-build-a onCPU0,1 only; no modelserver active.

Global CUDA lock designated 2026-09-13T17:06:06.131719+00:00: `/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock`. All future CUDA launchers must hold a nonblocking exclusive lock through process cleanup. Never unlink or replace this file. Historical controllers require an outer `flock` wrapper if reused. All earlier CUDA jobs were verified terminal before this migration. The manual table still records owner, admission and release evidence.


Native data staging observation 2026-09-15T00:03:42.703518+00:00: lead owns CPU-only low-priority copy PID1881489 (timeout1881488, exec86061), maximum1800s, 9,330,242,806 bytes. Reuses admitted CPT cache/rows/schedule; no model copy and no CUDA use. Active CUDA remains guard1831562/trainer1831597. Receipt: receipts/SFT-11-native-reusable-data-root-stage-launch.json.

Native data staging complete 2026-09-15T00:07:31.132276+00:00: timeout1881488/worker1881489 terminal exit0. Root independently rehashed all7files/9,330,242,806bytes. CPU/IO staging lease released. Runtime migration remains unadmitted. Receipt: receipts/SFT-11-native-reusable-data-root-stage-terminal.json.

CUDA owner 2026-09-15T00:21:24.452188+00:00: lead diagnostic guard1931870 (exec26458), max2400s, checkpoint66 packed-versus-padded numerics, no optimizer step. Prior guard1831562/trainer1831597 verified terminal0. Checkpoint194 independent CPU verification exec84611 runs alongside. No other CUDA launch until diagnostic guard terminal.

CUDA handoff 2026-09-15T00:29:14.471370+00:00: diagnostic guard1937112/child1937203 terminal0 and absent. Lead admits checkpoint194 matched anchor2k/8k/16k evaluation via r2-cpt194-review-root-v1/guard-command.json, maximum2400s, single CUDA lock. Receipt SFT-11-diagnostic-terminal-cpt194-eval-admission-root.json.

Verified CUDA owner 2026-09-15T00:32UTC: checkpoint194 matched evaluator guard1968957 / child1969153, exec5406, maximum2400s. Diagnostic terminal0. CPU root native resume proof exec31453, CUDA hidden.

CUDA released 2026-09-15T00:32:39Z: checkpoint194 evaluation guard1968957 and child1969153 terminal0, absent; allthree matched panels root verified. Native194to322 continuation is next, preparation only until lead admission.

CUDA admission 2026-09-15T00:43:49.408971+00:00: lead SFT-11 native194to322, CPU frontdoor PASS185318rows/cursor2048, source93b80b..., bound076b9105..., guard10800s/14GiB admission/6GiBsoft4GiBhard. Fresh native and E paths preserve194. Launch PID pending.

CUDA owner verified2026-09-15T00:44UTC: native194to322 guard2043215/attestation2043348, exec40910. Launch accepted15116MiBhostavailable; startupattestation underway. No competingCUDA untilguardterminal.

CUDA retry 2026-09-15T00:48:10.028282+00:00: guard2043215/attestation2043348 terminal1 beforemodel load, allPIDs absent. Lead retry-b after creating fresh archive forstartuptelemetry; same recipe076b9105 andsource93b80b, freshguardoutputsuffix-b.

CPU lease 2026-09-15T00:50:15.813153+00:00: root sourcewalk27to40 controller2067232 timeout2067237, exec95888, max7200s, cores4,6/nice10/idleIO, exclusiveexistingoutputlock. Separate worker data_expansion admitted10952fullpreeditreconstruction cores8,10,max45min; actualPIDforthcoming. CUDAguard2064033/attestation2064125 exclusivelyownedbyroot.

CUDA training verified 2026-09-15T00:55:10.776025+00:00: guard2064033/attestation2064125/trainer2068268,exec27712. Actualstep195complete,381finite/nonzerogradients,29.9GBpeakreserved,100percentGPU. Stop322/cursor4096. Firstattempt2043215terminal1beforemodelloadpreserved.

- 2026-09-15 01:27 UTC — SFT-11 root handoff controller PID 2112470, exec64714, packet `r2-cpt322-review-root-v1`: waits for current CUDA trainer2068268/guard2064033 terminal, verifies complete322 native/E payloads, then acquires the existing exclusive CUDA guard for matched development evaluation only. No training continuation/promotion. Current CUDA ownership remains2064033 until exit. Receipt `SFT-11-cpt322-evaluation-handoff-root-launch.json`.


### 2026-09-15 02:06 UTC — checkpoint322 paired canary

Lead controller 2179203 owns the next sequential CUDA guards: eight ordinary-reference updates, then eight varlen-candidate updates, both from independently verified checkpoint322. CPU preflights run first. Prior CPT322 and development evaluation guards are terminal. Existing stable CUDA flock remains mandatory; each arm has a 3600-second guard and 14 GiB host-memory admission. No production continuation is admitted. CPU rendering lanes 2144808/2144809 continue on cores 0/2 for 10,948 examples. No new cloud jobs launched. Receipt: `receipts/SFT-11-varlen322-root-launch.json`.


### 2026-09-15 02:14 UTC — sourcewalk merge and semantic continuation

All 41 replay shards are terminal and merged with 76,279 candidates. Semantic controller 2185670 and worker 2186251 own cores 8/10 for shards 27–40 under the 1,500-second timeout 2186250. Existing render lanes retain cores 0/2. No CUDA or cloud use is admitted to these workers. The ordinary CUDA canary completed eight updates and is publishing its full checkpoint before the packed arm can launch.


### 2026-09-15 02:21 UTC — packed canary admission recovered

Ordinary guard 2182311 completed successfully. Original packed guard 2236120 did not launch a child because available host memory was below 14 GiB. Root released clean cache from the completed ordinary checkpoint model and optimizer only; available Windows memory recovered to 24,835 MiB. Fresh controller 2254498, guard 2254504 and child 2254630 now own CUDA under the unchanged recipe and thresholds. Session 45709. Guard output has suffix `-b`; prior failure evidence is preserved. Payload-verifier controller 2234621 is terminal without execution; notebook_eval prepares a v2 verifier bound to the new terminal evidence.


### 2026-09-15 02:29 UTC — paired readout and CPU ownership

Packed guard 2254504 is publishing after all eight updates. Payload verifier 2281062 (session 96892) waits for controller 2254498, then owns cores 12/14 for full native and durable checkpoint verification. No automatic CUDA evaluation is authorized. No-op reconstruction controller 2268481 is terminal after a duplicate-geometry invariant failed; it has released cores 12/14. train_adapter prepares v3 that retains such candidates until provider-based deduplication. Existing semantic and render CPU leases continue.


### 2026-09-15 02:54 UTC — profiles and resumed semantic work

Paired training, full payload verification, both matched evaluations, and the attention microprobe are terminal. Full-state editing profile controller 2404414 (session 15891) owns sequential CUDA guards for 4K, 8K, 16K and 32K after a successful 2K probe. It releases only exact clean checkpoint cache before unchanged admission and stops on the first runtime failure. Semantic controller 2354583 / timeout 2355237 (session 24568) owns cores 8/10 for missing shards 35–40 while reusing 27–34; timeout is 5,400 seconds. No-op v3 reconstruction is terminal; provider remains unlaunched pending recovery of unchanged nonempty selections and CRLF cases.

- 2026-09-15 03:13 UTC: lead CUDA selected packed330 to ordinary450 continuation active; guard 2473526, controller 2473524. Update331 verified381 finite/nonzero gradient tensors. Previous missing-output-directory attempt terminal before updates. All profile guards terminal. Semantic27–40 and10948render16 jobs terminal; longer-context fallback preparation CPU0/2. All three Anyscale probes freshly TERMINATED03:01; no new paid launches.

- 2026-09-15 03:15 UTC: CPU0/2 fallback renderer root controller2477014, lanes2477233/2477234, max3600s each, 485 unchanged prediction inputs at32K/reserve2048. CUDA continued through336, WindowsAvailable25390MiB, no driver events.

- 2026-09-15 04:01 UTC: CUDA STOPPED: guard2473526 terminated training at03:45:13 for persisted host-memory soft-floor failure; completed421 unsaved, packed330 preserved. No current CUDA owner process. Lead reserves recovery; train_adapter prepares cadence/resume without CUDA or bulk reads. Semantic9534 render controller2513466 terminal03:47:28 (failure under review); no-op4100 reconstruction terminal. Notebook quality staging ready, not launched. All large checkpoint rehashes and broad worker scans stopped; schedule bulk I/O outside CUDA. Redundant native ordinary330 evicted only after independently verified E archive; selectedpacked330 and322 unchanged. Anyscale probe authenticated TERMINATED03:58; no newpaid launches.

- 2026-09-15T04:11:02.730412+00:00: Notebook CPU0/2 lead RUN-06 E750 cap-quality run e750-cap-quality-20260915T040000Z active, runner1343969/start24060642, first cap192 server1343993/start24061024, localhost18527. Root watchdog28800s, offline120s cases; no production latency claim. Local no-op provider lead controller2556499 (exec73655) CPU4/6 max9000s; firstshards10/12 complete. CUDA recovery CPU frontdoor exec86987 in progress; no CUDA launched yet. No competing bulk reads admitted.

- 2026-09-15T04:21:06.984407+00:00: CUDA lead recoveryv2 controller2630015 guard2630027 attestation2630218 trainer2634519, exec35038. Actualupdate331verified381finite/nonzerogradients, full2.516B training. Resume330cursor4224;save/stop354cursor4608;cadence24. AllotherCUDAjobsabsent. No competing bulk reads. NotebookCPUquality andno-opCPU4/6 continue.

- 2026-09-15T04:55:28.332195+00:00: Lead CUDA continuation354→450 active: controller2699989, guard2700261, attestation2700314, trainer2700542. Cadence24; latest completed update 372. Previous354 recovery and matched evaluation terminal, source354 accepted. Notebook quality6 runner1353768/server1353834 on six physical cores, port18529, watchdog28800s. Previous notebook two-thread run and pilot terminal. Both local provider render controllers terminal; Luna prepares repairs CPU-only. No competing bulk reads. Anyscale authenticated probe TERMINATED; no new paid launches.

- 2026-09-15T05:11:03.200344+00:00: CUDA continuation354→450 remains sole owner, completed402 and publishing. Root watcher2721744 (exec61883) waits exact training terminal then reservesCPU12/14 for full450 payloadverification, timeout1800s, no CUDA/retries. Notebook six-core quality run remains live1353768/1353834. All34 current-user visibleAnyscalejobs terminal05:07; freshbalance86.211311808USD. No new paid launches.

- 2026-09-15T05:15:32.392926+00:00: Root semantic9534 retry controller2726253/timeout2726289 (exec32482) activeCPU8/10, outer7200s, fresh render-16k-retry-rich-v1. All14inputshardhashes verified; firstoutputs supportedtarget-free. Oldpartialrenderpreserved. CUDAtraining412+,402durable. NotebookCPU6stilllive. No-opretrypreparationCPUonly, no launch yet.

- 2026-09-15T05:26:04.500287+00:00: No-op4100 freshparse retry leadcontroller2778430/timeout2778432 (exec85886) ownsCPU4/6, outer7200s, firstbothlanes supportedtarget-free. Semantic9534 remainsCPU8/10. CUDA434+,426durable,450CPUwatcherreserved12/14 onlyaftertrainingterminal. Notebookcap192completed andserverexited; cap384server1361367 tick24502540 live, runner1353768 same. No newcloudlaunches.

- 2026-09-15T05:31:59.117078+00:00: Final450save active. Root matched-eval controller2816227 (exec11805) reserves nextCUDA onlyafter verifier2721744 success and allpriortrainingPIDsabsent; guard2400s, no retries, no auto scientificpromotion. CPUrender ownersunchanged8/10 and4/6. UniongeometryprepCPUonly.

- 2026-09-15T05:50:53.217711+00:00: Lead reserves CPU12/14 frontdoor and subsequent exclusive CUDA guard for accepted checkpoint450 to706, cadence64, controller r2-cpt450-to706-cadence64-root-v1. Prior450 training and matched evaluation terminal and root accepted. Semantic retry remains CPU8/10; no-op retry terminal0 pending review. No paid launch.

- 2026-09-15T05:58:41.638115+00:00: Prior450to706 v1 failed before model load and v2 stopped during optimizer restore at2869MiB hostavailable, no updates. Fresh v3 controller3038867/guard3040840 exclusive CUDA; same recipe and guard thresholds, current data cache release confirmed, repeated clean source450 cache advice during restore. Source450 intact. No-op render terminal; semantic CPU8/10 continues.

- 2026-09-15T06:09:28.343242+00:00: Lead no-op228 fallback32 controller3092201/timeout3092202 session58046 owns CPU4/6, max3600s, target-free same228 inputs. Semantic9534 remainsCPU8/10; soleCUDA3038867/3040840 through477. Notebookcap768 active; no paid launches.

- 2026-09-15T06:17:49.068193+00:00: No-op32Kv1 controller3092201 terminal1 wrongprovider; corrected v2 controller3101514/timeout3101515 terminal0 all31shards228rows. CPU4/6render released. Root terminal-verifier session30981 waits semantic2726253 terminal then ownsCPU12 for exact14shard verification max180s. train_adapter CPU4 light data preparation38rows64K, no renderer. CUDA3040840 remains sole owner through493.

- 2026-09-15T06:23:09.181954+00:00: Root706verification watcher3122694/session48360 waits training3038867 terminal, then CPU12/14max1800s fullnative/E hashes; no CUDA launch. Root geometry diagnostic3124236/session39933 CPU6, two sequential max600s cohorts15006/10682; review-only. Semanticproducer2726253 terminal0, verifier root-v1 failed schema binding before outputs; microbatch preparesv3. No-op32Kallterminal;64KsourceprepCPU4. SoleCUDA3040840 through509.

- 2026-09-15T06:28:33.881374+00:00: No-op64K leadcontroller3129530/timeout3129531/session82092 CPU4/6 max3660s outer,38rows. Geometrydiagnostic3124236 terminal0; CPU6 released before64K. train_adapter CPU2 bounded5185cohortlocator metadata. CPT514 durablebytrainer; CUDA3040840 continues;706CPUwatcher3122694 dormantuntilterminal.

- 2026-09-15T06:40:17.446084+00:00: No-op64K3129530 terminal0;38rows rootverified29supported9holds. No-op128K3137723/session99322 terminal0 all9rows, outputreviewpending; CPU4/6free. Notebookquality6allcaps terminal, runner1353768 andserversabsent perterminalmonitor; port18529free. SoleCUDA3040840 through542;706verificationwatcher3122694 remainswaiting.
