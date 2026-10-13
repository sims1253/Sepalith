The current pre-admission blocker has a strong guest file-cache signature. Retry B never loaded its model. Root measured about 36.6 GiB of Linux file cache, 1.6 GiB of anonymous pages, no trainer, and a 37.9 GiB vmmemWSL working set. Owned Midtrain eviction reduced its resident pages to zero; the subsequent inventory found no resident pages in the inspected campaign model files. Windows available memory nevertheless stayed below the unchanged 8 GiB admission floor.

One root-owned pre-admission clean page-cache reclaim (`drop_caches=1`, followed by compaction), checked against the current WSL boot ID, is supported after the narrower owned-file operation proved insufficient. Retain the existing guard and its fresh Windows admission measurement. Record command completion or timeout separately from observed memory recovery. The earlier 15-second timeout remains a partial outcome. Do not automatically insert global reclamation into optimization: it can perturb unrelated file-reading latency and needs separate placement review if pressure recurs.

The original smoke's exact process-RSS/WDDM contribution remains unresolved because it had no PID/RSS time series or WDDM host-memory counters. Its retained before/after interval also spans trainer shutdown. The new no-model measurements provide stronger evidence about the present admission blocker, not a retrospective complete allocation trace.

The sampler is ready for an explicit root launch after a fresh guard has written `launch.json`. Root supplies the verified guard PID, start tick, its output path, and a new sampler output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-host-pressure-diagnosis-v1/sample_host_memory.py \
  --root-pid "$cpt_guard_pid" \
  --root-start-tick "$cpt_guard_start_tick" \
  --windows-log "$cpt_guard_output/host-memory.jsonl" \
  --output "$cpt_pressure_output" \
  --seconds 600 --interval-seconds 15
```

The sampler verifies the guard's launch/PID identity. It reads that guard's descendant PID/start-tick/RSS records, Linux memory counters and the existing guard's Windows readings. It adds no PowerShell query, model/data read, cache action, process signal or persistent service. It stops after its bounded duration or loss of the guard identity. Linux/Windows timestamps and sampler read overhead are recorded. Stale Windows readings and incomplete PID scans are marked explicitly. Summed RSS may double-count shared pages and cannot establish PSS, USS or WDDM ownership.

Nine CPU test groups passed, including PID reuse, reparenting, changing identities, scan bounds, malformed/stale Windows records and one read of the test process's own RSS. No monitor was launched. Large training-file validation remained paused during this diagnosis.
