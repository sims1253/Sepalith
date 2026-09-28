# RUN-01 export A memory review

This note is read-only evidence review. It did not read any model bytes, inspect tensor contents, launch a process, clear a cache, compact WSL, or change source.

## Evidence

The export-A guard recorded three Windows host samples:

| UTC | AvailableMBytes | delta vs 12 GiB (12,288 MiB) | commit ratio | page input/sec | page reads/sec | NVIDIA events |
|---|---:|---:|---:|---:|---:|---:|
| 2026-09-12T22:52:18.0799885Z | 16,889 | +4,601 | 0.602450 | 0 | 0 | 0 |
| 2026-09-12T22:52:34.6784724Z | 9,009 | -3,279 | 0.663800 | 175 | 11 | 0 |
| 2026-09-12T22:52:51.1269985Z | 6,968 | -5,320 | 0.689778 | 107 | 11 | 0 |

The v3 policy correctly stopped after two consecutive samples below its 12,288-MiB soft floor. The hard 4,096-MiB floor and 90% commit rule were not reached, and no NVIDIA driver event was reported. This is host-memory pressure, with paging activity, rather than evidence of an OOM kill.

The guard terminal record is stopped_or_failed, reason host_free_memory_below_soft_floor_persisted, child exit -15, elapsed 32.10388604499167 seconds. The F16 converter terminal record reports exit 0 in 17.250138356001116 seconds and cumulative_child_maxrss_kib=2443860 (about 2.33 GiB). On Linux, resource.getrusage(RUSAGE_CHILDREN).ru_maxrss is a high-water maximum for children, despite the field name; it is not a sum and only the completed converter command was recorded. The Q8 launch record exists at 22:52:39.927233Z; the Q8 output is partial at about 1.928 GB per root's report, and no Q6 command was launched.

## Retry assessment

A retry from the completed F16 is operationally safer than rerunning the converter if root first verifies the F16 file's size, GGUF header, full hash, and metadata closure without accepting the partial Q8. The existing export wrapper cannot resume: it always requires a fresh output directory and its command list starts with conversion. Use a separate fresh quant-only output directory and run each quantizer under the v3 guard, with a separate guard evidence directory. Preserve the failed export-A directory as evidence.

Before the retry, root may perform one targeted advisory cache drop for the completed F16 file after confirming no owned process maps or opens it, recording inode, size, and mtime before and after. This can reduce clean file-backed pages without changing file contents. It is not a guarantee of available memory and the quantizer will reread the F16. Do not use global cache flushing or a compaction loop.

The existing post_load_cache.release_after_load helper is not a direct fit for this retry: the v3 --release-cache-file admission accepts only an existing file named model.safetensors, and the helper waits for the training-loader marker Unsloth 2026.8.18 patched 42 layers. It therefore cannot be passed the generated model-F16.gguf through the guard's release option. Its default path also requests one-shot WSL compaction; any targeted use must avoid that side effect. A root-owned direct advisory with POSIX_FADV_DONTNEED on only the closed F16 file is the narrower option.

The measured converter high-water is below 12 GiB, but it does not establish quantizer peak RSS. The memory fall occurred before and during Q8 launch, and the partial Q8 does not prove that Q8 or Q6 can complete under the revised floor. Root should record per-command wall time and child RSS again on the quant-only retry, and stop on two low samples.

No quality, artifact-completeness, or successful retry claim is made here.
