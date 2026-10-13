# Full-weight CPT learning-rate pilot v1

This packet compares two Aurora-mix full-weight learning-rate pairs from the same admitted parent and the same immutable 384-draw schedule:

- hidden `1e-4`, side `1e-5`
- hidden `3e-4`, side `3e-5`

Each arm uses micro batch 1, gradient accumulation 16, 24 optimizer updates, and `constant_with_warmup` with exactly two warmup updates. The panel contains one exact 8,192-token all-token TRAIN chunk from each of 384 distinct packages and documents. It has 3,145,728 input tokens and 3,144,960 causal-loss tokens. Every source document was completely reassembled before the selected chunk was emitted. The panel has no package or document overlap with the fixed 499-row holdout and does not truncate or pad a training row.

The panel is a short LR experiment. It is not a production dataset cap, coverage claim, or substitute for the all-eligible cohort. Unselected eligible data remains queued.

The runner is the reviewed full-weight CPT trainer v2 closure with three scoped changes: pilot-specific resume identity and policy gates, a parent baseline evaluation before Trainer and optimizer construction, and the same fixed holdout evaluation after terminal update 24. Each arm writes only one terminal full checkpoint. Runtime training loss is telemetry; root selects an LR from finite behavior and change in fixed-panel mean causal NLL.

The current templates bind the saved merged CPT-250 parent. A cloud checkpoint at 1585 or 1902 requires a fresh template with exact parent files and identity; it cannot be substituted into these bindings.

At the measured warm 8K all-token forward/backward time of about 2.67 seconds per microstep, 16 microsteps plus the observed roughly 5.6–5.9 second optimizer step imply about 19–20 minutes of update compute per arm. This excludes model load, both 499-row evaluations, and terminal checkpoint I/O. Each evaluation processes 662,269 input and 661,360 loss tokens. The terminal full checkpoint is estimated at 17.251 GB; the measured E path took about 423 seconds through its sealed manifest. Two completed arms therefore retain about 34.502 GB before small reports, subject to root cleanup policy.

`root-commands.json` contains separate bind, template/bound preflight, run, and resume argv arrays. Root owns admission, CUDA lease, host guard, execution order, and stop decisions. The two arms must not run concurrently.
