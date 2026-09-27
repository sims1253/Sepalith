# No-op 4100 64K retry preparation

This packet materializes only the 38 prediction-time inputs whose first 32K hold reason is `complete_span_not_applicable` (18) or `complete_span_unresolved` (20). It retains the other 67 exact hold records. The accounting remains `3995 supported + 38 queued + 67 retained = 4100`; no row is excluded or admitted for training.

The generated inputs preserve the original JSONL line bytes and accept the two reviewed reconstruction schemas. They reject any top-level target, gold, or label field. The selected-policy manifest binds the full 4,100-row selected-context file by its reviewed SHA without rescanning that 49.9 MB file; classification comes from the separately hash-reviewed 32K outputs. Preparation scanned about 70.3 MB across the 32K outputs and original inputs.

`run64_lanes.sh` remains dormant. After root admission it invokes the exact reviewed no-op renderer wrapper and source (`114b0a…`) with a 65,536-token context, 2,048-token reserve, CPU cores 4 and 6, and a 3,600-second lane deadline. The unchanged inner wrapper has an 1,800-second per-shard deadline. A 128K escalation template remains fail-closed until an exact 64K terminal review identifies any still context-only rows.
