# E1000 native DEV preparation

This packet prepares a fresh CPU merge, F16/Q8 export, and corrected native DEV75 run for the E terminal checkpoint. It binds recipe `2e8038f81c8fee2112b7164e4e0c92c11474459dd1e810f8a86c2bf224ca2316`, frozen source `bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384`, full step 1000, and sampler cursor 16000. A step-500 checkpoint, another source, or another recipe is rejected.

Root runs the arrays in `root-commands.json` only after E/full1000 is complete and independently admitted. Run `merge_guard`, `prepare_export`, `export_guard`, `review_export`, and `bind_profile` in order. The binding helper calculates and verifies the completed metadata hashes itself, so the command has no placeholder. Then run the three `observation_sequence` arrays exactly as written: `observe_client.py` has no observation argument; `build_closure.py` follows; the last invocation alone uses `--check-policy-only`. Root completes a fresh native approval and runs `native-command-v1.json` under its CUDA lease.

After a complete E1000 native DEV75 result exists, root runs `comparison`; its wrapper calculates the result hash and the comparator records it. The comparison pins the incumbent, C250, and D500 result bytes and checks all four runs use the same 75 case IDs, corrected panel, prompts, targets, families, operations, packages, and no-op labels. It creates development evidence only and never makes a promotion decision.

No checkpoint, model, GGUF, or DEV output was read during preparation. The copied D500 harness remains byte-identical except for explicit E lineage checks and removal of stale generated observation/profile artifacts.
