# RL-11 expanded prediction contexts

This packet materializes a prediction-only context sidecar for all 11,505 rows in the current root-reviewed expanded corrected-plus-short TRAIN pool. It uses explicit limits of 4,096 full-sequence tokens and 1,024 target tokens including protocol EOS. No row is truncated, and the legacy 2,048/192 filter is not used.

The output contains 7,910 legacy context records copied as exact raw bytes, 3,184 corrected records rejoined to admitted DAT-05 provenance and their exact candidate envelopes, and 411 accepted short records rejoined to their source packets. The independent streaming verifier confirms exact row/provenance/sidecar order, 11,505 distinct IDs, exact prompt rendering, static source and replacement geometry, no CPT-validation identity, no duplicate prompt, and no target or reward key anywhere inside `context`.

Source audit evidence remains under `source_identity`, outside the model prediction context. Some provenance records contain target-related audit facts. An RL loader must project only `context`; the sidecar does not authorize a loader to serialize source evidence into the prompt.

Run the independent verifier from the plan worktree:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 docs/campaign/work/lead/r2-expanded-rl-context-v1/verify_expanded_contexts.py
```

The 11,505 rows close the currently reviewed corrected-plus-short pool. They do not close the campaign’s all-eligible-data policy. The broader audit contains 359,053 candidate TRAIN metadata rows not previously attempted across 45 source families. Those rows are not yet clean or admitted. They remain an active full-inventory processing queue without the former 6,690-row ceiling. The 238 previously rejected finish targets remain excluded until a semantic repair proves their supervision; 432 CPT-validation identities remain held out.

No RL schedule, model, rollout, optimizer step, CUDA action, or launch admission is part of this packet.
