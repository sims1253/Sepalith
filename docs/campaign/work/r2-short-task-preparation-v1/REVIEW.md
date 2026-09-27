411 TRAIN-only candidates are ready for root review: 401 literal short function tails and five pipe-edit/post-completion-no-op pairs. This is a candidate pool, not an admitted training mixture. The raw corpus also remains subject to root admission.

The builder checked all 1,055 frozen profile TRAIN source hashes. It excluded all 556 reserved CPT validation groups before source reads. It did not read the 294 profile validation documents or use DEV/final cases. The initial pass rejected 426 CR/CRLF documents and one file over 400 KB. Of the remaining 628 parse-valid files, 254 supplied candidates from 226 packages/groups. A three-finish-per-package ceiling limits concentration. The broader 14,098-document pool was not consumed.

Finish targets are literal source bytes from the final identifier or return statement through the existing closing brace. The prompt retains the entire preceding function slice. Every identifier in the tail is already visible in that prefix; this is a grounding proxy, not proof of unique intended behavior. This differs from the old finish builder's omitted outer-brace slicing. The truncated input function is intentionally incomplete; its complete gold buffer must parse. The saved audit rehashed all 254 contributing source files and checked all 401 source splices.

The five paired no-ops are restricted to the existing canonical pipe rewrite family. Each source has exactly two eligible sites on separate lines. The first simulated edit supplies history; the second supplies the training edit. The post-edit no-op is admitted to this candidate pool only when a full-file AST traversal finds zero remaining sites under the same `_pipe_ok` rule. `extract_pipe() == []` alone is insufficient because that extractor requires two sites. Both edit histories replay with exact hashes and versions. Their PRM03 history record type is `observed_edit_diff`, as required by the existing renderer, but provenance explicitly marks these events as source-derived simulations, not observed user activity. A completed pipe task does not prove that all other editing tasks are complete.

All 411 rows replay through the existing PRM03 renderer, parser and `build_training_row`. They have unique IDs and unique prompts, with zero contradictory prompt/target pairs. Maximum prompt length is 2,379 tokens including BOS; maximum complete output is 45 tokens including the protocol terminal and EOS. The original tokenizer is unchanged, with BOS 0, EOS/PAD 1 and split-special tokenization. No model framework was imported. R checks use the pinned Tree-sitter grammar only; no R code was executed.

No-ops represent only 5/411 rows (1.2165%) and 60/4,982 target tokens including EOS (1.2043%). Do not treat this pool's natural proportions as the intended training objective. Root must choose an explicit mixture and use the reviewed target-only loss path, masking positions before each row's `target_start` while retaining the full target and EOS. Candidate packets contain the canonical row, context, complete before/after buffers, source offsets or replay provenance, and parse results. Extracting their `row` fields is lossless; any sidecar conversion must preserve the full document binding. No training materialization, sampling schedule or loss-policy change is admitted here.

CPU validation: 15 synthetic positive/negative tests passed. A separate saved-packet audit passed 411/411 protocol/token/application checks, 401/401 literal source splices and 5/5 paired source replays. Synthetic controls exist only in test source and were not emitted as training rows. `source-pins.json` records 32 interpreter, first-party, parser, tokenizer and input pins. This is not a complete standard-library or OS closure claim.

Run the tests without changing the frozen artifacts:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -I -S /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-short-task-preparation-v1/test_short_candidates.py
```

The exact build command used was:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -I -S /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-short-task-preparation-v1/build_short_candidates.py
```

The builder refuses to overwrite `candidate-packets.jsonl`. To rebuild, copy its source to a fresh root-owned output directory first. To repeat `audit_saved_candidates.py`, copy the packet to a fresh directory first because that audit writes its two result files beside itself. The initial build took 193.20 seconds; the saved audit took 39.97 seconds. Both limited affinity and tokenizer threads to two cores and performed no model/GPU/network launches.

Remaining limits: this is a conservative, non-exhaustive initial yield. A rejected tail can end processing the remaining functions in that source file. CRLF support, additional detector-proven no-op families, broader-corpus harvesting, duplicate checks against the eventual combined SFT pool, and mixture/admission decisions remain separate work. No target was invented from a teacher assertion or derived from a DEV example.
