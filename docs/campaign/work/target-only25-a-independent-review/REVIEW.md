# SFT-06 target-only25-a independent review

Recommend cutting this trajectory. The completed run passes the startup and
checkpoint integrity checks, but it fails all three admitted continuation
floors. Root retains acceptance, budget accounting, and process release.

| Corrected DEV measure | Observed | Continuation requirement |
| --- | ---: | ---: |
| Exact edits | 24/43 | At least 26/43 |
| Correct strict no-ops | 23/32 | At least 25/32 |
| False suggestions on strict no-ops | 8/32 | At most 5/32 |
| Valid protocol | 71/75 | Reported diagnostic |
| Generation cap hits | 4/75 | Reported diagnostic |
| Exact finish-block edits | 0/6 | Positive finish or restraint evidence |
| Finish buffers passing tree-sitter R parse | 3/6 | Syntax evidence only |

The 75 cases are the exact ordered corrected DEV panel: 43 edits, 32 strict
no-ops, 69 packages, and six finish cases. Every saved raw response matches an
independent decode of its generated token IDs using the pinned local tokenizer.
The existing corrective readout and the frozen runtime protocol were reused.
Its three CPU tests pass, including EOS/cap distinction and exact UTF-16 edit
application. There are no embedded summary mismatches.

All three protocol-valid finish predictions were applied at their exact UTF-16
ranges with no repair; their saved `.after.R` buffers pass tree-sitter R parsing.
The other three hit the 512-token diagnostic cap without canonical EOS and were
not applied or parsed. Thus parse coverage is 3/3 applicable buffers and 3/6
finish cases. None is an exact target match. One passing buffer used 252 generated
tokens, above the delivered primary runtime's 192-token cap. This HF diagnostic
does not establish native/editor acceptance or R execution semantics. No R
interpreter or model weights were loaded by this reviewer.

Before optimizer update 1, the actual post-SFTTrainer gate passed at step 0.
Its first 16 rows match an independent reconstruction of the admitted draw
prefix. Four microbatches supervise 128, 560, 198, and 195 target tokens, totaling
1,081; prompt and padding supervision are zero, and target EOS is supervised.
The gate witnessed one actual fused-loss call: 0.0246937573 versus independent
suffix-logit target CE 0.0241892624. Absolute difference 0.0005044949 is below
the admitted 0.005 tolerance; this is tolerance agreement, not exact equality.
The probe performed no backward or optimizer update. Its completed receipt
precedes the first update timestamp.

Updates 1–25 are contiguous with finite recorded telemetry. The run stopped at
the admitted decision step, with source cursor 400. After the root supervisor
and host guard completed, every declared full-checkpoint file was checked:
12 files, 613,414,157 bytes, all matching size and SHA-256 with no extra files.
All 588 optimizer states are finite and at step 25. The saved scheduler matches
the original 200-step cosine horizon with six warmup steps and LR
0.00004882595527372152. Saved Python, NumPy, CPU, and CUDA RNG structures were
read on CPU without installing their state. These checks establish saved state
presence and consistency; continuation restore/replay equivalence remains
untested. Model adapter bytes were hashed only after terminal completion and
were never loaded. Parent model binding relies on root's admitted preflight.

Source identity is
`dcccf3ed2de98386223a0d06c994747e025363ebed80d6adc9d96bfee21a3e43`;
all 32 snapshot source files match their manifest. Checkpoint manifest SHA-256
is `6078f0c0058ea8de5b0be81c30bafa34fa9b1db6661a7f7df936ee58a56b3315`.
Recipe, tokenizer, renderer, data, source, checkpoint, and parser pins are in
`review.json`, `host-terminal-review.json`, and `parser-identity.json`.

Timing scopes are distinct: optimizer update durations sum to 79.260 seconds;
reported trainer runtime, including DEV evaluation, is 576.036 seconds; host
guard elapsed time is 697.151 seconds; outer supervisor elapsed time is
701.723 seconds. These durations must not be added together. Subtracting the
outer duration from the previous 1,775.577-second unallocated ceiling gives
1,073.854 seconds as arithmetic only; root owns the authoritative charge.
Available time does not override failed behavioral floors.

The 43 saved host samples show at least 8,655 AvailableMBytes, at most
105,702,604,800 committed bytes against a 137,353,273,344-byte limit, and no
recorded driver events. Paging was not uniformly zero: maximum page reads,
pages input, and pages output per second were 238, 757, and 7 respectively.
Sampling does not prove continuous host conditions. The host guard records
`completed` and child exit 0. The outer supervisor's `result` is the attempt
ID, not a success status. Worker process release was not independently checked.

No launch, continuation, checkpoint promotion, state/lease change, network,
GPU, or final-data operation was performed. Final remains sealed until both
2026-09-14T10:00Z and root's weight/harness freeze receipt.
