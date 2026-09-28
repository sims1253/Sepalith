# Selected R2 F16/Q8 qualification

This packet prepares a small, reproducible TRAIN-only comparison for the
selected `SFT11-task-global-b-500` target. It does not load model payloads and
does not make a promotion decision. The selected identities are:

| arm | path | bytes | SHA-256 |
| --- | --- | ---: | --- |
| F16 | `/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf` | 5,039,006,496 | `fe1c38a2b53519fb15eeb4ac36efdd7b6f60ad5a58451475b51308a93cb8a8ee` |
| Q8 | `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf` | 2,679,710,496 | `d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db` |

The panel contains eight byte-pinned TRAIN rows: four strict no-op rows and
one edit row from each of `format_propagation`, `na_rm_propagation`,
`pipe_rewrite`, and `rename_propagation`. The source is
`finish-corrected-train-v1/train-token-rows.jsonl`, 11,526 rows, SHA-256
`e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe`.
Rows are limited to prompt length 2,048 and target plus protocol EOS at most
192 tokens. `quant_pair_probe.py` sends the same native request to both arms:
manual BOS 0, context 4,096, cap 192, temperature 0, seed 0, and canonical
EOS 1. Native EOG 130073 is recognized and rejected as a noncanonical terminal;
the client never appends a source target tail.

Run the root-owned command only after the outer watchdog and server cleanup
wrapper are installed:

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
RUN=/home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-final-quant-pair-v1
RUN="$RUN" bash "$PLAN/docs/campaign/work/lead/r2-final-quant-qualification-v1/run-quant-pair-command.txt"
```

The command runs F16 and Q8 sequentially with identical server arguments. The
scorer uses one cold record per row for edit/no-op/parser counters and retains
warm records for latency. All eight rows remain in every denominator. It
reports exact raw-text and token-ID parity as an observation; quantization is
allowed to change tokens. Its top-level status is
`measured_requires_root_review` when all request keys are present, rather than
an automatic quality pass.

`r-parse-applied-buffer.py` applies each accepted edit body to the stored FIM
prefix/current/suffix context and runs `/usr/bin/Rscript --vanilla` with base
`parse(file=..., keep.source=FALSE)`. This is syntax-only: it never evaluates
or sources R code. The TRAIN artifact contains rendered FIM context, not the
complete original source files, so the result is explicitly a context
reconstruction check and cannot claim universal full-file buffer validity.

The prior `RUN-09-primary500-f16-dev-a` / `RUN-09-primary500-q8-dev-a` pair is
historical evidence for a different `SFT-primary-step500-runtime` identity,
with a 512-token cap and Vulkan serving. It is excluded from this selected R2
qualification.
