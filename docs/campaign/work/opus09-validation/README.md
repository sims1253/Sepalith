# RUN-06 Opus09 CPU validation preparation

This packet prepares a bounded, model-free check for the proposed DSpark full-block repair. It reads four targeted files from the pinned public llama.cpp tree, checks their SHA256 values and structural anchors, and runs a small Python reference model for output capacity, token positions, suffix cleanup, and EOS. It does not import llama.cpp or ggml, allocate a graph, load a model, start a server, use CUDA, or modify the pinned source.

The executable is [`validate_opus09.py`](validate_opus09.py). The recorded run is [`validation-result.json`](validation-result.json). Run it from the PLAN worktree with:

```text
python3 docs/campaign/work/opus09-validation/validate_opus09.py \
  --source-root /home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453 \
  --candidate /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-hillclimb/opus09-dspark-repair-run/response.json \
  --report docs/campaign/work/opus09-validation/validation-result.json
```

The run passed all 12 reference cases and all source checks at commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. The Opus09 response file existed but was zero bytes, so it remains a pending external input and no patch was applied or reviewed.

## Source map

The current draft path is in `common/speculative.cpp`:

- `:960` reads `dflash.block_size`; `:972` derives the trained maximum (`block_size` for DSpark, `block_size - 1` for legacy DFlash).
- `:1178` derives the batch row count from `n_max` for the pinned path. `:1182` places each row at `n + i` and marks every row `logits=true`.
- `:1213` reads DSpark confidence rows, and `:1222` samples each DSpark row. A nonzero confidence threshold can stop at the first low-confidence row.
- `:1416` checks the draft memory position after prefill. `:1590` removes the draft suffix from `dparams[seq_id].n_past`; the pinned call ignores its boolean return, so a future full-block repair must add the return and `seq_pos_max == boundary - 1` checks before claiming cleanup safety.

The context output limits are enforced in `src/llama-context.cpp`:

- `:1684` rejects a backend-sampled sequence after `n_outputs_max_per_seq` output rows.
- `:1782` requires `output_reserve(n_outputs_all)` to cover the logical batch.
- `:2326-2328` reserves sampling graph capacity from `n_outputs_max` and `n_outputs_max_per_seq`.
- `:3922` forwards the `llama_memory_seq_rm` boolean. The public declarations for `decode` and `output_reserve` are in `src/llama-context.h:144,226`.

There is no `src/models/*dspark*` file in this pin. DSpark is the Markov/confidence mode in `src/models/dflash.cpp`: `:240-249` reads and bounds `dflash.block_size`, `:291` feeds greedy `ggml_argmax` IDs into the next Markov position, and `:361`/`:444`/`:449` apply RoPE to position inputs. The graph requires equal-sized blocks (`:246`) and returns without constructing the head for a block larger than the trained size (`:249`).

## Executable reference invariants

The rejected Opus06-style shape is represented explicitly: a seven-row DSpark full block with `n_outputs_max_per_seq=4` and `n_outputs_max=4` fails before model allocation. A safe one-sequence seven-row shape requires at least:

```text
n_batch >= 7
n_outputs_max_per_seq >= 7
n_outputs_max >= 7
```

For `n_parallel > 1`, the total limits and combined batch scale to `n_parallel * block_size`; `n_ubatch` may split the batch but must remain positive and no larger than `n_batch`. The target verification context is separate: it needs `1 + proposal_k` rows per sequence. Changing `proposal_k` must not silently shrink the draft graph's full `block_size` decode.

The KV reference state machine appends positions at `n + i`, tests every accepted prefix from zero through the full block, repeats reject/accept cycles, requires a successful suffix removal, and detects a stale row even when a fake memory implementation reports success. It also tests EOS at every block position and rejects rows after EOS. These are reference invariants only; they do not prove a particular DFlash/ISWA or DSV4 memory implementation accepts the operation.

## Current gate

The source and reference checks are ready for a later candidate. A native validation remains pending until the external response is materialized, an isolated candidate copy can be checked for draft/target capacity, positions, cleanup return/postcondition, and EOS behavior, and root performs the model-free build or root-owned runtime gate. Draftless remains the safe default. The prior Opus06 review is [`review.md`](../serving-hillclimb/opus06-review/review.md); its full-block candidate was static-rejected because constructor capacity stayed at `n_max + 1` while the proposed decode emitted seven output rows.
