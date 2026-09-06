# Learned latent workspace memory: Qwen3.5 experiment specification

2026-09-06. **User-selected primary research direction; proposed, not executed.** The user prefers learned vectors or cached attention state consumed by a specially trained edit model because of their potential value. Superiority over text retrieval is unproven. This document authorizes no training, paid calls, installations or benchmarks. It complements [PROMPT-CONTEXT-RESEARCH.md](PROMPT-CONTEXT-RESEARCH.md); that document's static-analysis priorities and format findings still apply.

**Implementation status (2026-09-06):** [Executable bounded gate](../experiments/latent_memory/README.md), [recipe](../experiments/latent_memory/recipe.json), and [evidence report](../experiments/latent_memory/RESULTS.md) now exist. The report distinguishes CPU plumbing checks from the pending pretrained-decoder and learnability run. The recipe records smaller-encoder, fixed-byte-tokenizer, single-module QA, and no-cache deviations before training. The proposal below remains the broader destination.

## Architecture decision

Start with **module-local latent vectors projected to decoder input embeddings**. Feed those vectors through the edit model normally, then optionally cache the resulting complete hybrid prefix state. This separates reusable, independently encoded modules from order-dependent decoder state.

Do not begin with an encoder that directly predicts every layer's KV and recurrent tensors. Do not substitute generic retrieval embeddings for trained memory vectors. A retrieval embedding learns a similarity geometry; our encoder must learn what the edit decoder can recover and use. ICAE supplies a precedent for trained compact memory slots; gisting supplies a precedent for learned compressed positions and reuse. Neither establishes R editing performance or automatic compatibility with Qwen3.5. [ICAE](https://arxiv.org/abs/2307.06945), [Gisting](https://arxiv.org/abs/2304.08467).

The official Qwen3.5-2B config declares hidden width 2,048, 24 layers with three linear-attention layers followed by one full-attention layer, two KV heads of dimension 256, and partial rotary embeddings. These concrete values apply to that source config, not every Qwen3.5 model. Record the actual trained checkpoint/config hashes before implementation. [Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json).

The maintained Transformers implementation accepts `inputs_embeds`, applies rotary positions to full-attention Q/K, and uses convolution and recurrent state in its Gated DeltaNet path. Thus a normal forward over memory vectors initializes both kinds of layer state using the decoder's own parameters. This is the reason to prefer input vectors for v1. [Qwen3.5 implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Reproducible v1 recipe proposal

Freeze these values before launch; any changed value creates a new recipe ID. This is a specification, not a script claiming to run.

| Component | Proposed value |
|---|---|
| Decoder | One banked Qwen3.5-2B edit checkpoint selected by the main evaluation contract; record exact parent checkpoint, tokenizer, format and export hashes |
| Workspace input | Pre-edit module source, paths, exact signatures, and static facts available at that snapshot; no target replacement or post-edit metadata |
| Candidate selection | Existing static dependency/identifier ranking, frozen and identical across control and latent arms; maximum 4 modules per row |
| Module span | Maximum 1,024 encoder tokens; deterministic syntax boundaries, then deterministic truncation; record omitted ranges |
| Encoder | Separate bidirectional Transformer: 6 layers, width 512, 8 attention heads, FFN width 2,048, dropout 0.1; own 32,768-vocabulary tokenizer trained only on the training partition |
| Pooler | 32 learned queries cross-attend to each module's final encoder sequence; one block, 8 heads, followed by residual FFN and normalization |
| Projection | Shared linear 512→2,048 projection with bias, followed by a learned per-channel scale; initialization must match the decoder input-embedding RMS on training-only calibration data |
| Memory allocation | 32 slots/module × up to 4 modules = up to 128 latent positions; no padding slots inserted into the decoder sequence |
| Text allocation | Same local source and target format in each paired arm; 3,584 local text tokens maximum, 128 context positions for the strict decoder-position comparison; reserve 320 generated tokens |
| Decoder adaptation | Freeze base weights, tokenizer embedding and LM head; LoRA rank 16, alpha 32, dropout 0.05 on eligible 2-D token-mixer linear projections in both layer types; do not attach to convolution or norm tensors |
| Optimizer | AdamW; encoder/pooler/projection LR 1e-4, decoder LoRA LR 2e-5; weight decay 0.01, excluding norm/bias; global gradient clip 1.0; 5% warmup, cosine decay to 10% peak |
| Numeric reference | bf16 model operations with fp32 loss/optimizer state where supported; record actual recurrent-state dtype; start evaluation without a persistent cache |
| Seed | 1273 for the first screen; confirmation uses 1274 and 1275 only if the screen passes |

The encoder tokenizer is independent: its hidden vectors pass through a learned projection, not a token-ID translation. The decoder receives continuous vectors with its own hidden width; no vocabulary resize or new special-token IDs are necessary for these slots. Paths and static facts are encoded within each module. The transport carries a sidecar provenance manifest; it does not expose numeric vectors as text.

Use a new render ID for the memory-prefixed variant, keeping the underlying edit markers and replacement/parser semantics unchanged. The initial sequence is `[module slots in canonical selected order][existing edit prompt embeddings]`. Use ordinary causal decoder attention, with explicit consecutive text positions from zero through memory and prompt, and the text-only position convention supported by the pinned implementation. Do not reset positions at module boundaries in the decoder. Encoder positions restart independently within each module.

Record the concrete LoRA target-name list and trainable parameter count from module inspection before training. “Token-mixer linear projections” must resolve to a checked list in the frozen code version; do not silently broaden matching to all modules. If a kernel bypasses an adapted projection, that is an implementation failure, not a model result.

## Objectives and controls

**Stage A: self-supervised memory learning.** Proposed cap: 5 million scored target tokens, effective batch 8 module-target pairs. Given slots for a training module, reconstruct a sampled contiguous source region of at most 256 tokens or answer an exact symbol/signature/default-value query, with a 50/50 task mixture fixed by seed. The decoder sees the query and slots, not the original source region. Teacher forcing can encourage local copying after the first target tokens, so report first-token and exact-field recovery as well as average loss. Reconstruction is an initialization objective, not the final success criterion.

**Stage B: edit learning.** Proposed cap: 5 million scored target tokens, effective batch 8 edit rows. Train encoder, pooler, projection and decoder LoRA jointly on edit-target CE, masking loss on all prompt and latent positions. Use an 80/20 edit/reconstruction batch mixture. On 20% of edit rows remove memory entirely; on a separate 10% replace the candidate modules with irrelevant training-split modules while keeping the correct edit target. This tests robustness and teaches that memory is not automatically relevant. Freeze these rates before running.

Budget by both scored target tokens and total encoder/decoder tokens/FLOPs. Equal supervised tokens do not make the latent arm compute-matched: it also encodes modules and learns a pooler. Cache encoder results only after its weights freeze; reusing detached, stale vectors during joint training would change the optimization.

| Arm | Purpose |
|---|---|
| Base, unchanged checkpoint | Deployment anchor; no new training |
| Matched-training local-only LoRA | Measures benefit from extra adaptation alone |
| Matched-training exact retrieval LoRA | Main alternative: same candidate pool, raw snippets/signatures within 128 decoder context positions |
| Latent encoder + decoder LoRA | Primary proposed arm |
| Latent ablated at evaluation: absent/shuffled modules | Tests whether the decoder actually uses relevant memory |

All adapted arms share edit rows, target counts, seeds, schedules, local context and generation settings. Give controls equivalent auxiliary target exposure, with their own stated context representation, or report Stage A as extra pretraining rather than pretending it is matched. Report a second comparison at matched total estimated training FLOPs by extending the retrieval control within a preregistered cap. A text-summary control is useful if already available, but no separate summarizer training is required to begin the latent screen.

## Memory budget and cache semantics

Whole-workspace coverage is the destination; the four-module screen is a bounded
test of whether learned memory helps at all. Incrementally encode modules across
the workspace, then select their valid latent blocks for each edit. If that gate
passes, test a learned package/workspace pooler over module latents for global
conventions and cross-module relations, alongside selected detailed module memory.
Compare this hierarchy against module selection alone. Record which source
modules each pooled artifact depends on so additions, deletions and dependency
changes invalidate it. A single fixed-size workspace vector is not assumed to
preserve every identifier, literal or dependency; exact source remains available.

For this config, 128 projected bf16 slots occupy `128 × 2048 × 2 = 524,288` bytes (0.5 MiB). This is only the stored input-vector payload. In the six full-attention layers, an unquantized bf16 KV estimate is `6 × 2(K,V) × 2 heads × 256 × 128 × 2 bytes = 1,572,864` bytes (1.5 MiB), before allocation overhead. GDN recurrent/convolution state, encoder working memory, local-text state and serving snapshots are additional. These are shape-based estimates, not measured memory.

One slot occupies one decoder position, but is not one text token's worth of stored data or necessarily an equal end-to-end cost. Publish slot count, source tokens encoded, compression ratio, bytes, encoder time, decoder prefill/decode and peak memory separately. Compare quality at 32/64/128 slots only after the fixed 128-slot screen; select no slot count on the confirmation set.

**Reusable module artifact:** cache encoder/pooler/projection outputs by source hash, dependency-interface hashes, analyzer version, encoder/tokenizer/projection weights and memory schema. An edit invalidates its module; changed exports/signatures invalidate dependent facts. Exclude the active module from remote memory in v1 so current local text does not conflict with an old self-summary. Keep other modules fresh at their snapshot versions.

**Reusable decoder artifact:** cache the full hybrid state after the entire selected memory prefix, keyed additionally by decoder weights/LoRA, ordered module hashes, positions, numeric/cache format and backend revision. Reuse only for that exact prefix. A changed or reordered module requires replay from a valid earlier snapshot or a new memory-prefix prefill. Independent module states cannot simply be concatenated: later processing depends on earlier state.

The upstream hybrid cache represents recurrent and convolution states separately from ordinary attention KV and restricts simple cropping when recurrent state is present. A cache containing only the full-attention KV omits part of the model's history. Treat rollback support as a pinned-backend property. [Transformers cache implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/cache_utils.py).

Direct cross-model KV is not v1: projection bases, layer counts, head layouts, RoPE conventions and recurrent dynamics belong to particular trained weights. Matching tensor shapes does not match meaning. A separately learned mapping into all these states would be a new architecture and objective. Likewise, relocating same-model KV by adjusting RoPE alone does not recreate the recurrent path or prefix-conditioned hidden states.

## Local runner integration

Start with the pinned Transformers text model as the scientific reference. Then add a runner path that accepts a typed memory artifact plus text prompt, validates the manifest, projects or loads projected slots, and feeds vectors into the model. No public string-completions request should encode vectors as a fake prompt string.

llama.cpp's low-level batch interface permits input embeddings. That is a possible implementation surface, not evidence that Sepalith's pinned Qwen3.5 build supports this entire flow. Validate separate batches of memory embeddings and text tokens, causal positions, full hybrid state restore, and exported LoRA before claiming compatibility. The v1 encoder may use a separate runtime; its export and residency costs must be included. [llama.cpp input API](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h).

Keep cache misses off the keystroke path when possible: encode changed modules on idle/save, select from valid completed artifacts, and fall back to local text while fresh memory is unavailable. Do not serve stale assertions as current. Measure background CPU/bandwidth contention with the edit decoder, even when wall-time accounting amortizes encoding across many requests.

## Leakage protocol and artifacts

Split repositories/packages before tokenizer, compressor or selector training. For each edit, build every module, diagnostic, dependency and memory artifact from the **parent/pre-edit workspace only**. The target is supplied only to the loss/scorer. If a benchmark masks an existing span, remove that span from the encoder view too; a summary of the unmasked file is answer leakage. Omit sibling versions or generated artifacts containing the target when the task would not have had them available.

Each row records workspace commit/tree hash, target file/range, allowed source manifest, module content hashes, split, analyzer outputs, candidate order, tokenizer/recipe IDs and target hash. Each run records package/repository split hashes, code/backend revisions, checkpoint and LoRA hashes, exact training order, parameter list, token/FLOP ledger, failed batches and seeds. Each evaluation row records memory version, realized input positions, prediction, checks, latency regime and cache hit/invalidation status. These artifacts make the experiment reproducible; this note does not claim those manifests already exist.

## Staged falsifiers; no automatic launch

1. **Contract gate, no training:** ordinary token-ID and equivalent embedding-input paths must agree within preregistered numeric tolerance, with identical greedy output on fixed fixtures. Cache restore versus fresh forward must pass for complete hybrid state; changing a module must invalidate the cached prefix. Gate failure blocks the implementation, not the research hypothesis.
2. **Small learnability gate:** on training-only synthetic modules with random identifiers/defaults, memory must improve held-out exact fact recovery by at least 10 percentage points over a no-memory adapted control, with a paired CI above zero. Shuffled memory must materially reduce that gain. This rejects ignored or leaking memory before edit training. Stop if it fails; do not expand capacity without a new recipe.
3. **Edit screen:** use 160 development cases and 160 untouched confirmation cases as in the context research note, enriched for true remote dependencies. Freeze thresholds before Stage B. Continue only if the latent arm improves cross-file validator correctness by at least 5 points over the trained retrieval control on development, local/no-op regressions stay within 2 points, and relevant-versus-shuffled memory has a positive effect. These are screening bars, not significance claims at this sample size.
4. **Confirmation:** evaluate once on the untouched set and then, if promising, preregister a larger package-disjoint sample based on paired disagreement rates. Require paired uncertainty to support the claimed benefit; report small-sample guardrail uncertainty rather than treating a 2-point difference as well measured. Two additional seeds assess training variation before adoption.
5. **Economics/export:** in a quiet serialized window compare cold and warm edit traces, including changed-module encoding and complete prefix rebuilds. Adopt only if measured quality/cost meets a user-approved target; no invented latency threshold is substituted here. Export failure or excess cache cost is a serving limitation, not proof that latent compression has no scientific value.

Research priority is therefore projected latent memory plus decoder training. Direct learned hybrid-state injection remains a later alternative if this representation works but its prefill cost dominates. Readable compression and exact retrieval remain controls throughout.
