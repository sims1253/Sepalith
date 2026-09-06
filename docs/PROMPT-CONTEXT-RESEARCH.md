# Prompt context, static analysis, and learned workspace compression

Research date: 2026-09-06. Scope: the current Sepalith editor path and a proposed learned workspace-compression model. No training, serving benchmarks, model API calls, installations, or workspace-wide indexing were performed. Recommendations below are hypotheses, not measured Sepalith improvements. “Summary-network output” means a learned compressor in this note; it does not identify a particular paper.

## Recommendation

User-selected direction: **learned latent workspace memory consumed by a specially trained edit model**. Test module-local compressed vectors projected into Qwen3.5's input embedding space, with encoder training and decoder LoRA. This selection reflects the user's assessment of potential value; it is not evidence of superiority. The concrete staged specification is [LATENT-WORKSPACE-MEMORY.md](LATENT-WORKSPACE-MEMORY.md).

Keep a versioned local symbol/dependency index for module boundaries, evidence selection and invalidation. Readable summaries and exact retrieval remain controls, not prerequisites that latent memory must wait for. Cache encoder outputs per module; optionally cache the consuming decoder's complete hybrid prefix state after processing an unchanged selected memory prefix. Do not inject unrelated-model KV tensors. A readable summary, a learned input vector and a decoder state are different interfaces.

Use cyclomatic complexity first as a context-allocation and evaluation feature. Prioritize current diagnostics, definitions, signatures, dependency facts, and relevant tests as prompt content. A complexity score alone does not say what edit is intended or which branch is wrong.

## 1. What the repository actually does

| Evidence | Current behavior | Consequence |
|---|---|---|
| [`docs/prompt-format.md`](prompt-format.md), “The prompt,” “Truncation anchor,” and “Why this order” | Describes PSM with file prefix, history, cursor, suffix and end markers; proposes a persistent truncation anchor. | Treat this as design/history, not the implemented wire format. |
| [`context_build.ts:20–39`](../extensions/vscode-sepalith/src/context_build.ts) | Emits Zeta-2: suffix first, then filename and prefix, optional outline, CURRENT region, separator, FIM middle. | The existing checkpoint/renderer/parser combination is the baseline. Do not silently replace it with PSM during a context experiment. |
| [`context_build.ts:296–351`](../extensions/vscode-sepalith/src/context_build.ts) | Budgets prefix/suffix by approximately 6,000 characters, recalculates truncation per request, caps the pin at 4,000 characters elsewhere, and adds the outline separately. In the no-pin path, the suffix itself is not bounded here. | These are not a hard model-token budget or an implemented hysteresis anchor. Count actual tokenizer tokens, including metadata and generation reserve, in a new experiment. |
| [`extension.ts:497–508`](../extensions/vscode-sepalith/src/extension.ts) | Uses document symbols when available and a brace-scan fallback otherwise. | The document's “no provider means plain prompt” description is stale. Measure real fallback behavior. |
| [`context_builder.py:8–37`](../experiments/synthetic-data/context_builder.py) versus [`context_build.ts:127–146`](../extensions/vscode-sepalith/src/context_build.ts) and `formatOutline` | Python pins the top-level enclosing function and renders signatures with arguments. TypeScript chooses the innermost function/method symbol and renders line plus symbol name; the scan fallback supplies names. | Training/inference parity is not established merely because both implementations call their output an outline or pin. Nested functions and missing arguments are meaningful differences. |

Separate three contracts: **model compatibility** (architecture, tokenizer, special-token IDs, positions, export), **format compatibility** (section order, markers, target replacement semantics, stops/parser), and **context selection** (which facts fit). A new selector can be evaluated while the first two remain fixed. New section syntax needs a render identifier and matched training examples; a marker-shaped string is not automatically a special token.

The conditioning-ablation evaluator confirms index-matched plain/types/dropout rows and whole-region scoring in [`eval_ablation.py:1–16`](../experiments/eval/eval_ablation.py). The prompt document reports no benefit from type conditioning at the then-current scale. This bounded review did not recover its complete historical numerical readout; do not generalize that reported result to all static metadata or today's weights.

[H1](../experiments/harness_search/H1_RESULTS.md), sections 3–7, found no held-out improvement over the default for its frozen v7 checkpoint. Some outline configurations increased no-op proposals by 4–9 percentage points during search. Crucially, scenario prefixes had median about 270 characters and maximum 813: prefix caps were inert and pin/truncation nearly inert. H1 supplies a useful no-op warning and a held-out discipline, not a verdict against workspace retrieval or compression on real long-file tasks.

## 2. Static signals beyond the syntax tree

The following ordering is a proposed starting point. “Static” does not mean equally reliable: store source, analyzer version, document version, and an uncertainty field with every fact.

| Priority | Signal | Prompt representation | Main risk |
|---|---|---|---|
| 1 | Diagnostics on or near the edit, including relevant `ry` results | Rule ID, range, short message, implicated names; distinguish pre-existing from newly introduced diagnostics | Stale diagnostics, transient incomplete code, style warnings mistaken for a request to refactor |
| 1 | Resolved definition, exact signature/defaults, imported/exported names | Source-linked snippets for referenced symbols | Wrong resolution under dynamic dispatch or masking |
| 1 | Small backward dependency slice: definitions feeding the edited expression | Assignment/definition excerpt and dependency relation | Claiming a complete dataflow analysis where only lexical evidence exists |
| 2 | One-hop callers/callees and relevant test/examples | A few edges and exact usage excerpts | Global centrality retrieves popular utilities rather than the needed implementation |
| 2 | Data/schema facts with provenance | Column names, object class, named arguments, documented invariants | Runtime-specific facts incorrectly treated as universal |
| 2 | Recent local changes and package configuration | Small relevant diff; namespace and dependency version facts | History from after the target edit leaks its answer |
| 3 | Complexity, nesting, function size, fan-in/out | Prefer internal ranking/stratification; expose only for an explicit refactoring task | Correlation with size or domain mistaken for a diagnosis |

R's `codetools::findGlobals` identifies global functions/variables but explicitly calls its result approximate; `assign` and `rm` weaken its assumptions. `checkUsage` offers undefined-name and local/parameter-usage checks, but accepts closures, and its package helper requires loading the package. Do not source arbitrary workspace code simply to build editor metadata. Prefer available language-server/`ry` facts and syntax analysis, and label unresolved dynamic edges. [R findGlobals](https://search.r-project.org/CRAN/refmans/codetools/html/findGlobals.html), [R checkUsage](https://search.r-project.org/CRAN/refmans/codetools/html/checkUsage.html).

Cyclomatic complexity counts control-flow structure, not semantic intent. The R `lintr` implementation exposes a configurable complexity lint with a default limit of 15; that default is not evidence that 15 is Sepalith's useful threshold. Start by storing the analyzer's per-function value alongside size and nesting. Test whether it improves routing or predicts failures *after controlling for function length*. For prompt injection, compare the scalar against the actual relevant branch conditions at the same token budget. No source reviewed establishes a completion-quality gain from adding a complexity scalar. [lintr complexity documentation](https://lintr.r-lib.org/reference/cyclocomp_linter.html).

Repository graphs are useful selectors, not guaranteed semantic truth. Aider ranks a file dependency graph and selects symbols to fit a token budget; this is a practical baseline. RepoCoder supplies research evidence for retrieval in repository-level completion, using iterative retrieval and generation. Its result does not establish R performance or justify multiple sequential model calls in a keystroke path. Start with one bounded retrieval pass. [Aider repo map](https://aider.chat/docs/repomap.html), [RepoCoder](https://arxiv.org/abs/2303.12570).

## 3. What a learned workspace compressor could emit

| Representation | Training and evidence | Fit to Sepalith |
|---|---|---|
| Learned extraction of exact source spans | A selector learns which evidence helps the downstream task. LLMLingua-2 learns token retention from distilled data. | Text transport works; prefer whole signatures/statements rather than token deletion inside R code. Removing a negation, delimiter, or namespace qualifier changes meaning. |
| Learned discrete, readable summaries | RECOMP trains extractive and abstractive compressors for end-task utility and supports emitting nothing when context is unhelpful. | Text-based control for the user-selected latent experiment; retokenize with the edit model's tokenizer. |
| Soft memory slots | ICAE trains an encoder using reconstruction and language-model objectives, then instruction training; slots condition an associated decoder. | Requires an encoder/decoder compatibility contract and adaptation to repository editing. Retrieval embeddings are not a substitute for these learned slots. |
| Gist/KV representation | Gisting trains with attention masks so a small set of positions carries prompt information and can be reused. | Needs the corresponding trained model and state handling; it is not a generic way to turn arbitrary summary text into portable KV. |

These are mechanisms with published evidence, not drop-in R workspace systems. RECOMP evaluates language modeling and QA; LLMLingua-2 includes natural-language reasoning and long-context datasets. Their results support testing learned text compression, not assuming exact code facts survive. [RECOMP](https://arxiv.org/abs/2310.04408), [LLMLingua-2](https://arxiv.org/abs/2403.12968).

ICAE reports 4× compression in its Llama experiments. Its released v2 uses Mistral-7B and supports concatenating multiple compressed spans; this is a useful precedent for chunk-local memory, not proof of arbitrary whole-workspace fidelity. Gisting reports large prompt compression but only a modest wall-time gain in its measured setting. Compression ratio alone is not an editor latency result. [ICAE paper](https://arxiv.org/abs/2307.06945), [ICAE implementation](https://github.com/getao/icae), [Gist paper](https://arxiv.org/abs/2304.08467).

**Learned hierarchy proposal:** encode each function or module from its source and exact static facts. Cache a compact summary per module; optionally learn a package-level summary from those modules. At query time, a learned selector reads the cursor context and chooses module summaries plus exact source evidence. A small query-specific compressor can combine only that shortlist. Compare a learned file/module hierarchy against deterministic signatures and direct retrieval. Do not recursively summarize the entire repo at every keystroke, and do not treat a single fixed-size root summary as a lossless repository representation.

Train for edit utility, not attractive prose. Proposed targets include referenced-symbol recovery, exact signature/default recovery, choosing a correct relevant dependency, and edit-target likelihood conditioned on the summary. Add empty-context examples and irrelevant-candidate negatives. Teacher-generated summaries, if later authorized, need source checks; the teacher must see only the pre-edit workspace. Train the editor on the actual compressor's outputs, including missing or uncertain fields, rather than perfect hand-authored summaries alone. Keep exact identifiers, paths, signatures, and literals outside lossy abstraction where possible.

## 4. Serving, locality, and cache costs

The current extension sends a string prompt to `/v1/completions` (`extension.ts:109–110`). Discrete summaries fit that transport. The summary model can use a different tokenizer because its output is text; the decoder's tokenizer determines the prompt budget. A supported architecture/export is still required if the summary model itself is to run as GGUF.

Latent input is not categorically impossible in llama.cpp: upstream `llama_batch` accepts embedding vectors when token IDs are absent. That low-level entry point is not a ready-made ICAE or gist integration. A design must specify vector dimension, model/adapter version, positions, attention behavior, batching, memory lifetime and quantization validation. Gist KV reuse additionally needs the correct state semantics. Current upstream interfaces also do not establish support in Sepalith's pinned server build. [llama.cpp C API](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h).

Cache module summaries by content hash, analyzer/schema version, compressor weights/tokenizer and dependency-interface hash. An implementation-only change should update its local summary; a signature/export change should invalidate affected dependents. Workspace changes are incremental; compute summaries on save or idle, outside the request path. During rapid typing, use fresh local source plus unchanged, version-valid summaries from other modules. Omit stale facts that refer to changed definitions instead of presenting them as current.

Keep summary generation local in the proposed product. Bound its CPU/RAM use and cancel obsolete work. A whole-workspace model pass has a cold-start cost even if the result is tiny; a second resident model also consumes memory and bandwidth. Account for cold indexing, incremental updates, compressor latency, decoder prefill/decode, peak RSS/VRAM and invalidation frequency.

For a text-prefix cache, shared earlier tokens are what matter. Stable text later in a prompt is not independently reusable merely because that section did not change. The existing suffix-first format means an early suffix change can invalidate later context. In a future format-trained model, compare stable shared metadata first against a local-prefix-first arrangement on real traces; cursor-specific retrieval churn can erase the benefit of moving metadata first. Upstream server documentation describes common-prefix reuse, but the deployed build and flags need their own measurement. [llama.cpp server cache semantics](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

A useful accounting identity is `amortized cost = changed-module compression / requests-between-updates + retrieval + uncached prefill + decode`. Compare this with direct retrieval under the same edit workload. Do not count compressor work as free because it occurs in the background.

## 5. Controlled ablation before a rebuild commitment

The primary latent-memory experiment is specified in
[LATENT-WORKSPACE-MEMORY.md](LATENT-WORKSPACE-MEMORY.md), including trained
retrieval controls, latent ablations and encoder cost accounting. The text-only
design below is an optional separate context study; readable-summary arm D is
not a prerequisite or replacement for the user-selected latent experiment.
Its 512-token context allowance differs from the latent screen's 128 positions;
do not combine their results as if they were one matched-budget comparison.

First freeze the selected checkpoint, tokenizer, renderer, stop/parser rules, generation budget and cache policy. Save pre-edit workspace snapshots. Choose package/repository-disjoint development and confirmation sets with long-file and cross-file tasks; H1's tiny constructed prefixes cannot test this question. Include no-op cases, nested functions, dynamic/NSE code, stale-summary cases, API changes, and tasks where another file is provably needed. Exclude the answer span and future edits from every indexed artifact.

**Small first experiment:** 160 development cases and 160 untouched confirmation cases, balanced across four strata: local-only, long-scope, cross-file, and no-op/irrelevant-context. This is an exploratory screen, not a precise estimate of a one-percentage-point improvement. Derive the final power requirement from paired disagreements before a production claim.

Use a 4,096-token total input ceiling with the actual decoder tokenizer, holding the current-file allocation fixed at 3,584 tokens and reserving 512 for context treatments (including their delimiters). Use naturally long eligible rows so equal-budget pairs can actually fill their allocations; do not use padding as pretend evidence. Short/unfillable rows are a separately reported stratum with realized token counts. Keep the same raw current-file content across treatment arms.

| Arm | 512-token context treatment | Question |
|---|---|---|
| A | Additional nearest local source | Does remote context beat spending the budget locally? |
| B | Deterministic signatures plus ranked exact remote excerpts | Is retrieval enough? |
| C | B with live diagnostics/local dependency facts replacing an equal token allowance | Do semantic facts help beyond syntax? |
| D | Learned readable summaries from the same candidate pool, preserving exact identifiers/evidence excerpts within the allowance | Does learned compression improve information per token? |

Freeze the candidate pool and ordering before comparing B versus D so retrieval differences do not masquerade as compression gains. Also run a secondary quality-per-realized-token curve at 128/256/512 context tokens for the finalists. A no-extra-context baseline can be reported separately, but is not the equal-token causal contrast.

Test complexity cheaply after selecting a baseline: same metadata allowance with complexity/size/nesting versus size/nesting alone, then a routing-only variant that changes allocation without showing the scalar. Freeze thresholds on development data. Report results within function-length bands so a size effect is not attributed to complexity.

Score paired validator correctness, exact preservation of unrelated code, diagnostic repair without new diagnostics, no-op false proposals, referenced-symbol accuracy, and available executable checks. Audit retrieved evidence recall and summary factual precision separately to distinguish selector failures from decoder failures. Report paired confidence intervals/McNemar disagreements and per-stratum results. Keep the confirmation set closed until the selector, compressor, schema and thresholds are fixed.

Latency needs cold and warm editing traces in a quiet, serialized window, with identical cache policy and actual summary invalidations. Report full p50/p95 request latency and background resource cost; compare measured bytes and FLOPs for soft slots rather than calling one vector “one text token.” No compute is authorized by this research note.

A matched-training comparison should adapt the editor to B and D at the same training-row, token and optimizer budgets. A frozen model's failure to read an unfamiliar summary schema rejects immediate deployment, not learned compression as a family. The user-selected latent arm is specified separately with encoder training, decoder adaptation and runner integration fully charged; these text arms are controls, not a required first deployment.

## 6. Risks and decision boundary

- **Leakage:** generate indexes, summaries and diagnostics from the pre-edit snapshot only. Split before training the compressor or retrieval scorer; package versions and copied modules can cross a nominal file split. Include evidence hashes in eval records.
- **Lossy facts:** summaries may omit defaults, dispatch constraints, side effects or exceptional branches. Keep exact evidence accessible and measure omissions; a confident prose summary is not a correctness certificate.
- **Train/inference mismatch:** analyzer versions, nested-scope rules, missing LSP, summary schema, raw-code ratio and format must appear in training or explicit robustness tests.
- **No-op regression:** more context can prompt unnecessary edits. H1 makes this a concrete local risk, not a hypothetical one.
- **Latent portability:** changing decoder weights or position/attention conventions can invalidate compressed memories. Version and regenerate them; do not assume cross-model interchangeability.

Decision now: preserve the current serving baseline while pursuing the user-selected latent-memory research specification. Start with projected module-memory vectors and a trained decoder, then evaluate caching of its complete hybrid state. Text summaries and retrieval provide measured alternatives. Neither the old conditioning result nor H1 closes this question. No training or serving experiment is launched by either document.
