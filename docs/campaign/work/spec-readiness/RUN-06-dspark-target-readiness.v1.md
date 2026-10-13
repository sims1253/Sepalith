# RUN-06 DSpark target readiness

**Observed:** 2026-09-12, Europe/Berlin  
**Status:** preparation complete; current Midtrain/SFT/RL DSpark integration is **NO-GO**  
**Owner:** worker-runtime  
**Scope:** CPU/read-only compatibility inspection. No model forward, weight download, cache generation, training, server launch, GPU, SSH, package change, or state edit was performed.

## Result

The released `MiniCPM5-2B-DSpark` is not admitted against the campaign's
Midtrain-derived target. The released draft was trained for the released
`MiniCPM5-2B` target, while the campaign target is the independently pinned
`openbmb/MiniCPM5-2B-Midtrain` revision and will later have SFT/RL changes.
The two targets have encouraging published/local shape similarities, but an
exact target/draft pair, tokenizer identity, hidden-state cache, and converter
route are not present. Published DSpark acceptance therefore cannot be used as
evidence for this target.

The pinned b10453 runtime does contain a `draft-dspark` implementation. That
means a future exact pair could be screened without changing target weights,
but runtime support is not proof that the released draft can consume this
target. The current converter's `--dspark` path accepts only
`DeepseekV4ForCausalLM`; its Qwen conversion module contains a
`Qwen3DSparkModel` class but the top-level dispatch does not make that a
general MiniCPM/Llama conversion path.

`ngram-mod` remains the only immediately available model-free comparator. The
lead's current TRAIN mechanical-parity result has no speed gain, so it is not
promoted by this packet.

## Pinned target and tokenizer

The inspected target is `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain`
at revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`.

| Item | Observed identity |
| --- | --- |
| HF weight file | `model.safetensors`, 5,033,557,128 bytes, SHA256 `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad` |
| Target config | SHA256 `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`; `LlamaForCausalLM`/`LlamaConfig` |
| Geometry | 42 layers; hidden 2048; FFN 6144; attention heads 16; KV heads 2; head dimension 128; vocabulary 130560 |
| Special IDs | BOS 0; raw config EOG list `[1, 130073]`; campaign canonical terminal EOS 1; pad 1 |
| Target tokenizer | `tokenizer.json` SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| Tokenizer config | SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`; add BOS/EOS false |
| Tensor header | 381 tensors; no observed `mtp`, `nextn`, `draft`, `dspark`, or `speculator` keys |
| Existing BF16 GGUF | `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-bf16.gguf`, SHA256 `047060045f8e5e896ca44e8933775b9f41686375cefccc216171711534d7c715`; 381 tensors |

The raw config's second EOG ID must remain an explicit termination decision in
any future serving smoke. The pinned PRM-07 contract requires manual BOS 0,
`add_special_tokens=false`, no chat template, and the stored terminal ID 1.

## Compatibility matrix

The status in this table is a readiness decision for this campaign, not a
quality claim.

| Path | Evidence and concrete integration work | Decision |
| --- | --- | --- |
| **Released MiniCPM5-2B-DSpark** | The official card describes a draft for released `MiniCPM5-2B`: five draft layers, 323,776,001 parameters, seven tokens per pass, and target taps `[1, 10, 20, 30, 39]`. Those taps are in the 42-layer campaign target and the target hidden/vocabulary sizes are shape-compatible on paper. The released target files and draft files are absent from the bounded local artifacts, so their exact hashes, tokenizer map, target revision, metadata, and conversion output are unverified. | **NO-GO for current target.** Conditional artifact-only readiness is possible after an exact released target/draft pair is supplied and independently hashed. The card's reported acceptance is not transferable to Midtrain, SFT, or RL weights. |
| **Adapted DSpark** | Official DeepSpec trainer code exposes Qwen3 and Gemma4 DSpark implementations, not a MiniCPM/Llama implementation. A MiniCPM port needs target-cache generation, a Llama/MiniCPM DSpark config/model/trainer path, exact tap extraction, loss-mask/cache manifests, draft export, and b10453 converter metadata. The shared 2048 hidden/vocabulary geometry reduces shape risk, but does not establish attention, rotary, layer-type, mask-token, or state-dict compatibility. | **NO-GO pending a scoped port.** Candidate only if the port and a TRAIN-only smoke fit the reassigned two-GPU-hour ceiling; no port or cache is authorized by this packet. |
| **Lighter AR draft** | Existing local MiniCPM1B layer-drop directories are ordinary `LlamaForCausalLM` candidates with the same vocabulary/tokenizer family but hidden size 1536, FFN 4608, and 12/16/20 layers. `draft-simple` does not consume target hidden taps, so different hidden size can be operationally possible after conversion. Pair quality, target-specific training, and end-to-end speed are unknown; no converted candidate was admitted here. | **CONDITIONAL LOW-COST SCREEN, not a winner.** Require exact tokenizer/output-vocabulary and GGUF load checks, then the same exact-output and latency screen. Do not infer DSpark acceptance or quality from it. |
| **MTP** | The Midtrain tensor header has no MTP/NextN fields. The pinned converter only exports MTP when the architecture advertises MTP support, and an attached untrained head is not a target-trained draft. Existing unrelated Qwen MTP artifacts cannot be paired with this target. | **NO-GO.** Reserve only for a target-specific trained artifact. |
| **EAGLE-3** | b10453 supports `draft-eagle3`; its loader requires exactly three target extract layers and target hidden-size metadata (`src/models/eagle3.cpp:3-22`). No MiniCPM artifact or recipe was found. The generic `--target-model-dir` converter route is necessary but does not supply a MiniCPM EAGLE checkpoint. | **RESERVE / NO-GO now.** Requires a target-specific checkpoint, tokenizer, three taps, converter metadata, and exact parity. |
| **DFlash** | b10453 supports block-diffusion drafts and reads `target_layer_ids`, target hidden size, `dflash.block_size`, mask token, and extracted target layer inputs (`common/speculative.cpp:907-1014`). Official DeepSpec examples are Qwen3/Gemma4; no MiniCPM recipe/artifact was found. | **RESERVE / NO-GO now.** Requires target-cache training and a target-specific export. |
| **ngram-mod** | Model-free b10453 mode. The pinned flags are available and the existing lead result has exact mechanical parity but no speed gain. | **Comparator only.** No promotion from this packet. |

## Exact source findings

The pinned runtime is commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` at
`/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453`.

Relevant source identities:

| Source | SHA256 | Read points |
| --- | --- | --- |
| `common/speculative.cpp` | `81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3` | DSpark/DFlash state and dimensions 907-1014; DSpark block and confidence handling 1210-1224; auto-detection by `markov_w1.weight` 2258-2265; runtime dispatch 2471-2503 |
| `common/arg.cpp` | `566a8122afc02bbfcca572b7ac8d7d7d985b2b087b780434be75f588184f458d` | `--spec-draft-n-max`, `-md`, `--spec-type`, and ngram-mod arguments 4076-4192 |
| `docs/speculative.md` | `0983aa1e725075228cc82eafb068856d72519e76aab9c84cd7a244f8bb20a7d7` | DSpark conversion/serve example 81-101; current documentation says only Qwen3 draft backbones are supported |
| `convert_hf_to_gguf.py` | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` | `--dspark` and `--target-model-dir` declarations 120-165; DeepSeek-V4-only dispatch 261-270 |
| `conversion/qwen.py` | `65c61155458078232dd3f9d23284710fa39c29cf7ecc341194c825cf5334f43f` | Qwen DFlash/DSpark classes 632-718; target tokenizer and target-layer metadata handling |
| `llama-server --help` capture | `7d98b7efe9a31df6213913e35109fa89e0d23e0d45a690bbb4d03c63a55b53e1` | `docs/campaign/work/spec-readiness/llama-server-help.txt`; exit 0; confirms `draft-dspark`, `-md`, and ngram modes |

The source file itself was read at the points above. This memo does not modify
the converter.

The smallest converter repair, if the released/adapted draft is later
authorized, is to add an explicit `Qwen3DSparkModel`/MiniCPM-Llama dispatch
under `--dspark`, route it through the target-model metadata path, emit
`target_layer_ids`, `dflash.block_size`, mask-token, target hidden-size, and
tokenizer metadata, and add a conversion fixture that fails on any target or
tokenizer mismatch. This is a proposed repair, not an executed change. A
model-class port is still required on the DeepSpec training side.

## Smallest future TRAIN-only smoke

This is a preparation gate. It does not authorize downloading a released
draft, creating a cache, or launching a server.

1. Freeze one exact target parent and record target GGUF, target config,
   weights, tokenizer JSON, tokenizer config, converter, server binary, and
   source commit hashes. Require tokenizer JSON SHA256
   `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`,
   vocabulary 130560, BOS 0, canonical EOS 1, and the PRM-07 tokenization
   policy. Reject any released DSpark whose target revision or tokenizer map
   is not independently proven.
2. Select exactly 12 admitted TRAIN rows: six in the short prompt band and
   six in the long prompt band, using fixed row IDs and stored token IDs.
   Do not use DEV, S1, or any held-out evaluation trace. For an adapted draft,
   generate target responses and the target hidden-state cache at the exact
   draft taps from those TRAIN rows only; bind the cache manifest and target
   parent hash.
3. Convert the draft in BF16 with `--target-model-dir` only after the
   converter supports the draft architecture. Inspect GGUF architecture,
   target-layer metadata, block size, mask token, hidden size, vocabulary,
   tokenizer metadata, and Markov tensors. Do not quantize before BF16
   structural checks pass.
4. Run one baseline and one candidate repetition for each of the 12 rows at
   temperature zero, with the unchanged manual BOS/EOS protocol and a 64-token
   completion cap. For DSpark, the candidate command shape is:

   ```text
   llama-server -m TARGET.gguf -md DSPARK.gguf \
     --spec-type draft-dspark --spec-draft-n-max 7 \
     -t 2 -tb 2 -b 256 -ub 256 -np 1 -c 10240 -ngl 0 -fa on --jinja
   ```

   Record target/draft output token IDs, first divergence position, terminal
   reason, drafted/accepted token counts, prompt length, TTFT, wall time,
   parser/load errors, and the exact runtime/source hashes. Use
   `--spec-type draft-simple --spec-draft-n-max 2` for the lighter AR screen.
5. The 12-case screen passes mechanics only when all 12 candidate outputs
   exactly equal their paired baseline outputs, the prompt IDs and terminal
   handling match, no target tap/cache/parser/load error occurs, and all
   metadata checks pass. Report latency as a screen; make no quality claim
   from it. Promotion still requires the registered 20 short + 20 long traces,
   two repetitions, exact parity, at least 1.4x end-to-end speed, bootstrap
   lower bound above 1.0, and no p95 regression from `SPECULATIVE-PATHS.md`.

### Stop rules

Stop the path before training or serving if any exact target/draft/tokenizer
identity is missing, a tap is outside the target layer range, the hidden or
vocabulary dimensions disagree, the converter cannot emit the required
metadata, the 12-case screen has one output mismatch, or the registered
latency gate is not credible. Do not expand beyond the existing conditional
two-GPU-hour reassignment from optional consolidation, and do not refresh the
draft after a target change without regenerating a TRAIN-only cache.

## Artifact identities to bind later

The following are the known identities for a future receipt:

* target revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`;
* target weights SHA256 `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad`;
* target config SHA256 `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`;
* target tokenizer SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`;
* target tokenizer config SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`;
* converter SHA256 `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f`;
* server binary SHA256 `123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804`;
* b10453 source commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`.

No released DSpark target or draft hash is currently bound. A future exact
pair receipt must add both file hashes, the model-card/repository revision,
DSpark config hash, target-layer list, block size, mask token, cache manifest
hash, and output parity records.

## Primary sources

* [DeepSpec repository and workflow](https://github.com/deepseek-ai/DeepSpec)
* [Official DSpark trainer](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/trainer/dspark_trainer.py)
* [Official Qwen3 DSpark model](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/deepspec/modeling/dspark/qwen3/modeling.py)
* [Official Qwen3 DSpark configuration](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/config/dspark/dspark_qwen3_4b.py)
* [Released MiniCPM5-2B-DSpark card](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)

These sources establish the public model/trainer contracts only. They do not
establish compatibility or acceptance for the campaign's Midtrain/SFT/RL
weights.
