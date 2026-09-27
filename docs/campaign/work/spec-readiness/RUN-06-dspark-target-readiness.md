# RUN-06 DSpark target readiness

**Observed:** 2026-09-12, Europe/Berlin

**Status:** preparation v2; released DSpark is structurally ready for a bounded pair screen, with target-weight coupling still open

**Owner:** worker-runtime
**Scope:** CPU/read-only source and public metadata inspection. No large weight download, model forward, target cache, training, server launch, GPU, SSH, package change, or state edit was performed.

## Decision

The released `MiniCPM5-2B-DSpark` is **eligible for a bounded artifact and
pair screen against the campaign target**. It is not a quality or latency
winner. The released draft was trained against the released `MiniCPM5-2B`
weights, while the campaign target is Midtrain-derived and will receive SFT/RL
weight changes. That is a coupling to measure after the target is selected,
not a reason to reject the experiment when the target structure and tokenizer
match.

The small public metadata check found that the released target's config and
tokenizer are byte-identical to the pinned campaign Midtrain config and
tokenizer. The released draft config is the expected
`Qwen3DSparkModel` with the exact 42-layer target geometry, five target taps,
block size 7, and vocabulary 130560. The normal b10453 converter registration
route resolves this architecture to `conversion.qwen.DSparkModel`; the special
`--dspark` switch is a separate DeepSeek-V4 export path and must not be used
for this Qwen3 DSpark config.

An official public GGUF also exists. Its remote file identity is recorded
below, but it was not downloaded. Root can choose that artifact or convert the
released source checkpoint after a bounded transfer approval. The remaining
gates are local GGUF/header verification, pairing with the root-admitted SFT
target, and exact native PRM-03 parity/latency measurement.

Adapted DSpark remains **NO-GO pending a MiniCPM/Llama training port**. The
lighter AR draft remains a **conditional screen-only** candidate. MTP remains
NO-GO because the Midtrain target has no MTP/NextN tensors. EAGLE-3 and DFlash
remain reserved without target-specific artifacts. `ngram-mod` remains a
comparator; the lead's current mechanical-parity result has no speed gain.

## Pinned target and released metadata

The campaign target is
`/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain` at revision
`8dc5f6055b90fe4b9422340810b270b9569f37f3`.

| Item | Campaign target | Released target metadata |
| --- | --- | --- |
| Config | SHA256 `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180` | Same SHA256; official `openbmb/MiniCPM5-2B`, API revision `12a3808a956f869c767195e9266b59c4d21d92e2` |
| Tokenizer JSON | SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` | Same SHA256, byte comparison passed |
| Tokenizer config | SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` | Same SHA256, byte comparison passed |
| Geometry | `LlamaForCausalLM`; 42 layers; hidden 2048; FFN 6144; heads 16; KV heads 2; head dimension 128; vocabulary 130560 | Same values |
| Special IDs | BOS 0; raw EOG `[1, 130073]`; pad 1 | Same values |
| Campaign weights | `model.safetensors`, 5,033,557,128 bytes, SHA256 `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad` | Not downloaded; official source tree reports 5,033,557,096 bytes, LFS OID `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c` |
| Target tensor header | 381 tensors; no observed `mtp`, `nextn`, `draft`, `dspark`, or `speculator` keys | Not inspected as weights |

The PRM-07/PRM-08 protocol remains manual BOS 0, no automatic special tokens,
no chat template, and terminal EOS 1. The second raw EOG ID 130073 remains a
native termination guard; it is not a reason to change the frozen serving
profile.

Released DSpark metadata was fetched from the official repository without its
weights:

| Item | Identity |
| --- | --- |
| Source repo | `openbmb/MiniCPM5-2B-DSpark`, API revision `114a20fdbf53220712c7fbdd7dccddbf1dedebb4` |
| Source file list | `.gitattributes`, `README.md`, `config.json`, `model.safetensors`; source tree SHA256 `79f1f91a9ba664f4183ddd6bb7c00b033259388527aec570c71a3627d8ba49b3` |
| Source config | SHA256 `bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b` |
| Source weight | 647,558,522 bytes; LFS OID `ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97`; not downloaded |
| Draft contract | `Qwen3DSparkModel`; five layers; hidden 2048; FFN 6144; heads 16/KV 2/head 128; target layers 42; taps `[1, 10, 20, 30, 39]`; block 7; mask token 75982; vocabulary 130560; BF16 |
| Draft config hash | `bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b` |

The official public GGUF search found:

| Repository/file | Remote identity | Interpretation |
| --- | --- | --- |
| `openbmb/MiniCPM5-2B-DSpark-GGUF/MiniCPM5-2.6B-DSpark.gguf` | repo revision `a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3`; 652,730,240 bytes; LFS OID `57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26` | Official public BF16 GGUF. Filename says 2.6B while the config/card says MiniCPM5-2B; bind the downloaded file hash and inspect its header before use. |
| `aj9o9/MiniCPM5-2B-DSpark-GGUF` | F16 652,732,352 bytes, Q8_0 349,218,720 bytes; remote LFS OIDs recorded in the receipt | Third-party converted alternatives. Their README says conversion used `--target-model-dir`; no trust or quality promotion is implied. |

The official target GGUF repository also exists with F16, Q4_K_M, and Q8_0
files. It is a useful vendor-target comparator, but campaign acceptance must
use the root-admitted Midtrain/SFT target identity.

## Normal converter route

The v1 memo incorrectly treated the special `--dspark` guard as a generic
converter block. The traced normal path is:

1. `convert_hf_to_gguf.py:241-247` calls
   `get_model_architecture`; the released config's first architecture is
   `Qwen3DSparkModel`.
2. `conversion/__init__.py:364-372` maps that name to module `qwen` and
   returns the registered class. `ModelBase.register` stores the decorator
   entry at `conversion/base.py:1142-1150`, and
   `ModelBase.from_model_architecture` retrieves the same registry entry at
   `base.py:1160-1164`.
3. `conversion/qwen.py:701-711` registers `Qwen3DSparkModel` as
   `DSparkModel`, with `model_arch=DFLASH`, and normalizes the flat public
   config into `dflash_config`.
4. `conversion/qwen.py:636-665` requires `--target-model-dir`, reads its
   `architectures[0]`, and calls `get_model_class` for the target. For the
   campaign target this resolves `LlamaForCausalLM` to `conversion.llama.LlamaModel`.
   The draft therefore reuses the target tokenizer/vocabulary path.
5. `conversion/qwen.py:667-677` emits `dflash.block_size` and stores public
   taps as GGUF extract-layer IDs one greater than the public IDs. The GGUF
   tensor mapping includes `markov_w1`, `markov_w2`, and confidence projection
   names (`gguf-py/gguf/tensor_mapping.py:1319-1334`).
6. `convert_hf_to_gguf.py:261-270` applies `--dspark` only when the source
   architecture is `DeepseekV4ForCausalLM`. That branch selects
   `DeepseekV4DSparkModel`; it is not needed for the released
   `Qwen3DSparkModel` draft.

The no-weight registration exercise resolved:

```text
draft_architecture= Qwen3DSparkModel
draft_class= DSparkModel
draft_model_arch= MODEL_ARCH.DFLASH
target_architecture= LlamaForCausalLM
target_class= LlamaModel
target_model_arch= MODEL_ARCH.LLAMA
```

This establishes converter dispatch and metadata plumbing. It does not prove
that a downloaded GGUF has complete tensors or that the candidate matches the
campaign weights numerically.

Pinned source identities:

| Source | SHA256 | Read points |
| --- | --- | --- |
| `conversion/__init__.py` | `7f5d0adaa7f932420418351994f65c14e7b05de951edf388b3d78ee4e7004997` | normal module map and `get_model_class` |
| `conversion/base.py` | `7dea4942ddc397c6d0189633265e42910e9178c5ce9af0a2d6c0dc08bda2de2b` | registration and registry lookup |
| `conversion/llama.py` | `17285e94476383dd84e484ef4209cf0e509a2a1ac93516b27f0ea057baa8961f` | `LlamaForCausalLM` registration |
| `conversion/qwen.py` | `65c61155458078232dd3f9d23284710fa39c29cf7ecc341194c825cf5334f43f` | DSpark class, target tokenizer and metadata route |
| `convert_hf_to_gguf.py` | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` | architecture dispatch and special DeepSeek-only flag |
| `gguf-py/gguf/tensor_mapping.py` | `2f40f1596e77478d4884c5a94a6799737d36a6eee34a09af6008692635195b7c` | DSpark Markov/confidence tensor names |
| `common/speculative.cpp` | `81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3` | target taps, hidden dimensions, block size, mask, Markov detection |
| `common/arg.cpp` | `566a8122afc02bbfcca572b7ac8d7d7d985b2b087b780434be75f588184f458d` | `draft-dspark`, `-md`, and draft length flags |

## Compatibility matrix

| Path | Current evidence and coupling | Decision |
| --- | --- | --- |
| **Released MiniCPM5-2B-DSpark** | Public config and campaign target share all required dimensions, EOG IDs, and tokenizer bytes. Normal converter registration works. The draft was trained on released target weights, so acceptance after Midtrain/SFT changes is unknown and must be measured. | **READY for bounded artifact conversion/structural screen and experimental pair.** No quality or speed promotion. |
| **Adapted DSpark** | Official DeepSpec trainer has Qwen3/Gemma4 implementations, not a MiniCPM/Llama trainer. A port needs target-cache generation, model/config/trainer work, and a converter/metadata fixture. | **NO-GO pending a scoped port.** |
| **Lighter AR draft** | Local MiniCPM1B layer-drop candidates are Llama models with hidden 1536 and 12/16/20 layers. `draft-simple` can avoid target hidden taps, but candidate quality and speed are unmeasured. | **Conditional screen only.** |
| **MTP** | Midtrain header has no MTP/NextN fields; unrelated Qwen MTP artifacts cannot pair with this target. | **NO-GO.** |
| **EAGLE-3** | Runtime supports it and requires exactly three target extract layers plus target hidden metadata (`src/models/eagle3.cpp:3-22`), but no MiniCPM artifact/recipe is bound. | **Reserve.** |
| **DFlash** | Runtime requires target-layer extraction, target hidden size, block size, mask token, and a target-specific draft (`common/speculative.cpp:907-1014`). | **Reserve.** |
| **ngram-mod** | Model-free mode is available; current lead mechanical parity has no speed gain. | **Comparator only.** |

## Smallest actual pair screen

This is a future live gate. It uses no held-out S1 or DEV data and does not
require a target cache for the already-trained released DSpark.

1. Root binds the downloaded draft file hash and GGUF header. Require
   architecture `dflash`, Markov tensors, `dflash.block_size=7`, mask token
   75982, extract layers `[2, 11, 21, 31, 40]`, hidden/vocabulary dimensions
   2048/130560, and tokenizer SHA256
   `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`.
2. Root binds the selected campaign target GGUF and its parent config,
   tokenizer, SFT-merge, and weights hashes. A target weight change leaves the
   released draft eligible for remeasurement; it does not trigger automatic
   cache regeneration or draft refresh.
3. Use exactly 12 admitted TRAIN rows, six short-band and six long-band,
   fixed IDs and stored prompt IDs. Do not use DEV, S1, or held-out traces.
   The native PRM-03 profile is context 4096, batch/ubatch 256, parallel 1,
   six target threads, two HTTP threads, and completion `n_predict=192`.
4. Launch the baseline target with the existing profile, then the candidate
   with only the draft options added:

   ```text
   llama-server -m TARGET.gguf -md DSPARK.gguf \
     --spec-type draft-dspark --spec-draft-n-max 7 \
     -t 6 -tb 6 --threads-http 2 --parallel 1 \
     -c 4096 -b 256 -ub 256 -ngl 0
   ```

   The client must preserve the frozen wire contract: `/tokenize` with
   `add_special=false`, `parse_special=false`; `/completion` with
   `[0] + stored prompt IDs`, `n_predict=192`, temperature 0, returned token
   IDs, and canonical terminal EOS 1. Do not add `--jinja` or send prompt text
   in this native profile.
5. Run one baseline and one candidate request per row. Record exact returned
   IDs/raw text, first divergence, terminal reason, drafted/accepted counts,
   prompt length, TTFT, wall time, and parser/load errors. The current
   `runtime_native_probe.py` accepts only `baseline` and `ngram-mod` arms, so a
   separate DSpark client arm or root-owned probe extension is a live blocker;
   no file outside this packet was changed.
6. The 12-case screen is mechanically green only with 12/12 exact baseline
   outputs, matching prompt/terminal geometry, complete metadata, and no
   load/parser/tap errors. Report latency as a screen, with no quality claim.
   Promotion still requires the registered 20 short + 20 long traces, two
   repetitions, exact parity, at least 1.4x end-to-end speed, bootstrap lower
   bound above 1.0, and no p95 regression.

For a target weight change, first rerun this pair screen. Refreshing the draft
or creating a target-specific cache is justified only by measured usefulness
and must remain inside the existing conditional two-GPU-hour reassignment from
optional consolidation. The released draft itself needs no automatic cache
regeneration.

## Remaining gates and stop rules

The coupling blockers are concrete and finite:

* no local released DSpark GGUF hash has been recorded yet;
* no GGUF header/tensor inventory has been inspected for the public file;
* root must supply the selected target GGUF and parent identity;
* the existing native probe needs a DSpark arm or an equivalent root-owned
  client while retaining context 4096 and cap 192;
* exact output parity and latency remain unmeasured against Midtrain/SFT
  weights.

Stop the candidate screen if the downloaded GGUF does not contain the expected
architecture, Markov tensors, block/tap/mask metadata, hidden/vocabulary
dimensions, or tokenizer identity; if any output diverges; or if the formal
latency gate is not credible. A target checkpoint change alone is a reason to
remeasure, not to discard or silently refresh the draft. No quality claim is
made from vendor acceptance or this readiness packet.

## Artifact and evidence files

The packet records small metadata only; no large model file was transferred.

| Artifact | SHA256 |
| --- | --- |
| `released-dspark-metadata/model-api.json` | `5fa58687e693b10041d42f47b94448e06e399390aa825e4cfa67ac44f2e345cc` |
| `released-dspark-metadata/config.json` | `bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b` |
| `released-dspark-metadata/README.md` | `136eca2280c1fb7d67c18fa0f4540999fcbc028d3ed0c347c94d8f157854e333` |
| `released-dspark-source-tree.json` | `79f1f91a9ba664f4183ddd6bb7c00b033259388527aec570c71a3627d8ba49b3` |
| `official-dspark-tree.json` | `3f52fbbfe1bf123196f063bf5f74d659728cf58999c3035d68d9e3e69fb16c81` |
| `openbmb-MiniCPM5-2B-DSpark-GGUF-README.md` | `a0fd808d177b5ae6c1e1cf99202845966adba05c5e4be01eec8524b3c64e4801` |
| `released-target-metadata/model-api.json` | `fb3411b0817056016952f15fce524fad42723dc8876d854b9d52dfffc0954cdf` |
| `released-target-metadata/config.json` | `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180` |
| `released-target-metadata/tokenizer.json` | `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| `released-target-metadata/tokenizer_config.json` | `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` |
| `released-target-metadata/special_tokens_map.json` | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `released-target-metadata/generation_config.json` | `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1` |
| `released-target-source-tree.json` | `0c78386c1d90cffe2cdc3544eba4d8aca63b6cb210fb9f5bf1ae57fef04cc92e` |
| `openbmb-MiniCPM5-2B-GGUF-api.json` | `1da6bf3564ab9f9baeff132f34bcc73c3d0c0408a6195168ae164541f7401f46` |
| `official-target-tree.json` | `e7dfaade0f2ed7c9efca6d511105585d409d81d418c5a1dd4e102de34c5fb640` |
| `dspark-model-search.json` | `ad017c6881a4618d95213ade702a6b6de25a819620ad811523f60f0d4f107400` |
| `normal-converter-registration.txt` | `06a262d595f2db0a80cdd01f0b1e5c481a5c1ca1490ebd7b10d84acacfd7590e` |
| `normal-converter-dispatch.txt` | `6a47b527a7ed27d7d8f4b09c3ebadaf6735e709e3c1cacdb0884b718f795e87a` |
| `released-metadata-comparison.txt` | `79847a880c5964b332c2170be41b9660451a2725944479be5c4aa0ab7208f5c5` |

The v1 artifacts remain preserved as:

* `RUN-06-dspark-target-readiness.v1.md`, SHA256
  `298e962f5f88af5e6e9fa63755a5e9fb611274800be78d4e06e2af2885c90071`;
* `RUN-06-dspark-target-readiness.v1.json`, SHA256
  `666d72821c8a6bd1a18b662a716fe10e7977df26f088296cc6eeadc886d9ba07`.

## Primary sources

* [Released MiniCPM5-2B-DSpark card](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)
* [Official MiniCPM5-2B-DSpark GGUF repository](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF)
* [Released MiniCPM5-2B target](https://huggingface.co/openbmb/MiniCPM5-2B)
* [DeepSpec repository and workflow](https://github.com/deepseek-ai/DeepSpec)
* [Official DSpark trainer](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/trainer/dspark_trainer.py)

These sources establish public model and trainer contracts. They do not
establish acceptance or latency for the campaign's Midtrain/SFT/RL weights.
