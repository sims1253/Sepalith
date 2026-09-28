# RUN-08 MiniCPM5 DSpark draft readiness recovery

Observed 2026-09-14. This is a CPU-only readiness packet. It performed no GPU/CUDA initialization, target forward, cache generation, training, cloud launch, weight download, model export, or sealed-final action.

## Decision

The smallest implementable target-matched path is a fresh five-layer DSpark draft trained against a target-generated cache made from a deterministic 256-row subset of the admitted R2 TRAIN rows. Use the existing official `Qwen3DSparkTrainer` and `Qwen3DSparkModel` with the MiniCPM5 target config cloned into the draft. The released GGUF is useful as a serving and geometry reference, but it cannot warm-start this path: only its GGUF is present locally, no GGUF-to-HF reverse converter is present, and the official trainer starts from scratch (or resumes its own checkpoint).

The first target must be a root-admitted theta0 HF snapshot. Generate the teacher completions with that snapshot at temperature zero and protocol EOS 1, then run the target once over each complete sequence to capture the five hidden taps and final hidden state. Existing stored R2 tails are authored labels; they are valid prompt/provenance input but are not a substitute for the target-generated tails required by `SPECULATIVE-PATHS.md`.

A 128-row, 64-step pilot is the storage-safe fallback if the host cannot reserve 8.1 GB for the 256-row cache. The pilot only establishes that the target-matched training/export path works. It cannot establish quality or promotion eligibility.

## Evidence from the current tree

* `docs/campaign/SPECULATIVE-PATHS.md` requires the target-matched draft to use TRAIN rows, preserve S1 as held out, and charge adaptation plus any final refresh to the same conditional two-GPU-hour reassignment.
* `docs/campaign/receipts/RUN-07-theta0-draft-acceptance.json` screened the released DSpark against theta0 and found no speed gain. That screen used a draft trained for a different released target and did not train a target-matched draft, so it does not close this path.
* `docs/campaign/work/r2-task-mixture-v1/verified-corrected-short-token-rows.jsonl` is 8,526/8,526 `split=train`, renderer `zeta2-prm03-v1`, tokenizer SHA `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, and file SHA `85e2d86d4d2fc7f17cae9659dbc62d6eaa679ca49594537a9f317d68a120204e`. Every row passed BOS 0, one terminal protocol EOS 1, target-boundary, target-tail, and 192-token including-EOS checks.
* The local target config is ordinary `LlamaForCausalLM`: 42 layers, hidden 2048, intermediate 6144, attention 16/KV 2, head dimension 128, vocabulary 130560, BOS 0 and PAD/EOS 1. Config SHA is `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`.
* The locally verified released draft is `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-bf16/MiniCPM5-2.6B-DSpark.gguf`, 652,730,240 bytes, SHA `57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26`. Its geometry is dflash, five blocks, block size 7, hidden 2048, taps `[1,10,20,30,39]` in the HF config (stored GGUF layers are `[2,11,21,31,40]`), mask 75982, confidence head, and Markov rank 256. It has no target embedding or LM head because runtime supplies those from the target.
* No DeepSpec checkout, MiniCPM trainer, target cache, or HF DSpark checkpoint was found in the known local model/training paths. The local R2 cloud payload is a task-SFT control payload; it does not contain DeepSpec or a DSpark cache/trainer.

The CPU checks and derived cache sizes are in `cpu-contract-128.json`, `cpu-contract-256.json`, and `cpu-contract-512.json`, produced by `validate_cpu_contract.py`.

## Exact implementation seams

### 1. Teacher rows (new small adapter)

Add a DeepSpec-side `prepare_minicpm5_teacher_rows.py` or equivalent cloud payload stage. Read only the selected IDs from the admitted TRAIN JSONL. For each row, retain `prompt_ids = row["input_ids"][:row["target_start"]]` (including BOS 0), and call the root-admitted theta0 target with:

```text
do_sample=False, temperature=0, max_new_tokens=192,
eos_token_id=1, pad_token_id=1, use_cache=True
```

Do not apply a chat template or automatic BOS/EOS. Accept only a completion that ends in exactly protocol EOS 1 within the cap; record the target model path/hash, generation settings, selected row IDs, and output-row hash. A cap hit, control token, duplicate EOS, or other protocol failure stops the stage. This stage is the only new teacher-generation code; `campaign_eval.py` contains reusable greedy HF generation and terminal handling patterns, but its DEV/evaluator contract should remain unchanged.

### 2. Pretokenized target cache bridge (required code)

The official `scripts/data/prepare_target_cache.py` already provides target hooks, distributed ranges, asynchronous writer, manifest construction, and shard finalization. Its `ConversationCollator` is unusable for R2 rows because it requires `conversations` and applies a Qwen/Gemma chat template. Add a `PretokenizedCollator` (or a small wrapper script importing the official hook/writer functions) that consumes the generated rows directly:

```python
ids = torch.tensor(row["input_ids"], dtype=torch.long)
attention_mask = torch.ones(len(ids), dtype=torch.long)
loss_mask = torch.zeros(len(ids), dtype=torch.long)
loss_mask[row["target_start"]:] = 1  # target body + terminal + protocol EOS
```

Run the existing `run_target_forward_with_hooks` with `target_layer_ids=[1,10,20,30,39]`. It already resolves a generic target's `.model.layers`, captures each layer hook, and returns `target_last_hidden_states`. Write both `target_hidden_states` and `target_last_hidden_states`; the latter is required for aligned target logits in the DSpark loss. Keep the v2 writer/index protocol unchanged. Add manifest fields for `target_weights_sha256`, source-row hash, selected-ID hash, tokenizer SHA, renderer, generation recipe, target layer IDs and target config SHA. The official validator compares the target path string, so the additional hashes are needed to prevent a path alias from silently reusing a stale cache.

### 3. Llama-compatible DSpark config (one model patch)

Reuse `Qwen3DSparkTrainer`; its trainer only builds the draft config, calls `Qwen3DSparkModel`, passes cache tensors, and computes the standard CE/L1/confidence loss. In `deepspec/modeling/dspark/qwen3/config.py`, after cloning `target_config`, set the field that the Qwen3 attention class unconditionally reads:

```python
draft_config.sliding_window = getattr(target_config, "sliding_window", None)
```

MiniCPM's `LlamaConfig` supplies the other fields used by the Qwen3 implementation (`hidden_size`, `intermediate_size`, `hidden_act`, `rms_norm_eps`, `head_dim`, attention/KV heads, attention bias/dropout and `rope_parameters`). The draft config should be:

```text
architectures: ["Qwen3DSparkModel"]
num_draft_layers: 5
block_size: 7
target_layer_ids: [1, 10, 20, 30, 39]
mask_token_id: 75982
markov_rank: 256, markov_head_type: vanilla
confidence_head_alpha: 1.0, confidence_head_with_markov: true
loss_decay_gamma: 4.0, ce_loss_alpha: 0.1, l1_loss_alpha: 0.9
```

For the bounded A10G pilot use `num_anchors=256` if the 512-anchor default exceeds memory; record this as a training setting and do not claim it is the released recipe. Start with `torch_compile=False`, local batch 1, global batch 16 or 32, no sharding, and a fixed `max_train_steps=128`. The official `BaseTrainer` copies the target embeddings and LM head to the draft and freezes them; no target backpropagation is needed after cache creation. It prints “Training from scratch” when no DeepSpec checkpoint exists. This is the smallest complete route from the current local assets.

### 4. Export (existing converter, no source repair)

After a successful DeepSpec checkpoint, use the pinned b10453 converter's normal Qwen DSpark dispatch, with the selected target directory for vocabulary/metadata:

```bash
python3 /home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py \
  <draft-checkpoint> --outtype bf16 --outfile <draft.gguf> \
  --target-model-dir <root-admitted-target-hf-dir>
```

Do not pass converter `--dspark`: in this pinned converter that flag is restricted to `DeepseekV4ForCausalLM`. The already verified normal route is `Qwen3DSparkModel -> conversion.qwen.DSparkModel -> DFlashModel.set_vocab(target_model_dir) -> LlamaModel`. The exporter removes draft embed/lm-head tensors, imports target tokenizer metadata, writes block/tap/mask metadata, and preserves Markov/confidence tensors. A CPU header audit must fail unless architecture `dflash`, five blocks, block 7, taps `[2,11,21,31,40]`, mask 75982, hidden 2048, tokenizer arrays, and expected top tensor shapes are present.

Runtime arm integration remains root-owned: the current `runtime_native_probe.py` accepts baseline/ngram arms and needs a DSpark arm. The serving screen must bind the selected target and draft hashes and use `-md`, `--spec-type draft-dspark`, and `--spec-draft-n-max 7`.

## Cache and budget

The canonical v2 cache stores int32 IDs, uint8 attention/loss masks, five BF16 tapped states and one BF16 final state. That is exactly 24,582 bytes per sequence token; each index record is 56 bytes. The CPU audit measured the following deterministic hash-order subsets:

| subset | sequence tokens | cache payload | cache GiB | index |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 169,705 | 4,171,688,310 B | 3.885 | 7,168 B |
| 256 (default) | 325,866 | 8,010,438,012 B | 7.460 | 14,336 B |
| 512 | 657,067 | 16,152,020,994 B | 15.043 | 28,672 B |
| all 8,526 | 10,505,668 | 258,250,330,776 B | 240.5 | 477,456 B |

The two-hour replacement ledger is a hard ceiling, including a possible final refresh. A conservative schedule for one A10G is:

| phase | cap | action |
| --- | ---: | --- |
| theta0 teacher + 256-row cache | 0.35 GPU-hours | target generation plus five taps and final hidden state |
| draft adaptation | 0.55 GPU-hours | 128 optimizer steps, fixed cache, no target forward |
| export/header/CPU checks | 0 GPU-hours | run on CPU; no promotion claim |
| final target cache refresh reserve | 0.35 GPU-hours | only after the unchanged draft fails or benefits from final-target remeasure |
| final draft refresh reserve | 0.55 GPU-hours | same config, same 256 IDs, final target cache |
| unplanned/retry reserve | 0.20 GPU-hours | hard stop; never borrow from final evaluation |
| **total** | **2.00 GPU-hours** | replacement only |

The cloud control receipt records an existing Anyscale all-in ceiling of USD 20 for this conditional work, within the already approved campaign ceiling. No cloud job is authorized by this packet. If the first eight-row profile projects above its phase cap, stop and use the 128-row/64-step fallback or leave the pillar unpromoted. If the final target is unchanged and the draft passes the root-owned runtime screen, skip the refresh and preserve the reserve.

## CPU checks run

```text
python3 validate_cpu_contract.py --selected-rows 128 --json-out cpu-contract-128.json
python3 validate_cpu_contract.py --selected-rows 256 --json-out cpu-contract-256.json
python3 validate_cpu_contract.py --selected-rows 512 --json-out cpu-contract-512.json
python3 -m py_compile validate_cpu_contract.py
```

All passed. The validator uses only the JSONL/config metadata and pure-Python struct arithmetic. It does not import torch or transformers and never reads model weights. It verifies every source row, stable ID selection, target tail geometry, loss-mask boundary, target config, v2 cache dtypes/shape, per-token bytes and 56-byte index records.

## Primary references

* [SPECULATIVE-PATHS.md](../../SPECULATIVE-PATHS.md)
* [RUN-06 DSpark target readiness](../../receipts/RUN-06-dspark-target-readiness.json)
* [RUN-06 released artifact preparation](../../receipts/RUN-06-released-dspark-artifact-preparation.json)
* [RUN-07 theta0 draft acceptance](../../receipts/RUN-07-theta0-draft-acceptance.json)
* [DeepSpec README](https://github.com/deepseek-ai/DeepSpec)
* [DeepSpec DSpark trainer](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/trainer/dspark_trainer.py)
* [DeepSpec target-cache preparation](https://github.com/deepseek-ai/DeepSpec/blob/main/scripts/data/prepare_target_cache.py)
* [DeepSpec cache protocol](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/data/target_cache_dataset.py)
* [Pinned llama.cpp converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
