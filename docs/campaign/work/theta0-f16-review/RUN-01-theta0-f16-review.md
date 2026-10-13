# RUN-01 theta0 F16 export review

Review result: `pass_metadata_extent_runtime_stop_gate`  
Export: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-runtime-gguf/model-F16.gguf`

The stopped converter record reports exit code 0 in 17.250138356001116 seconds. The export is exactly 5,039,006,496 bytes with SHA256 `ee7ba2fd7e7b6446d74224e1e1f9e4724fbcfa54e56e60c4b623981d1e4f45ed`. The hash used 1 MiB reads and `POSIX_FADV_DONTNEED` after each completed range.

The metadata-only GGUF audit reports version 3, header and reader tensor count 381, with 296 F16 and 85 F32 tensors. The derived Llama architecture inventory has the same 381 names and shapes. Tensor inventory SHA256 is `e9be8430178529a123cd683310230af8e5900c8932e67869cd72b62be0add92c`; names/shapes SHA256 is `3ce24d2ce37d77274e8e374fc6a111dedd9007a6e51d5775c0f1fdb9abe62f9a`. The tensor data begins at offset 5,145,376, has no overlaps or gaps, and ends exactly at the file size. Aggregate tensor bytes are 5,033,861,120.

The parent theta0 metadata is bound by parent-manifest SHA256 `1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12`. Config SHA256 is `f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991`, generation config SHA256 is `7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269`, tokenizer JSON SHA256 is `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, and tokenizer config SHA256 is `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`. The parent manifest's merged-weight entry is `model.safetensors`, 5,033,557,128 bytes, SHA256 `499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d`; it was stat-checked but not read.

GGUF architecture scalars match the parent config: Llama, vocabulary 130,560, context 131,072, hidden size 2,048, intermediate size 6,144, 42 blocks, 16 attention heads, 2 KV heads, head dimension 128, and RoPE theta 5,000,000. Parent tokenizer token IDs are exact across all 130,560 entries, the 129,794 BPE merges match after converting HF merge pairs to GGUF's joined strings, and the 9,060-byte chat template matches exactly.

The old SFT-primary-step500 F16 export has the same tensor names, shapes, order, and F16/F32 type inventory. Its accepted prior hash is `50f523af997f2f77dcbd187adde8a36dbc41529932703fd016e506b172761725`. Metadata differences are limited to the expected run label and converter bookkeeping plus tokenizer flags: step1000 has `general.name = SFT Primary Step1000 Theta0`, `GGUF.kv_count = 33`, `tokenizer.ggml.add_bos_token = false`, and `tokenizer.ggml.add_eos_token = false`; step500 has `SFT Primary Step500 Runtime`, `GGUF.kv_count = 32`, `add_bos_token = true`, and no `add_eos_token` key. The step1000 BOS/EOS flags match the theta0 tokenizer configuration.

One runtime gate remains explicit. GGUF stores scalar `tokenizer.ggml.eos_token_id = 1`, while the HF config and generation config bind EOS IDs `[1, 130073]`. Token 130073 is present as `<|im_end|>`. Native runtime EOG/stop handling needs a separate verification gate; this audit does not prescribe an API change.

The Q8 companion is partial and not accepted; no Q6 candidate was reviewed or admitted. No model inference, tensor-value comparison, quantized quality claim, or live runtime acceptance was performed. The parent model weight was not read; only its manifest and file size were checked.

Validation command:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python3.10 docs/campaign/work/theta0-f16-review/verify_theta0_f16.py
```

The full report and check booleans are in `theta0-f16-metadata-review.json`; the copied receipt is [RUN-01-theta0-f16-intermediate-review.json](../../receipts/RUN-01-theta0-f16-intermediate-review.json).
