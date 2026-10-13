# RUN-06 released DSpark artifact preparation

Status: **complete for the bounded CPU artifact screen; pending root transfer, runtime load, and paired screen**. This packet downloaded one official file and inspected its GGUF header, embedded metadata, tokenizer records, and tensor inventory. It did not launch a model or server and makes no quality or latency claim.

## Artifact and transfer

The only downloaded object was `MiniCPM5-2.6B-DSpark.gguf` from the pinned Hugging Face repository revision `a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3`:

- URL: `https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF/resolve/a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3/MiniCPM5-2.6B-DSpark.gguf`
- Native path: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/released-dspark-bf16/MiniCPM5-2.6B-DSpark.gguf`
- Size: `652730240` bytes
- SHA256: `57df08640f0534a1aac075d1c8bdacdb2b7e5815da6f4e5cfd39ecac3a3f0c26`

`prepare_dspark_download.py` used one bounded `curl` process with `CUDA_VISIBLE_DEVICES` empty and one-thread BLAS/OpenMP variables. It wrote to a unique same-directory temporary file, verified size and SHA256, fsynced the file, promoted with an exclusive hard link, removed the temporary file, and fsynced the directory. An existing final file would have been verified and left untouched; a mismatched or concurrent final file would have caused refusal. The final directory contains only the verified GGUF and no part file. The transfer log and machine-readable transfer record are `dspark-download-curl.log` and `dspark-download.json`.

## CPU-only GGUF audit

`inspect_dspark_gguf.py` used the pinned `gguf.GGUFReader` from llama.cpp b10453 source commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. It reads only the header and metadata; it does not read tensor payloads or initialize a model. The complete report is `dspark-gguf-audit.json`.

The DSpark file is GGUF v3 with 37 on-disk key/value records (40 fields exposed by the reader after its three synthetic `GGUF.*` fields), 62 tensors, and reader data offset `5126496`. Its tensor types are 39 BF16 and 23 F32. The embedded geometry is:

- architecture `dflash`, five draft blocks, context length `131072`
- hidden/embedding `2048`, FFN `6144`
- attention heads/KV heads `16/2`, key/value length `128/128`
- RoPE base `5000000.0`
- block size `7`
- target extraction layers `[2, 11, 21, 31, 40]`
- mask token `75982`, `sample_from_anchor=true`, confidence head present

The source DSpark config (SHA256 `bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b`) identifies target layers `[1, 10, 20, 30, 39]`; the GGUF converter stores the corresponding one-based layer values above. This is a representation detail, not a changed tap set.

The inventory is exactly five copies of the eleven expected block tensors plus seven top-level tensors. The latter are `fc.weight`, `enc.output_norm.weight`, `conf_proj.weight`, `conf_proj.bias`, `markov_w1.weight`, `markov_w2.weight`, and `output_norm.weight`. The Markov tensors are `[256, 130560]`; the confidence projection is `[2304, 1]` plus a one-element bias; and the draft has no target `token_embd.weight` or `output.weight`, as expected for a DFlash/DSpark draft.

The campaign target used for the direct comparison is `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step500-runtime-gguf/model-Q8_0.gguf`, size `2679710464`, SHA256 `f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256`. It has 381 tensors (296 Q8_0 and 85 F32). Both file identities were recomputed during the audit.

## Tokenizer binding

Tokenizer equivalence was tested from the embedded GGUF fields themselves. The full `tokens`, `token_type`, and `merges` arrays match exactly between the draft and campaign Q8 target:

| Embedded field | Count | Canonical value SHA256 |
| --- | ---: | --- |
| `tokenizer.ggml.tokens` | 130560 | `c65bbb55f7bb9c8fc602752a463e92ea62f124976bd8a863453c25978184e824` |
| `tokenizer.ggml.token_type` | 130560 | `572212fc3ad4b628ba16108f5566a09dc5e5684cd6ee38c1678c723382858ffd` |
| `tokenizer.ggml.merges` | 129794 | `47593f7ea99535bc76465f6384bc0a4949cdcf6c857ce0a5a35379237828d192` |

The common tokenizer metadata, model/pre-tokenizer identifiers, BOS/EOS/unknown/padding IDs (`0/1/130074/1`), and chat template are exact. The three intentional runtime metadata differences are: the draft has `add_bos_token=false` while the target records `true`; the draft adds `add_eos_token=false`; and the draft adds `mask_token_id=75982`. The native client still sends manual BOS `0` and stored IDs under the locked protocol. The comparison does not expect an HF `tokenizer.json` SHA inside GGUF; the embedded arrays and fields are the binding test.

## Proposed root-owned Vulkan pair

The b10453 help capture proves `-md/--model-draft`, `--spec-type draft-dspark`, `--spec-draft-n-max`, `-ngl`, and `-ngld`. The official DSpark GGUF usage shape also specifies `-ngl 99 -ngld 99 -fa on`; the campaign profile below keeps the locked context/batch/thread geometry and the native stored-ID wire contract. Root must substitute paths after copying and hashing the files on the Vulkan host:

```text
/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-vulkan-avx2/bin/llama-server \
  -m /root-admitted/frozen-sft-step500-q8.gguf \
  -md /root-admitted/released-dspark-bf16/MiniCPM5-2.6B-DSpark.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 1840X \
  -t 6 -tb 6 --threads-http 2 --parallel 1 \
  -c 4096 -b 256 -ub 256 -lv 4 -ngl 99 -ngld 99 -fa on
```

The target must be the root-admitted Q8 artifact with the hash above, and the draft must retain the exact pinned hash above. The baseline and candidate need paired server settings, exact manual-BOS stored prompt IDs, temperature zero, `n_predict=192`, returned IDs, and canonical terminal EOS 1. Do not add `--jinja` to this native probe. Root should first check load/offload logs and then run the bounded TRAIN-only mechanical screen; a successful load is structural eligibility only. The released draft was trained for the released MiniCPM5 target, so its published acceptance and speed do not transfer to this SFT target without the paired measurement.

Remaining live gates are root-owned: transfer both files, launch one server, verify binary/source/model hashes and actual Vulkan offload, run the paired TRAIN screen, record draft/accepted counts and exact output parity, and decide whether the arm is useful. No final or DEV data was read or emitted by this packet, and no promotion is implied.

## Reproducibility records

- Transfer script: `prepare_dspark_download.py`
- Transfer record: `dspark-download.json`
- Transfer log: `dspark-download-curl.log`
- GGUF audit: `inspect_dspark_gguf.py`, `dspark-gguf-audit.json`
- Receipt: `docs/campaign/receipts/RUN-06-released-dspark-artifact-preparation.json`
