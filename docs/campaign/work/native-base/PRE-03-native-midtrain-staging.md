# PRE-03 native Midtrain base staging

Status: **paused at root request for host-memory safety**. The pinned public weight transfer was started with one CPU thread and stopped after the root reported possible PC/OOM and heavy pagefile activity. The partial file is preserved. It was not hashed, promoted, resumed, deleted, or read after the stop.

## Transfer identity

The requested source is the exact public revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`:

```text
https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain/resolve/8dc5f6055b90fe4b9422340810b270b9569f37f3/model.safetensors
```

Expected weight identity:

- bytes: `5033557128`
- SHA256: `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad`
- final path: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/model.safetensors`

The direct `curl` stream used no credentials, cookies, authorization headers, or custom request headers. It was bounded by both GNU `timeout 420s` and a reader-side stream ceiling of `5033557129` bytes. The transfer wrote only to a unique same-directory part file.

## Pause evidence

Observed progress before interruption:

| Elapsed | Received | Approx. rate |
| ---: | ---: | ---: |
| 10.0 s | 780,140,544 bytes | 74.14 MiB/s |
| 20.1 s | 1,711,276,032 bytes | 81.39 MiB/s |
| 30.1 s | 2,558,525,440 bytes | 81.13 MiB/s |
| 40.1 s | 3,539,992,576 bytes | 84.18 MiB/s |

After the interrupt, the preserved part measured `4194304000` bytes by `stat`. The Python staging process and its curl child are no longer active; the curl log records exit 23 after the reader closed its pipe. The final weight does not exist, and no part file was removed.

The unique preserved part is:

```text
/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/.model.safetensors.part-3316664-1789231351948352402
```

No full SHA256 or fsync promotion was attempted because the transfer was stopped. Resumption requires root to release the memory/I/O pause; the current script deliberately starts a fresh unique stream and does not overwrite or silently trust a partial.

## Small files

Before starting the weight stream, the six requested small files in the existing frozen source directory were individually checked against `PRE-05-minicpm-artifacts.json`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 704 | `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180` |
| `generation_config.json` | 213 | `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1` |
| `tokenizer.json` | 9894271 | `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| `tokenizer_config.json` | 94391 | `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` |
| `special_tokens_map.json` | 551 | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `chat_template.jinja` | 9060 | `cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c` |

The small files were not copied before the pause. The native directory therefore contains only the preserved weight part.

## Root continuation gate

Root must first clear the host-memory/pagefile condition. After release, the remaining work is a fresh bounded transfer or an explicitly root-approved continuation strategy, followed by full-size/SHA verification, file and directory fsync, exclusive promotion, and copying/verifying only the six small files. No model load, GPU, SSH, cloud, server, evaluation, or RL operation is part of this packet.

Receipt: `docs/campaign/receipts/PRE-03-native-midtrain-staging.json`.
