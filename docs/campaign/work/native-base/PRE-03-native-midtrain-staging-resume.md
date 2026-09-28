# PRE-03 native Midtrain staging resume

Status: **complete after root memory/I/O release**. The initial paused receipt and 4,194,304,000-byte part were preserved until root re-admitted this bounded operation. A single HTTP Range suffix was downloaded, validated before append, joined to the preserved part, fully hashed, fsynced, and exclusively promoted. The six requested companion files were then copied from the already frozen source directory and individually verified.

## Weight identity

- Source: `openbmb/MiniCPM5-2B-Midtrain`, revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`
- URL: `https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain/resolve/8dc5f6055b90fe4b9422340810b270b9569f37f3/model.safetensors`
- Native path: `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/model.safetensors`
- Bytes: `5033557128`
- SHA256: `38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad`

The preserved part began at offset `4194304000`. The request returned `HTTP/2 206` with exactly:

```text
Content-Range: bytes 4194304000-5033557127/5033557128
Content-Length: 839253128
```

The suffix hash is `28d668ed85d2e963fb4137f50a3501ce934c2a5508b28300cca6aa226cd984c8`. The combined file has the expected full size and SHA256. File and directory fsync completed, and promotion used an exclusive hard link so an existing final could not be replaced. Both temporary part files were removed only after successful verification.

The response header evidence was redacted to remove the signed redirect `Location`; the retained header record keeps the 206 status, exact range, length, pinned repository commit, linked size, and linked ETag. No credentials, authorization headers, cookies, or custom request headers were supplied or retained.

## Companion files

Each source file was checked against `PRE-05-minicpm-artifacts.json` before copying and rechecked after copy:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 704 | `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180` |
| `generation_config.json` | 213 | `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1` |
| `tokenizer.json` | 9894271 | `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` |
| `tokenizer_config.json` | 94391 | `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` |
| `special_tokens_map.json` | 551 | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `chat_template.jinja` | 9060 | `cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c` |

The native directory now contains exactly these seven files. No model load, GPU, SSH, cloud, server, evaluation, or RL operation was performed.

## Records

- Resume launcher: `midtrain-resume.launch.json`
- Redacted range headers: `midtrain-resume.headers`
- Transfer log: `midtrain-resume.log`
- Terminal record: `midtrain-resume.terminal.json`
- Machine-readable result: `midtrain-resume.json`
- Initial pause record: `docs/campaign/receipts/PRE-03-native-midtrain-staging.json`
- Final resume receipt: `docs/campaign/receipts/PRE-03-native-midtrain-staging-resume.json`
