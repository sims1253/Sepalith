# RUN-09 theta0 Q8 local-CUDA DEV preparation

This packet prepares a root-owned native CUDA evaluation of the selected
step-1000 theta0 Q8 artifact. It contains no model launch and makes no quality
or Vulkan-equivalence claim. Root must review the input identities, host/GPU
lease, and fresh output location before running it.

The local b10453 CUDA bundle is already present at
`/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453`.
The model is the Q8 output admitted by
`RUN-01-theta0-quant-c-lead-review.json`:

* `/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf`
* 2,679,710,496 bytes, SHA256
  `22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559`

The server binary is build 10453, commit `3cb7ffb1a`, SHA256
`e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee`.
The preparation inspection used only `--version` and dynamic-library metadata;
it did not load the model or contact a server. The root launch must rehash the
model before starting the server. The launch environment removes inherited
`GGML_CUDA_*` values and sets `CUDA_VISIBLE_DEVICES=0` and
`GGML_CUDA_GRAPH_OPT=0`; graph option 1 is excluded by the known corruption
finding.

## Quality arm

Use a new ext4 run directory and a port other than the existing CPU server's
`18099` (the default is `18403`):

```sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
RUN=/home/m0hawk/.local/state/sepalith/campaign-20260915/runs/theta0-q8-cuda-dev-<unique>
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B \
  "$PLAN/docs/campaign/work/theta0-q8-cuda-dev-preparation/launch_theta0_q8_cuda_dev.py" \
  --run-root "$RUN" --port 18403 --mode quality
```

The helper owns only this fresh process group and sends TERM, then KILL after
15 seconds, on normal completion, interruption, or the 1,200-second server
guard. It never deletes an existing directory or kills by name. It waits for
`/health` status `ok` and `/props` `n_ctx=4096` before invoking the client.
The server profile is `ctx=4096`, `batch=256`, `ubatch=256`, `parallel=1`,
`threads=6`, HTTP threads 2, and `ngl=99`.

The quality client makes 75 independent requests from the corrected DEV panel
(`43` edit rows and `32` strict no-op rows), with the pinned HF tokenizer,
manual BOS `0`, native `/tokenize` parity, `n_predict=512`, greedy decoding,
`cache_prompt=false`, and a 120-second combined tokenize-plus-stream deadline
per case. Its 1,100-second global deadline leaves a 60-second reserve. Raw
text, returned IDs, stop/EOS, protocol result, parser result, scores,
denominators, and partial streams remain in the client output. Protocol or cap
failures stay visible and do not become quality successes.

The accepted scorer predates the six-row corrected panel and hard-codes the old
panel digest. `run09_corrected_dev_client.py` is a narrow adapter: it verifies
the corrected panel SHA and the unchanged scorer SHA, changes only the
scorer's expected panel digest, and calls the normal scorer. It rejects every
other panel path. The correction is therefore explicit and reviewable; scoring,
rendering, tokenization, and transport remain those of the accepted client.

## Optional 192-cap diagnostic arm

Run a separate fresh directory with the same server helper and
`--mode diagnostic`:

```sh
RUN=/home/m0hawk/.local/state/sepalith/campaign-20260915/runs/theta0-q8-cuda-train-diagnostic-<unique>
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B \
  "$PLAN/docs/campaign/work/theta0-q8-cuda-dev-preparation/launch_theta0_q8_cuda_dev.py" \
  --run-root "$RUN" --port 18403 --mode diagnostic
```

This uses only the existing four-row TRAIN fixture and
`runtime_native_probe.py`, with `cap=192`, `context=4096`, a 5,000 ms combined
foreground deadline, and a separate 60,000 ms diagnostic timeout. It is a
mechanical/native timing and token/EOS/cache diagnostic, not a DEV quality
run. Its denominator must not be merged with the 75-case quality denominator.

## Input and acceptance gates

The corrected panel is
`docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl`,
75 rows, SHA256
`7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035`; its
manifest is SHA256
`4528ff6dbd8cfe70580901d9254a1a6931dc8c334c5dd2c55e49cf92d1a5377a`.
The original immutable panel remains SHA256
`b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`.
The HF tokenizer files are checked by the client as tokenizer JSON
`3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, config
`e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`, revision
`8dc5f6055b90fe4b9422340810b270b9569f37f3`; identity is vocab 130560, BOS 0,
EOS/PAD 1, EOS text `</s>`.

Before any promotion decision, root must verify all 75 ordered IDs, the 43/32
denominators, exact tokenizer/native prompt IDs, canonical EOS/protocol/cap
gates, response completeness, partial streams, raw terminal IDs, and the
server's actual model/binary/offload evidence. NLL is unavailable on this
native route. A passing CUDA result remains a CUDA result; the notebook Vulkan
target still requires its own validation.
