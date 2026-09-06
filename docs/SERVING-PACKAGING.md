# Serving and quant export

The extension uses a native llama-server process and OpenAI-compatible HTTP.
The packaging baseline is llama.cpp b10453, commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. Updates require a reviewed manifest
and model smoke/evaluation results. Clients do not follow nightlies.

## Quant exports

The original three positional export arguments and environment overrides still
work. The default remains Q8_0. The default converter now comes from the permanent
pinned source checkout; a missing converter fails before model loading. Explicit
`LLAMA_CONVERT` and `LLAMA_QUANT` overrides remain available for architecture forks.
Do not assume spark models work in the stock release.

Build calibration text from an assembled training split:

```sh
python3 experiments/training/build_imatrix_corpus.py \
  /mnt/h/sepalith/datasets/sft_v7/train.jsonl \
  experiments/models/quant-calibration/sft_v7.txt
```

The sampler preserves rendered special tokens, samples up to 64 rows per family,
and records the source hash, seed, family counts, and output hash. This is a
family-balanced calibration candidate; it is not a quality result. Keep evaluation
rows outside this corpus. `cal_length.py` calibrates diffusion lengths and has no
role in quantization.

Export from the banked merged B4 checkpoint, retaining the full precision GGUF:

```sh
NO_LORA=1 .venv-sft/bin/python experiments/training/export_gguf.py \
  unused /mnt/h/sepalith/runs/pft1_b4_merged packaging_b4 --keep-f16
```

Compute activations on that model. CPU flags avoid using the GPU, but this work still requires a CPU resource
window and must not overlap a quiet benchmark. Use the matching CUDA binary and `-ngl 99` only in an available GPU
window. Increase chunks for a release calibration after measuring coverage.

```sh
experiments/bin/llama/llama-b10453/llama-imatrix \
  -m experiments/models/packaging_b4-f16.gguf \
  -f experiments/models/quant-calibration/sft_v7.txt \
  -o experiments/models/quant-calibration/b4-imatrix.gguf \
  --parse-special --no-ppl --chunks 32 -c 512 -t 8 -ngl 0
python3 experiments/training/export_gguf.py unused unused packaging_b4_imatrix \
  --f16 experiments/models/packaging_b4-f16.gguf \
  --imatrix experiments/models/quant-calibration/b4-imatrix.gguf \
  --tiers Q6_K Q4_K_M IQ4_XS
python3 experiments/training/export_gguf.py unused unused packaging_b4_control \
  --f16 experiments/models/packaging_b4-f16.gguf \
  --uncalibrated --tiers Q4_K_M
```

Non-Q8 tiers default to Q8_0 output and embedding tensors. Each can be changed to
Q6_K with `--output-type` / `--embedding-type`. Both Q4 arms must use the same
policy so the comparison isolates imatrix. Q8 retains the original quantizer
behavior. Each GGUF has a JSON receipt with size, hash, quantizer path, policy,
and imatrix hash. Failed quantization removes its partial file.

Run `experiments/eval/eval_scenarios.py` on Q8 and both Q4 arms with identical
settings. Use separate stems because the harness keys result files by model stem.
The local runner `scripts/packaging/eval_quant_smoke.sh` defaults to three rows
per family; set `SEPALITH_EVAL_CAP=150` for the full scenario set. The calibration
runner `scripts/packaging/quant_smoke.sh` defaults to eight chunks and accepts
`SEPALITH_IMATRIX_CHUNKS` for a larger run. It records the calibration source
model and matrix hashes; the exporter rejects a mismatched receipt.
A small `--cap` is a smoke check; the full held-out battery and paired per-row
comparison are required before promoting a footprint tier. Include noop false
positives and latency when selecting a release model.

## Runtime distribution

Use upstream prebuilts at the fixed release and normalize them into individual
files. This avoids a client archive extractor and keeps SHA-256 and size checks
per file. The GitHub Actions workflow `runtime-bundles.yml` prepares CPU/Vulkan
Windows and Linux bundles and Metal macOS bundles. It uploads CI artifacts;
it does not publish them. Windows ARM64 currently gets CPU, Linux ARM64 gets CPU
or Vulkan, and macOS gets Metal.

```sh
python3 scripts/packaging/bundle_runtime.py linux-x64-cpu out/linux-x64-cpu \
  --base-url https://github.com/sims1253/Sepalith/releases/download/runtime-b10453-v1
python3 scripts/packaging/manifest.py \
  --bundle out/linux-x64-cpu/bundle.json \
  --model experiments/models/YOUR-VALIDATED-MODEL.gguf \
  --model-url https://huggingface.co/OWNER/MODEL/resolve/COMMIT-SHA/YOUR-VALIDATED-MODEL.gguf \
  --tokenizer-revision YOUR-IMMUTABLE-TOKENIZER-REVISION \
  --model-revision YOUR-IMMUTABLE-MODEL-REVISION \
  --output release.json
```

Repeat `--bundle` for each validated platform. The manifest pins every file by
hash and size. Hosting uses GitHub Releases for runtime files and the release manifest,
and Hugging Face for GGUFs (user choice, 2026-09-06). Upload the files in each
`release-assets/` directory as flat GitHub attachments. Their platform-prefixed
remote names avoid collisions; the downloader restores the original local names
needed by the shared libraries. Use an immutable runtime release tag and an HF
commit SHA in every model URL. SHA-256 checks pin the downloaded bytes too.

The GitHub repository is `sims1253/Sepalith`. The Hugging Face model repository,
validated model, and release tag are still to be selected; no publication has
occurred and the extension has no default manifest URL. Public downloads use
HTTPS directly and need no Python, Git LFS, or Hugging Face SDK on user machines.
GitHub release files must be under 2 GiB each; runtime files fit well within that
limit. GGUFs live on HF independently of that ceiling. See
[GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and [HF downloads](https://huggingface.co/docs/hub/models-downloading).

In VS Code, set `sepalith.manifestUrl` to the reviewed manifest URL. The extension
shares its model cache with the Zed launcher (`~/.local/share/sepalith` on
Linux/macOS, `%LOCALAPPDATA%/Sepalith` on Windows, or `SEPALITH_HOME`),
downloads on first start, verifies
cached files, and starts the selected executable on loopback. Model and server
path overrides support development. A model override works with a managed
runtime only when its SHA-256 and size match the manifest. Manifest and executable settings have application scope.

Auto selects Metal on macOS. On Windows/Linux it selects Vulkan if `vulkaninfo
--summary` reports a discrete or integrated GPU; when that tool is absent it checks Linux DRM vendor IDs or Windows video
controller inventory. These are hardware hints; successful server startup
validates the driver. If neither probe finds a GPU, it selects CPU. Users
can also explicitly select a backend. Bundle selection falls
back to CPU when the preferred bundle is absent. A managed GPU process that fails to spawn or exits retries once with the CPU
bundle. Readiness timeouts are reported without retry. Managed
CPU starts with zero GPU layers; GPU bundles request 99 layers. Manual servers
default to zero layers and expose `gpuLayers`.

Raw llama-server remains the extension spine. The existing lifecycle and HTTP
client already work, and direct bundles let converter/export/runtime pins share
one release. This choice does not depend on node-llama-cpp backend claims.

## Validation and remaining release work

The Linux CPU bundle was downloaded from upstream, normalized, and used for a
real health and completion smoke with `b2_qwen35_08b-Q8_0.gguf`. It occupies about
63 MB unpacked, including shared-library aliases and CPU variants. The extension
build and manifest/cache-integrity tests run locally. Windows/macOS/Vulkan need
native-machine smoke checks; the packaging workflow alone does not validate them.

No release is published. Before release: choose the hosted model and hosting
account, run the full quant battery, validate backend startup/fallback on target
machines, and validate the Zed integration in the editor. The first release targets VS Code
and Zed. Zed uses its built-in `zeta2` OpenAI-compatible provider plus generated
native-shell launchers; see [the Zed setup](../extensions/zed-sepalith/README.md). Runtime and GGUF files stay outside the VSIX.

Upstream reference: [b10453 release](https://github.com/ggml-org/llama.cpp/releases/tag/b10453).
The pinned checkout's `tools/imatrix/README.md` and quantizer `--help` were checked
for the flags above.

Local quant results and validation evidence: [packaging smoke](QUANT-PACKAGING-SMOKE.md).

## Validation and cache policy

Run `python3 scripts/check_product.py` with extension dependencies installed. It
uses fake downloads, small archives and tiny child processes; no real model or
network access is needed. Core contracts have a separate
`python3 scripts/check_core.py` command.

Release manifests require `modelProfile` with `renderer: "zeta2-v1"`,
`task: "r-next-edit"`, and nonempty `tokenizerRevision` and `modelRevision`
identifiers. The release author must supply immutable revisions. Hashes pin the
distributed model bytes; a profile declaration does not itself demonstrate edit
quality. Python manifest/launcher tools and the TypeScript consumer validate
shared fixtures before release outputs are used.

The editor stores a validated manifest per URL and revalidates it on use. Normal
starts use that manifest and verified cached files, enabling offline restart.
**Sepalith: Refresh runtime manifest** explicitly fetches a replacement; failures
preserve the old cache. Restart the owned server to apply a refreshed manifest.
A managed model override must match its declared hash and size. Different local
models require explicit manual server configuration.

Owned server shutdown and readiness-timeout cleanup share TERM/KILL escalation
and wait for exit. They never signal a process merely because it answers the
configured port. Release readiness still requires real editor, model-quality and
target-platform checks; offline fixtures do not substitute for them.
