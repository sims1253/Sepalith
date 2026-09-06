# Bounded learned module-memory gate

This directory implements the first learnability gate from
[the proposal](../../docs/LATENT-WORKSPACE-MEMORY.md). It trains continuous module
slots and decoder adapters. It does not change the product prompt or train an
editor. See [RESULTS.md](RESULTS.md) for what has actually run.

## Frozen experiment

[recipe.json](recipe.json) is the preregistration. Changing it requires a new ID.
The synthetic generator assigns 512 training and 160 evaluation packages to
separate repositories **before** generating source or using a tokenizer. Each
package has four random function identifiers and four different random defaults
from 10–99. One query per package asks for a default. Package IDs never enter the
model input. The evaluation unit is an independent synthetic package, not an edit.

The three trained arms receive identical targets, training order, 128 optimizer
steps, effective batch eight (serial accumulation), and seed 1273. Each sees
1,024 examples and 3,072 scored decoder tokens, including EOS. The decoder
initialization is the banked B4 merge in `/mnt/h/sepalith/runs/pft1_b4_merged`.

| Condition | Input and training |
|---|---|
| Base | Banked decoder, query only; no additional training |
| Local | Query only; mixer LoRA training |
| Exact retrieval | The same selected module as raw source, then query; mixer LoRA training |
| Latent | Learned module slots, then query; encoder and mixer LoRA training |
| Absent | Trained latent decoder, with its slots removed |
| Shuffled | Trained latent system, with the next evaluation package's module in a fixed cyclic derangement |

All generation uses greedy decoding, at most four new tokens, and the pinned EOS.
Score the decoded integer after stripping surrounding whitespace. Record exact
field recovery, first generated token recovery, target-only CE, and raw IDs.
The scorer receives the target; the inference context builder receives an
`InputView` with no target or future-edit fields.

**Pass:** latent exact recovery exceeds local by at least 0.10, and relevant
memory exceeds shuffled by at least 0.10. Both paired percentile bootstrap 95%
intervals must have a lower bound above zero (10,000 package resamples, seed
55173). No seed or capacity search follows automatically. Absent and retrieval
scores are reported even though they are not extra pass thresholds. A failed or
incomplete gate blocks expansion; it does not disprove all latent architectures.

**Total run caps:** 5,400 wall seconds for the foreground worker (including load,
weight verification, base evaluation, all three arms, and final evaluation);
20,000 scored training tokens; 1,000,000 decoder training positions; 1,100,000
encoder input bytes; 2e16 estimated training FLOPs; 27e9 allocated VRAM bytes.
The allocator receives the VRAM limit before model load. Any failed batch stops
the run. The parent kills its own worker at the wall limit. There is no paid
infrastructure, confirmation seed, edit training, or automatic retry.

`6 × parameters × positions` is the ledger's conservative parameter-matmul
**proxy**, including the frozen decoder and the encoder. It is not a measured
hardware FLOP count and omits explicit attention/elementwise accounting. Training
arms match target exposure, not total FLOPs. A second FLOP-matched retrieval run
belongs to a later preregistration; this bounded screen does not make an economic
or superiority claim over retrieval.

## Architecture and justified deviations

[decoder-inspection.json](decoder-inspection.json) records the local config,
backend hashes, all runtime target names, saved tensor names, shapes, and parameter
counts. Inspection uses the actual safetensors header and a meta-device model;
it does not claim a numerical pretrained-model test.

- Decoder: Transformers 5.5.0 `Qwen3_5ForCausalLM`, 1,881,825,088 parameters,
  width 2,048, 24 layers: 18 Gated DeltaNet and six full-attention layers.
- Saved tensors begin with `model.language_model`; the pinned Transformers
  conversion maps that prefix to runtime `model`. Loading rejects missing,
  unexpected, or mismatched tensors. Config/weights/tokenizer/backend hashes enter
  the run identity. A directory name alone does not authorize training.
- Adaptation: rank 16, alpha 32, dropout 0.05, **114 exact targets / 7,382,016
  trainable parameters**. Each GDN layer adapts `in_proj_qkv`, `in_proj_z`,
  `in_proj_b`, `in_proj_a`, `out_proj`; each full-attention layer adapts
  `q_proj`, `k_proj`, `v_proj`, `o_proj`. The GDN targets include `linear_attn`,
  which a generic attention regex can miss. MLPs, convolution, norms, token
  embeddings and LM head stay frozen. Forward hooks and gradients verify every
  target on the first actual training step.
- Gate encoder: two bidirectional Transformer layers, width 128, four heads,
  FFN width 512, learned byte positions, dropout 0.1. Thirty-two learned pooling
  queries cross-attend to the module, followed by residual FFN and normalization,
  projection to 2,048 and a learned channel scale. Training-only calibration
  matches the initial projected RMS to decoder input embeddings.
- This deliberately reduces the proposal's six-layer width-512 encoder for a
  cheap wiring/learnability screen. UTF-8 bytes map to IDs 1–256; zero is padding.
  No encoder tokenizer is trained. Thus no holdout content can influence tokenizer
  fitting. Maximum input is 1,024 bytes; oversized modules fail, rather than lose
  facts through unreported truncation. The frozen generated modules use 134 bytes
  including their path. Masks record omitted ranges before encoding.
- This gate uses **one module and default-value QA only**, not four modules or a
  reconstruction/edit mixture. Random defaults make memorized repository facts
  unhelpful. Exact retrieval uses the same full module (at most 66 actual decoder
  tokens), within its 128-token allowance. Latents use 32 positions; no arm adds
  pretend padding evidence. This is not an equal-realized-position comparison.
- A module's 32 bf16 slots occupy 131,072 payload bytes. The gate tests a learned
  information bottleneck, not demonstrated byte compression: its source is much
  smaller than that payload. Whole-workspace pooling and editing remain later work.
- Scientific reference: eager full attention and the unpatched PyTorch GDN path.
  Fused GDN/norm installations are rejected by the backend pin. The inspected GDN
  implementation converts recurrence to fp32; model operations use bf16 on GPU,
  while encoder, adapter master parameters, CE, and AdamW state use fp32. This
  trades speed for an inspectable input/gradient path. No Unsloth monkeypatches.

## Positions, loss, and artifacts

Sequences are `[module slots][query][preceding target tokens]`. Explicit 2-D
positions run consecutively from zero; the text model expands these into its
four position planes (text plus three rotary planes). Encoder positions restart
per module. Ordinary causal attention applies in the decoder. No decoder padding
or position reset occurs at the boundary.

`target_loss` sends only the hidden positions that predict target tokens to the
LM head. It scores the first target from the final query position, then scores
remaining target/EOS tokens. Prompt and memory positions receive no direct loss;
gradients can still flow through them. A contract test compares this result with
full logits and explicitly shifted `-100` labels.

**Cache reuse is disabled**, including during greedy decoding. Every forward uses
`use_cache=False`, and the entry point rejects supplied state. GDN has convolution
and recurrent state in addition to full-attention KV; no hybrid restore or KV
portability is claimed. Generated tokens trigger a full prefix replay.

Projected payloads use safetensors and the core `LatentMemoryManifest`. Identity
includes encoder weights/config, byte-tokenizer convention, adapted decoder and
backend identity, decoder tokenizer, dtype, shape, source scope and layout.
Consumption checks current scoped source hashes, reads the payload once, hashes
those exact bytes, deserializes those bytes, then checks tensor keys, dimensions,
dtype and finiteness. Changed, added or removed sources invalidate memory.
Training never reuses detached encoder outputs. Evaluation saves and validates
every consumed payload, including shuffled donors.

`git_parent_view` reads explicit blobs from an edit commit's first parent. It
never reads the edited checkout. The reviewed allowlist excludes generated and
unavailable sibling artifacts. Masked spans are removed before encoding, and
copies containing a removed span are rejected. The synthetic gate does not mine
external repositories or post-edit data.

Each run produces:

- `freeze.json`, dataset/split/source/backend/checkpoint hashes and a tokenizer
  budget audit in the preparation directory;
- `real-decoder-contract.json`, requiring token IDs versus equivalent embeddings
  within atol=rtol=0.02 in bf16, plus identical three-token greedy continuation;
- `adaptation.json`, exact training order, checkpoints every 32 steps, final
  encoder/adapter weights, optimizer/RNG state and hashes for each trained arm;
- `predictions.jsonl`: all paired predictions, targets, source/memory manifests,
  donor IDs, realized positions, first-token checks, CE and per-forward timings;
- `ledger.jsonl`: hardware, versions, code identity, calibration, per-example
  training counts, losses, optimizer steps, gradient attachment, evaluation
  forward counts, failed batches and cumulative compute;
- `result.json` and `supervisor.json`: scientific verdict or explicit incomplete
  status, wall cap, and launch/end times. Timings include this research path's
  overhead and are **not quiet serving benchmarks**.

`restore_checkpoint` validates hashes and loads weights into an identically
configured model/encoder. Optimizer state is retained for a separately declared
resume recipe; this experiment never resumes automatically.

## Reproduce and schedule

Use Python 3.10 and a separate environment. No root/core environment changes:

```sh
uv venv /absolute/env --python /usr/bin/python3.10
uv pip sync --python /absolute/env/bin/python experiments/latent_memory/requirements.lock
PYTHONPATH=packages/sepalith/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  taskset -c 23 /absolute/env/bin/python -m unittest experiments.latent_memory.test_contracts -v
```

CPU preparation is executable now. Select a new output directory:

```sh
PYTHONPATH=packages/sepalith/src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  taskset -c 23 /absolute/env/bin/python -m experiments.latent_memory.cli prepare \
  --checkpoint /mnt/h/sepalith/runs/pft1_b4_merged --output /absolute/preparation
```

This freezes data, recipe, tokenizer and inspected implementation. It deliberately
marks `weights_pinned: false`: reading and hashing the 3.76-GB weight file waits
for the resource window. `run` rejects this preparation artifact. There is no
learnability result at this stage.

Once the current GPU owner and quiet benchmark owner release a complete window,
record the operational claim in the shared ledger under the current coordinator's
protocol. The user's task already authorizes this bounded gate. Create a window
record containing actual release/claim evidence and an expiry timestamp:

```json
{
  "owner": "codex-latent-memory",
  "claim_record": "actual shared ledger claim text",
  "previous_owner_release": "actual prior-owner release text",
  "quiet_benchmarks_released": true,
  "expires_at": 0
}
```

Replace `0` with a future Unix timestamp covering the full 5,400 seconds. The
record is an operator attestation, not a cross-agent lock. A fresh `nvidia-smi`
compute-process check must also pass. WSL can omit compute PIDs, so this does not
replace the owner releases or quiet-window inventory. Do not use a temporary
utilization dip as a release.

In that window, rerun `prepare` to a **new** directory with `--pin-weights --window
/absolute/window.json`. It pins the actual weight bytes before any training.
Keep the resulting freeze immutable. Then run once:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  taskset -c 16-23 /absolute/env/bin/python experiments/latent_memory/entry.py run \
  --frozen /absolute/pinned-preparation --output /absolute/run \
  --window /absolute/window.json
```

If using the shared runner after operational cutover, snapshot this entire
research directory plus `packages/sepalith/src/sepalith/{__init__,memory}.py`.
Use `make_runner_recipe` with `--snapshot`, `--frozen`, `--window`, `--python`, and
`--output` to create the concrete immutable recipe. The helper never enqueues or
resumes a queue. It must run in the reserved window because it verifies and hashes
weights. The runner recipe declares data, weights and window inputs and all main
result/checkpoint artifacts. Respect the existing authoritative runner state;
creating another queue does not acquire resources. A runner success can accompany
a scientific **FAIL**; both are retained.
