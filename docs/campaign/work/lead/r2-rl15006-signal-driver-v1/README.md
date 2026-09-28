# RL-11 15,006-row signal pilot driver

This packet prepares an inference-only TRAIN pilot. It scans and rehashes the
complete 15,006-row admitted pool, then generates four candidates for each of
the fixed 112 prompt groups. It performs zero optimizer updates. Each group is
published atomically only after all generated-ID hashes match the reward-ID
hashes. Resume accepts only a contiguous prefix of complete, identity-matching
groups.

`driver-config.template.json` is intentionally unrunnable. Root must bind an
accepted merged model and a fresh absolute output path. `bind_signal_driver.py`
requires root's expected model-manifest, merged-weight aggregate, and tokenizer
hashes and then rehashes every non-hidden model file. A later byte change or an
unlisted file fails preflight.

CPU preflight:

```sh
nice -n 10 env PYTHONDONTWRITEBYTECODE=1 python3 \
  source/experiments/training/signal_pilot_driver.py preflight \
  --config driver-config.template.json
```

After root selection, from this packet directory:

```sh
nice -n 10 env PYTHONDONTWRITEBYTECODE=1 python3 \
  source/experiments/training/bind_signal_driver.py \
  --template driver-config.template.json \
  --model-path MODEL_PATH --model-manifest MODEL_MANIFEST \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --expected-merged-weights-sha256 MERGED_WEIGHTS_SHA256 \
  --expected-tokenizer-json-sha256 TOKENIZER_JSON_SHA256 \
  --pilot-output ABSOLUTE_FRESH_OUTPUT --output root-bound-config.json

CUDA_VISIBLE_DEVICES=LEASED_DEVICE nice -n 10 env PYTHONDONTWRITEBYTECODE=1 \
  python3 source/experiments/training/signal_pilot_driver.py run \
  --config root-bound-config.json
```

The live path uses the pinned Unsloth `FastLanguageModel` API and the copied
`CampaignPRM03Reward`. Verified framed fragments use only their admitted
diagnostic suffix. Unverified syntax rows never call R. Parser service errors
create an infrastructure-failure record and leave the group uncommitted.

The live GPU/model load has not been run in this CPU-only preparation. Exact
VRAM use and throughput remain unknown until root selects the model and admits
a GPU lease. The R 4.6.1 parser identity is inherited from the admitted reward
buffer, but this packet did not execute live candidate parses. A failed parser
record is not automatically overwritten on resume; root must inspect it before
choosing a fresh output or an explicit repair.
