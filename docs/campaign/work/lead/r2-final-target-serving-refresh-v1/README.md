# Final-target serving refresh preparation

This packet is an executable, fail-closed refresh path for a future selected
full-weight target. `refresh-template.json` deliberately has `target: null`
and `launch_authorized: false`; it cannot emit a runnable command until root
copies it to a fresh binding, supplies a selected full-weight target manifest,
artifact paths and hashes, and records launch authorization.

The calibration input is the existing 84-row TRAIN-only imatrix corpus (seven
families, 80 packages, 64 chunks). The quality and latency input is the
existing 40-row TRAIN-only panel with one cold and one warm request per row.
Neither contains DEV or final data. The native token contract is BOS 0,
PAD/EOS 1, native EOG `[1,130073]`, vocabulary 130560. The panel uses context
4096 and cap 192 to preserve direct serving-screen comparability; every result
reports cap hits. This cap is a measurement profile and does not redefine the
training or model output limit.

After root fills `binding.root.json`, run each preparation step before its
separately guarded GPU/CPU action:

```sh
PACKET=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-target-serving-refresh-v1
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
$PYTHON -B "$PACKET/prepare_refresh.py" --binding binding.root.json --stage export --out commands.export.json
$PYTHON -B "$PACKET/prepare_refresh.py" --binding binding.root.json --stage imatrix --out commands.imatrix.json
$PYTHON -B "$PACKET/prepare_refresh.py" --binding binding.root.json --stage quantize --out commands.quantize.json
$PYTHON -B "$PACKET/prepare_refresh.py" --binding binding.root.json --stage probe --out commands.quant-quality.json
$PYTHON -B "$PACKET/prepare_refresh.py" --binding binding.root.json --stage serving --out commands.serving.json
```

The emitted sequence is HF-to-F16 conversion, the pinned b10453 imatrix run,
and Q8_0/Q4_K_M/IQ3_M quantization with the same imatrix. It refuses existing
output paths at creation stages and hashes every input at each later stage.
`probe_bound.py` hashes the exact target quant and optional draft before it
delegates requests to the reviewed panel client. It adds target, tokenizer,
quant, draft and binding identities to each result without changing HTTP,
rendering, EOS, timeout or parser behavior.

Run `analyze_refresh.py --binding binding.root.json` with the F16 ordinary result as the quant baseline and
each quant result as a candidate. For speculative serving, use the Q8 ordinary
result as baseline and the Q8 ngram/released/existing-trained results as
candidates, adding `--require-draft-counters` for the speculative comparison. The analyzer requires the exact root binding hash, target, tokenizer, quant and draft hashes, exact
returned token IDs and raw text, accepted protocol results, and paired request
keys. It reports p50/p95 end-to-end latency, cap hits and server-reported draft
acceptance. Missing draft counters remain `missing`; they are never treated as
zero. The existing trained DSpark was trained for the old step-500 target and
is explicitly a cross-target trial. It cannot be described as target-matched.
The released draft is also hash/header bound. No native MTP path is asserted.

This packet prepares commands only. Root must run servers sequentially under
the existing CUDA lease/watchdog, retain `/health`, `/props`, logs and cleanup
evidence, and make any selection or promotion decision. No model, quant,
draft, or latency claim exists while the target is null.
