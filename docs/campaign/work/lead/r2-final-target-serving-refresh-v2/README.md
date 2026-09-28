# Final-target serving refresh v2

This fresh packet preserves v1 and remains target-null, launch-disabled, and TRAIN-only. Root must select one sealed dense full checkpoint and create a separate selection receipt, binding, and admission before any export or serving action.

The binding validates every file in the full checkpoint manifest, including the exact dense `model.safetensors`, config, tokenizer companions, checkpoint identity, and root selection. It also verifies a 111-file transitive converter source closure at the pinned llama.cpp commit. F16 and each quant require `audit_gguf_norms.py`: it reads all 85 saved norm tensors without loading the model, requires the reviewed MiniCPM norm mapping, and proves every mapped GGUF norm remains F32 and bit-exact. A downstream stage requires the preceding norm receipt.

Two disjoint TRAIN-only panels implement the frozen SPEC geometry. N-gram uses 20 2K and 20 8K prompts once. Draft pairing uses fresh 12 2K and 12 8K prompts twice. Both use context 10240, cap 64, greedy decoding, one request slot, and no authored target tail. The longer prompts come from full-file TRAIN context candidates whose training admission remains unclaimed; use here is evaluation only.

`analyze_refresh_v2.py` derives the exact request key set from the panel manifest and rejects missing, duplicate, extra, nonaccepted, nonfinite-latency, or unbound results. Cap hits use explicit final stop evidence. Candidate promotion requires exact token/text/termination parity, complete draft counters, a deterministic paired-bootstrap 95% lower speedup bound of at least 1.4x, and candidate p95 no worse than baseline. A valid analysis may return `no_winner`.

After root creates an admitted binding, use fresh absolute output paths:

```sh
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-target-serving-refresh-v2
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage export --out "$RUN/export-commands.json"
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage audit-f16 --out "$RUN/audit-f16-command.json"
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage imatrix --out "$RUN/imatrix-command.json"
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage quantize --out "$RUN/quantize-commands.json"
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage audit-quants --out "$RUN/audit-quant-commands.json"
$PY -B "$P/prepare_refresh_v2.py" --binding "$BINDING" --stage serving --out "$RUN/serving-commands.json"
```

The server commands remain launch inputs for the root-owned CUDA lifecycle wrapper. This packet does not start a server and does not by itself supply PID, port, watchdog, health, telemetry, or cleanup evidence. Root must use fresh isolated run directories and the existing guarded server lifecycle. Imatrix semantic completion still needs a root execution receipt proving exit zero, 64 chunks, and no non-embedding coverage mismatch before quant admission; v2 validates calibration inputs but does not parse a future imatrix log. This is the one explicit unfinished execution gate.
