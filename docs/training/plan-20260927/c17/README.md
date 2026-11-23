# Run the C17 CPU checks and notebook benchmark

Use Python with `sepalith` on `PYTHONPATH`. The parity check also needs `gguf`,
`jinja2`, and Node with native TypeScript support. On the PC, the main checkout's
`.venv/bin/python` has these Python dependencies. CPU work uses `nice -n 10`
and at most eight threads.

From the repository root, run:

```bash
nice -n 10 env OMP_NUM_THREADS=8 PYTHONPATH=packages/sepalith/src \
  /home/m0hawk/sandbox/Sepalith/.venv/bin/python scripts/packaging/release_parity.py \
  --gguf /mnt/e/sepalith/campaign-20260915/quant-work/c19-100/c19-100-imat-Q6_K.gguf \
  --gguf /mnt/e/sepalith/campaign-20260915/quant-work/c19-100/c19-100-mixb-e4-Q4_K_M.gguf \
  --output /tmp/C17-serving-parity.json
PYTHONPATH=packages/sepalith/src:scripts/packaging \
  python -m unittest scripts/packaging/test_release_prep.py -v
```

The parity command reads the pinned PR #89 source from git into a temporary
directory. Fetch `origin/feat/extension-prm03` if the pinned object is absent.
It executes the TypeScript renderer against 224 golden contexts and eight
synthetic contexts. It reads GGUF metadata without inference. It reports the
chat-template mismatch, since editing uses the raw completion path.

## Run on the notebook after access is granted

Copy only the reviewed code and the two approved GGUFs to the notebook.
Verify the model SHA256 hashes against `receipts/C17-quant-evidence.json`.
Pin llama.cpp `3cb7ffb1a` for both CPU and Vulkan builds, matching the quant gate. Record compiler,
OS, CPU, RAM, driver and power mode, binary hashes, and server logs beside the
receipts. Plug in power, close other workloads, and use the same power mode
for all four configurations.

Start a local server for Q6_K on CPU:

```bash
/path/to/cpu/llama-server -m /path/to/c19-100-imat-Q6_K.gguf \
  --host 127.0.0.1 --port 18099 -c 8192 -t 8 -np 1 -ngl 0
```

In another terminal, run the benchmark with the verified values:

```bash
PYTHONPATH=packages/sepalith/src python scripts/packaging/notebook_latency.py \
  --backend cpu --tier Q6_K --runtime-commit VERIFIED_COMMIT \
  --model-sha256 VERIFIED_SHA256 \
  --server-command 'COPY THE EXACT SERVER COMMAND HERE' \
  --output /path/to/results/Q6_K-cpu.json
```

Stop that server. Repeat for mixB-e4 with `--tier Q4_K_M-mixB-e4`.
Then repeat both with the Vulkan build, `-ngl 99`, and `--backend vulkan`.
Confirm Vulkan offload in each log; a silent CPU fallback fails the run.
The script records the supplied identity and backend labels, so verify them
against the files and logs. It does not inspect the remote server process.

Expect 40 measured requests per configuration, after eight excluded warm-ups.
TTFT ends at the first nonempty content event. Tokens/s is llama-server's own
`predicted_per_second`, which excludes the first token sampled during prefill;
samples with fewer than two predicted tokens are rejected.
The benchmark uses uncached prompts to make the measurement reproducible;
it does not claim to measure PR #89's warm-cache editor experience.
No output quality score is inferred from these synthetic prompts.
At `3cb7ffb1a` the default log verbosity hides the device and offload lines.
`c17/notebook/run_config.sh <tier> <backend>` runs one measured configuration
with the exact command above, and also samples iGPU sysfs. Then
`c17/notebook/probe_config.sh <tier> <backend>` restarts the same command with
`-lv 4`, without measuring, to capture those lines. `env_capture.py` writes
`env.json`. On the PC, run
`python scripts/packaging/latency_receipt.py --result R.json --logs c17/latency-logs --output receipts/C17-latency-<tier>-<backend>.json`.
It turns each result into a receipt and checks the device evidence.
Keep the four JSON receipts and server logs. Present p50 and p95 TTFT, decode
rate and wall time to the user before release acceptance.
