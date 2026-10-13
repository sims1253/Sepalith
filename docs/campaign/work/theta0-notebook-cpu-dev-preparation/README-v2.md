# Theta0 Q8 CPU DEV75 comparator, v2

This v2 packet corrects the earlier preparation. The matched quality contract is a 4096-token server context and the pinned scorer's fixed 512-token completion cap. The notebook runs only the pinned AVX2 CPU `llama-server` (`-ngl 0`); the planning host runs the unchanged corrected DEV client, panel, HF tokenizer, and scorer over a loopback SSH forward.

Root runs this from the planning host after the remote binary and model admission. It does not run on the notebook:

~~~sh
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
RUN=$PLAN/docs/campaign/work/theta0-notebook-cpu-dev-preparation/run_theta0_notebook_cpu_dev_v2.py
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
LOCAL_RUN=/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-cpu-dev-v2
/usr/bin/timeout --foreground --signal=TERM --kill-after=20s 1200s \
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -B "$RUN" --run-root "$LOCAL_RUN"
~~~

The script derives the fresh remote run directory as `/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/RUN-09-theta0-q8-cpu-dev-v2` when the local run directory has that basename. Use a unique basename for retries. The remote helper stages under the matching `.staging` sibling, refuses collisions, verifies the runtime identity's dependency inventory plus the Q8 model hash, and records `launch.json`, `live-device-audit.json`, `server.log`, and `terminal.json` remotely. The local run records `preflight.json`, `remote-ready.json`, `client.log`, `quality.json`, and `terminal.json`.

The local client command is assembled with these explicit planning-host paths:

- client: `docs/campaign/work/theta0-q8-cuda-dev-preparation/run09_corrected_dev_client.py`
- panel: `docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl` (75 rows)
- tokenizer: `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain`
- execution root: `/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912`
- Python: `/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python`

The client deliberately has no `--cap` option. `run09_native_dev_quality.py` supplies the pinned `COMPLETION_CAP = 512`, so runtime settings cannot select a gold operation or family. The server profile is `4096/512`, threads `6`, batch and ubatch `256`, HTTP threads `2`, parallel `1`, and `ngl=0`. Local scorer/tokenizer environment uses offline HF mode and empty CUDA visibility; remote environment clears `GGML_*` and `SEPALITH_VK_TRACE`, sets the CPU thread profile, and requires no Vulkan DRI device.

The v2 tests are static and launch-free:

~~~sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  "$PLAN/docs/campaign/work/theta0-notebook-cpu-dev-preparation/test_run_theta0_notebook_cpu_dev_v2.py"
~~~

This packet proves command construction, identity/path binding, cap and placement, CPU environment cleanup, and import side-effect policy. It does not prove remote availability, model loading, server latency, scorer output, editor application, or quality parity. The earlier v1 files remain preserved for audit; v2 is the corrected preparation.
