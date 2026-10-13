# R2 draft provider payload v5 (unarmed preparation)

This packet is a fresh v5 root-bound entry wrapper for the stable
`r2-draft-cloud-profile-v2` and `r2-draft-target-runtime-v3` packets. It does
not submit a provider job during preparation. A real run requires a fresh
binding, a locally staged payload, and an independently armed root watchdog.

The binding uses schema 2 and must set the exact image
`docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9`,
instance `g5.2xlarge`, and `max_retries: 0`. It must contain the watchdog
armed timestamp, receipt SHA, absolute deadline, private repository and a
fresh prefix ending in the run UUID. The entry never obtains or writes a
watchdog receipt itself.

The staged input map is explicit. `root` contains the candidate and
uv-resolved requirements files, the complete portable source root, a target
manifest plus its five pinned canonical model files, the exact TRAIN JSONL, the
released draft safetensors file, and its sibling `config.json`. The binding
pins the manifest hash, TRAIN hash, target five-file hashes, public draft
weight hash and size, public sibling config hash, target candidate merged hash,
and DeepSpec revision. The entry checks the profile source inventory and every
listed source hash before it invokes a model loader. It does not discover
model or TRAIN files.

The Ray image is Python 3.11, so the entry does not pretend that a managed
interpreter is pre-staged. During setup it installs the pinned `uv==0.11.23`
bootstrap tool into a fresh run directory, runs `uv python install 3.10.19`,
then discovers it with `uv python find --managed-python --no-python-downloads
--no-project --resolve-links 3.10.19`. It probes the resolved CPython,
requires `Python.h` under the run-owned install, creates a run-owned venv,
and installs the resolved requirements with that venv. Every uv cache and
Python install path is explicit and excluded from durable upload.

The payload does not carry the multi-gigabyte model files. After the venv is
ready, a separate `download_staging.py` child downloads the private target
from commit `257487b64044600fee8c27cbcb1ceb19e424a9ac` under the immutable
`r2-draft-target/b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4`
prefix, and downloads the released public draft from revision
`114a20fdbf53220712c7fbdd7dccddbf1dedebb4`. It writes into fresh run-owned
directories, verifies every target/public hash, and records only repository,
revision, filename, size, and hash metadata. The private token is supplied to
that child through `HF_TOKEN`, scrubbed when it exits, and never enters a
receipt. The absolute workstation path in the shipped target manifest is
relocated to the fresh run-owned `target-model` directory as an absolute
path in the derived runtime manifest (the consumer resolves model paths from
its own CWD);
all model identity hashes remain unchanged.

The candidate requirements file is retained for top-level closure checking;
the resolved file is used for installation. The canonical closure contains
`torch==2.11.0`, `transformers==5.5.0`, `safetensors`, `tensorboard`, and
`prettytable`, among other pinned packages. `matplotlib` is intentionally
absent and the runtime import probe verifies the actual required imports after
installation.

The 7,200-second watchdog window is allocated as follows:

| Phase | Maximum |
| --- | ---: |
| managed runtime, source checks, dependency install, private sentinel | 1,800 s |
| profile-v2 (eight rows) | 3,600 s |
| durable success or failure upload | 600 s |
| root final-refresh reserve | 900 s |
| watchdog cleanup tail | 300 s |

The first three phases may consume at most 6,000 seconds. The entry computes
that work deadline as the earlier of the provider deadline and the absolute
watchdog deadline minus the 900-second final-refresh reserve and 300-second
cleanup tail. Profile-v2's 3,600-second limit is enforced separately; it does
not authorize spending the complete watchdog window.

The child training environment removes credential variables, leaves `HOME`
unchanged, and places `TMPDIR`, `HF_HOME`, `XDG_CACHE_HOME`, `TORCH_HOME`, and
`TRITON_CACHE_DIR` below the profile run. The raw target cache and all transient
cache roots remain outside the upload root.

After the private sentinel succeeds, the entry runs:

```text
<managed-python> docs/campaign/work/r2-draft-cloud-profile-v2/profile_orchestrator.py \
  --execute --run-dir <fresh-profile-run> \
  --target-model-manifest <staged-target-manifest.json> \
  --train-data-path <staged-TRAIN.jsonl> \
  --source-root <staged-source-root>
```

`stage_durable.py` copies hash-bound success outputs, checkpoints, stage
receipts, teacher metadata, and the profile persistence manifest into a fresh
`durable-upload` directory. On a useful partial failure it copies available
checkpoints and failure receipts from approved roots. It excludes the raw
cache, temporary roots, and framework caches in both cases. `artifact_upload.py`
then offers a private sentinel operation and separate `success` or `failure`
operation. The final uploader receives only `durable-upload`; provider and root
readback remain independent gates.

The packet was CPU-tested with fourteen tests, including the relocated-manifest consumer check. The tests use synthetic bindings,
small temporary receipts/checkpoints, a fake private repository client, and
hash mismatch failures. They do not access a provider, network, model weight,
or TRAIN content.

A separate static-input gate streamed the staged TRAIN file only to verify its pinned SHA; it did not parse TRAIN rows or read model payloads.

Run the bounded tests with:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -m unittest discover \
  -s . -p 'test_cloud_entry.py' -v
```

Root still owns staging the private target and released draft, the final
binding and watchdog, dependency admission, real CUDA profile execution,
private upload credentials, independent readback, final refresh, and the
sealed-final decision. The original control entry remains unchanged.

## v5 preparation boundary

Preparation ID: 6cbc8d41ab934c34acf38cb6a3d59ddb. The source packet is unarmed and unsubmitted. The original absolute deadline is `2026-09-14T04:05:17.229855Z`; the derived work deadline before the 900-second final-refresh reserve and 300-second watchdog tail is `2026-09-14T03:45:17.229855Z`. Root must bind a fresh UUID, watchdog receipt, deadline, target-manifest hash, and private credentials at admission. No inherited binding, submission, or watchdog receipt is included in this v5 payload.
