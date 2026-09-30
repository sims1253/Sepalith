# Kaggle smoke jobs

The smoke controller exercises a private remote GPU job from submission through
verified download. It is separate from the local experiment runner. It does not
import the experiment queue, train the production model, or dispatch follow-ons.

```text
prepare -> submit once -> inspect version 1 -> collect -> verify -> archive
               |
        connection lost
               |
        inspect the same ID; never repeat push
```

## Run one smoke

Use Python 3.10+ for the controller and an authenticated Kaggle CLI 2.2.4.
Install that CLI in its own Python 3.12 environment. The CLI supplies credentials;
the uploaded source contains none. The kernel has internet disabled and attaches
no datasets. Keep state outside every Git worktree.

```sh
python3 scripts/cloud/kaggle_job.py prepare --state ~/.local/state/sepalith/kaggle-smoke-NEW
python3 scripts/cloud/kaggle_job.py submit --state ~/.local/state/sepalith/kaggle-smoke-NEW --kaggle /absolute/venv/bin/kaggle
python3 scripts/cloud/kaggle_job.py status --state ~/.local/state/sepalith/kaggle-smoke-NEW --kaggle /absolute/venv/bin/kaggle
python3 scripts/cloud/kaggle_job.py collect --state ~/.local/state/sepalith/kaggle-smoke-NEW --kaggle /absolute/venv/bin/kaggle
```

`prepare` freezes all uploaded code and metadata. `submit` checks both hashes and
requires at least 0.25 GPU hours remaining. It requests the T4 shape with a
600-second server timeout. Job code also sets a 120-second alarm. Setup and output
collection can affect quota charges; code elapsed time is not billed time.

Each job uses a unique private slug and version 1. Do not edit or repush that slug.
A filesystem lock excludes another controller using the same state directory.
It cannot exclude other state directories, notebook users or legacy scripts.
The state is flushed before submission. A timeout, CLI error or interrupted
controller leaves an uncertain submission. Inspect the recorded remote ID.
A failed status lookup does not establish that the job never started.

`collect` requires remote completion, then checks the job/source identity, output
hashes and numerical acceptance before publishing the artifact directory. A corrupt
or incomplete download can be collected again without submitting another job.
The controller never deserializes the downloaded PyTorch checkpoint.

For a failed job, retain its state and retrieve its logs through the Kaggle CLI.
Before an explicit retry, record the remote version's terminal status and failure
cause. Prepare a new directory and job ID. Do not reset `status` to `prepared`.
If the launch remains uncertain, investigate it instead of starting a replacement.

## Acceptance and environment

The fixed seed is 3407. A two-parameter FP16 regression fits `y = 2x + 1` over 128
synthetic points for 64 steps. Every loss must be finite, loss must fall by at least
100 times, and a saved/restored checkpoint must have MSE below 0.01. The receipt
reports `scientific_verdict: not_applicable`: this is a numerical and operational
smoke, not evidence to adopt a model or training method.

The recipe runs in Kaggle's supplied Python/PyTorch image. It records the actual
interpreter, versions, GPU and capability. The image is not pinned by digest, so
this is a compatibility probe, not a reproducible scientific environment contract.
A scientific remote recipe still needs pinned dependencies and immutable input
identities, plus its own evaluation and adoption gates.

The first run and archive references are in
[the validation record](validation/2026-09-06-editor-kaggle.md).

## Allocate future free compute

| Allowance | Useful next work | Prerequisite |
|---|---|---|
| GPU | One FP16-compatible model pilot; then bounded independent arms | Verify the selected model and dependency stack on the assigned card |
| TPU | A TPU-native embedding or batch-compute task | An actual queued task and a tested JAX/XLA path |
| Benchmark model credits | Compare models on existing edit cases; check blind judge agreement | Verify account credit limits, freeze cases and scoring, protect held-out data |

The earlier investigation found mixed-dtype failures for the production
Qwen3.5/Unsloth recipe on T4. This tiny smoke does not resolve those failures.
The four-arm LR sweep already completed on Anyscale; do not submit it again from
an older parked plan. The 300-step results do not justify changing production LR.
Benchmark credits fund model inference, not training hours.

A future runner adapter must keep remote IDs after the local process exits,
account for remote resources separately, and block duplicate submission after an
uncertain launch. Local process-group supervision does not manage a Kaggle job.

## Offline checks

```sh
python3 -m unittest discover -s scripts/cloud -p test_kaggle_job.py -v
```

Tests use fake CLI responses and temporary files. CI never submits a Kaggle job.
See the [official CLI contract](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
for submission timeouts and version-specific status/output commands.
