#!/usr/bin/env python3
"""Push, poll and fetch the C12 vLLM census-throughput trial kernel (``vllm_census_entry.py``).

A thin wrapper over ``infer_push.py`` (card C03): same bootstrap (GitHub
codeload at a pushed commit, code digest check, no secret in the source, the
harness pins in the system Python), same quota check (R7) and ledger line in
``~/.local/state/sepalith/kaggle/quota-ledger.jsonl``. Differences: the kernel
slug is ``sepalith-vllm-<run-id>``; the code is ``packages/sepalith/src`` plus
``scripts/kaggle/{infer_entry,vllm_census_entry}.py``; the entry installs vLLM
itself into a separate venv, so Kaggle's torch and the harness pins stay as
they are for the transformers reference; ``--upload-hf`` is refused (the trial
outputs are uploaded with the C12 card run, batched, R8).

Examples::

    # Free CPU check: inputs, vLLM install and import, no GPU quota.
    python3 scripts/kaggle/vllm_census_push.py --run-id c12-vllm-cpu-check --cpu --estimate-hours 0 \\
        --timeout-hours 1 -- --cpu-dry-run
    # The bounded GPU trial (user cap: 1 GPU-hour).
    python3 scripts/kaggle/vllm_census_push.py --run-id c12-vllm-trial --estimate-hours 0.8 --timeout-hours 0.95 --
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_push  # noqa: E402

ENTRY = "scripts/kaggle/vllm_census_entry.py"


def configure():
    infer_push.CODE_PATHS = ("packages/sepalith/src", "scripts/kaggle/infer_entry.py", ENTRY)
    old = 'str(root / "scripts/kaggle/infer_entry.py"), "--run-id"'
    if old not in infer_push.BOOTSTRAP:
        raise SystemExit("infer_push.BOOTSTRAP changed; update vllm_census_push.py")
    infer_push.BOOTSTRAP = (infer_push.BOOTSTRAP.replace(old, f'str(root / "{ENTRY}"), "--run-id"')
                            .replace('"/tmp/sepalith-inference"', '"/tmp/sepalith-vllm"')
                            .replace("Sepalith C03 inference bootstrap", "Sepalith C12 vLLM trial bootstrap"))
    infer_push.slug = lambda run_id: f"sepalith-vllm-{run_id}"[:50].rstrip("-")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--upload-hf" in argv:
        raise SystemExit("the vLLM trial does not upload to HF; C12 uploads its outputs in one batched commit")
    configure()
    return infer_push.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
