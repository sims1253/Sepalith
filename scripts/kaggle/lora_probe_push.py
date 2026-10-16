#!/usr/bin/env python3
"""Push, poll and fetch the LoRA-on-T4 probe kernel (``lora_probe_entry.py``).

A thin wrapper over ``infer_push.py`` (card C03): same bootstrap (GitHub
codeload at a pushed commit, code digest check, no secret in the source), same
quota check (R7) and ledger line in
``~/.local/state/sepalith/kaggle/quota-ledger.jsonl``. Differences: the kernel
slug is ``sepalith-lora-<run-id>``, the code is ``scripts/kaggle/{infer_entry,
lora_probe_entry}.py`` only, peft 0.20.0 (the PC pin) is installed next to the
C03 pins, and ``--upload-hf`` is refused (probe outputs stay local).

Example::

    python3 scripts/kaggle/lora_probe_push.py --run-id lora-probe-20261016 --estimate-hours 0.75 \\
        --timeout-hours 0.9 -- --phases r16+r64,r16-gc2048+q4-r16,ddp:r16 --updates 40 --ddp-updates 30
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_push  # noqa: E402

ENTRY = "scripts/kaggle/lora_probe_entry.py"


def configure():
    infer_push.CODE_PATHS = ("scripts/kaggle/infer_entry.py", ENTRY)
    infer_push.PIP_PINS = (*infer_push.PIP_PINS, "peft==0.20.0")
    old = 'str(root / "scripts/kaggle/infer_entry.py"), "--run-id"'
    if old not in infer_push.BOOTSTRAP:
        raise SystemExit("infer_push.BOOTSTRAP changed; update lora_probe_push.py")
    infer_push.BOOTSTRAP = (infer_push.BOOTSTRAP.replace(old, f'str(root / "{ENTRY}"), "--run-id"')
                            .replace('"/tmp/sepalith-inference"', '"/tmp/sepalith-lora"')
                            .replace("Sepalith C03 inference bootstrap", "Sepalith LoRA probe bootstrap"))
    infer_push.slug = lambda run_id: f"sepalith-lora-{run_id}"[:50].rstrip("-")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--upload-hf" in argv:
        raise SystemExit("the LoRA probe does not upload to HF")
    configure()
    return infer_push.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
