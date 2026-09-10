#!/usr/bin/env python3
"""Kaggle no-bf16 bisect: three 10-step probes for the GDN dtype crash.

Background: docs/research/2026-09-06-kaggle-compute-integration.md
§gpu-blocker. The b4 GDN recipe dies on Kaggle T4 (no bf16) at q_proj with
"expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 !=
c10::Half". Prime suspect: unsloth_zoo's "Will smartly offload gradients
to save VRAM!" patch (unsloth_zoo/gradient_checkpointing.py — fires only
on low-VRAM 16GB cards; never on the 5090/A10G), casting LoRA/base leaves
to half while GDN tensors stay bf16.

Each probe runs the standard 10-step flow (train_sft.py with STEPS=10,
env inherited from the kernel bootstrap incl. SFT_FP16=1) in a FRESH
python subprocess, so unsloth patch state cannot leak between probes:

  P1 offload_disable   — neuter unsloth's smart gradient-checkpoint
                         offload to plain torch.utils.checkpoint
                         (use_reentrant=True) BEFORE model load.
  P2 float_post_attach — standard flow, then model.float() + .cuda()
                         after get_peft_model, before training.
  P3 unsloth_upgrade   — pip install -U 'unsloth>=2026.9.2' unsloth_zoo
                         first, then the standard flow unchanged.

HELP — wall/GPU time, launch, decision rule:

  WALL / GPU TIME (T4x2 batch session; quota bills session wall-clock):
    staging (pins + data + audit, proven) ......... ~4 min
    per probe (base pull + 10-step train) ......... ~10-12 min
    total BISECT=1 kernel ......................... ~35-40 min (~0.6 GPU-h)

  LAUNCH: BISECT=1 in the kernel bootstrap env; kaggle_sft_entry.sh runs
    this script INSTEAD of the single train flow (same staging/pins/audit
    first). Knobs: BISECT_STEPS (default 10), BISECT_PROBES (comma subset
    of offload_disable,float_post_attach,unsloth_upgrade — default all),
    BISECT_RESULT (default /kaggle/working/bisect_result.json),
    BISECT_TIMEOUT (per-probe seconds, default 2700).

  DECISION RULE (pre-registered):
    any probe passing => rerun THAT probe at STEPS=300 as the
      confirmation arm (BISECT_STEPS=300 plus BISECT_PROBES=<winner>, or
      promote the winning shim to the single train flow), then
      eval_loss@300 + the standard noopFP/format battery before any
      recipe change.
    all probes crashing => door closed: append the signatures to the
      research doc §gpu-blocker; Kaggle GPU stays fp16-safe standard
      archs + CPU pipeline only; LR-scale work stays on Anyscale.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

# Fixed order: the upgrade probe mutates the shared system pip env, so it
# runs LAST whatever subset/order the caller requests.
PROBE_ORDER = ("offload_disable", "float_post_attach", "unsloth_upgrade")

TRAIN_REL = os.path.join("experiments", "training", "train_sft.py")
DTYPE_ERR = "expected mat1 and mat2 to have the same dtype"

# P1 preamble: neuter the smart offload BEFORE train_sft imports unsloth.
# unsloth/models/_utils.py from-imports patch_unsloth_smart_gradient_-
# checkpointing at ITS import time (called with use_gradient_checkpointing
# ="unsloth", which train_sft passes), so neutering the zoo module attrs
# here — before unsloth is imported anywhere — poisons those from-imports
# too. The _utils bound names are overwritten defensively as well.
P1_PREAMBLE = (
    "import torch\n"
    "import torch.utils.checkpoint as _ckpt_mod\n"
    "_PRISTINE_CHECKPOINT = getattr(_ckpt_mod, '_old_checkpoint', _ckpt_mod.checkpoint)\n"
    "_PRISTINE_CKPT_FN = getattr(_ckpt_mod, '_old_CheckpointFunction', _ckpt_mod.CheckpointFunction)\n"
    "def _plain_checkpoint(function, *args, use_reentrant=None, **kwargs):\n"
    "    kwargs.pop('use_reentrant', None)\n"
    "    return _PRISTINE_CHECKPOINT(function, *args, use_reentrant=True, **kwargs)\n"
    "def _plain_apply(function, preserve_rng_state=True, *args, **kwargs):\n"
    "    kwargs.pop('use_reentrant', None)\n"
    "    return _PRISTINE_CHECKPOINT(function, *args, use_reentrant=True, **kwargs)\n"
    "def _disable_offload(*args, **kwargs):\n"
    "    _ckpt_mod.checkpoint = _plain_checkpoint\n"
    "    _ckpt_mod.CheckpointFunction = _PRISTINE_CKPT_FN\n"
    "    try:\n"
    "        import transformers.modeling_utils as _mu\n"
    "        _mu.checkpoint = _plain_checkpoint\n"
    "    except Exception:\n"
    "        pass\n"
    "    return None\n"
    "import unsloth  # establish the guarded import chain first\n"
    "import unsloth_zoo.gradient_checkpointing as _gc\n"
    "_gc.patch_unsloth_smart_gradient_checkpointing = _disable_offload\n"
    "_gc.unpatch_unsloth_smart_gradient_checkpointing = _disable_offload\n"
    "_gc.patch_unsloth_gradient_checkpointing = _disable_offload\n"
    "_gc.unpatch_unsloth_gradient_checkpointing = _disable_offload\n"
    "_gc.patch_gradient_checkpointing = _disable_offload\n"
    "_gc.unpatch_gradient_checkpointing = _disable_offload\n"
    "_gc.unsloth_offloaded_gradient_checkpoint = _plain_checkpoint\n"
    "_gc.unsloth_gradient_checkpoint = _plain_checkpoint\n"
    "_gc.unsloth_checkpoint = _plain_checkpoint\n"
    "_gc.UnslothCheckpointFunction.apply = _plain_apply\n"
    "try:\n"
    "    import unsloth.models._utils as _uu\n"
    "    _uu.patch_unsloth_smart_gradient_checkpointing = _disable_offload\n"
    "    _uu.unpatch_unsloth_smart_gradient_checkpointing = _disable_offload\n"
    "    _uu.patch_unsloth_gradient_checkpointing = _disable_offload\n"
    "    _uu.unpatch_unsloth_gradient_checkpointing = _disable_offload\n"
    "except Exception as _e:\n"
    "    print(f'BISECT-P1: _utils pre-bind skipped ({type(_e).__name__})', flush=True)\n"
    "_disable_offload()\n"
    "print('BISECT-P1: offload-disable shims installed (plain torch ckpt)', flush=True)\n"
)

# P2 preamble: coerce the PEFT-attached model to fp32 after get_peft_model
# but before training, by wrapping trl.SFTTrainer (train_sft does
# `from trl import SFTTrainer` at module level, i.e. AFTER this runs, so it
# picks up the wrapper). .cuda() re-pins after the dtype cast.
P2_PREAMBLE = (
    "import torch\n"
    "try:\n"
    "    import trl.trainer.sft_trainer as _sft_mod\n"
    "    _OrigSFT = _sft_mod.SFTTrainer\n"
    "except Exception:\n"
    "    import trl as _sft_mod\n"
    "    _OrigSFT = _sft_mod.SFTTrainer\n"
    "class _Float32SFTTrainer(_OrigSFT):\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        _m = kwargs.get('model', args[0] if args else None)\n"
    "        if _m is not None:\n"
    "            try:\n"
    "                _m.float()\n"
    "                try:\n"
    "                    _m.cuda()\n"
    "                except Exception:\n"
    "                    pass\n"
    "                _d = sorted({str(p.dtype) for p in _m.parameters()})\n"
    "                print(f'BISECT-P2: post-attach fp32 coerce done, dtypes={_d}', flush=True)\n"
    "            except Exception as _e:\n"
    "                print(f'BISECT-P2: fp32 coerce FAILED ({type(_e).__name__}: {_e})', flush=True)\n"
    "        super().__init__(*args, **kwargs)\n"
    "_sft_mod.SFTTrainer = _Float32SFTTrainer\n"
    "try:\n"
    "    import trl as _trl_top\n"
    "    _trl_top.SFTTrainer = _Float32SFTTrainer\n"
    "except Exception:\n"
    "    pass\n"
    "print('BISECT-P2: fp32-coerce trainer wrapper installed', flush=True)\n"
)

# P3 needs no preamble: the upgrade runs in the parent before the plain flow.
P3_PREAMBLE = "print('BISECT-P3: post-upgrade standard flow (no shims)', flush=True)\n"

PREAMBLES = {
    "offload_disable": P1_PREAMBLE,
    "float_post_attach": P2_PREAMBLE,
    "unsloth_upgrade": P3_PREAMBLE,
}

# Tail shared by all probes: exec the repo trainer with the standard argv
# shape (MODEL STEPS DATA OUT RESUME=""), env carrying the knobs.
RUNPY_TAIL = (
    "import os, runpy, sys\n"
    "_train = os.path.join(os.environ['BISECT_TRAIN_DIR'], 'train_sft.py')\n"
    "sys.path.insert(0, os.environ['BISECT_TRAIN_DIR'])\n"
    "sys.argv = [_train, os.environ['BISECT_MODEL'], os.environ['BISECT_STEPS'],\n"
    "            os.environ['BISECT_DATA_DIR'], os.environ['BISECT_OUT_DIR'], '']\n"
    "runpy.run_path(_train, run_name='__main__')\n"
)


def _env(name, default):
    return os.environ.get(name, default)


def _result_path():
    p = _env("BISECT_RESULT", "/kaggle/working/bisect_result.json")
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    except Exception:
        p = os.path.join(_env("OUT_DIR", "/tmp"), "bisect_result.json")
    return p


def _unsloth_versions():
    try:
        from importlib.metadata import version
        return f"unsloth=={version('unsloth')}/unsloth_zoo=={version('unsloth_zoo')}"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def _signature(output, returncode):
    if returncode == 0 and "DONE" in output:
        return "train completed: DONE marker at STEPS=" + _env("BISECT_STEPS", "10")
    m = re.search(r"expected mat1 and mat2 to have the same dtype[^\n]{0,120}", output)
    if m:
        return m.group(0).strip()
    for pat in ("CUDA out of memory", "RuntimeError", "Error"):
        hits = [ln.strip() for ln in output.splitlines() if pat in ln]
        if hits:
            return hits[-1][:300]
    tail = "\n".join(output.splitlines()[-5:])
    return f"rc={returncode}, no DONE marker; tail: {tail[:300]}"


def _run_flow(probe, train_dir, log_path, timeout):
    """Run the (possibly shimmed) standard flow in a fresh subprocess."""
    wrapper = os.path.join(tempfile.gettempdir(), f"bisect_{probe}_wrap.py")
    with open(wrapper, "w") as f:
        f.write(PREAMBLES[probe] + RUNPY_TAIL)
    env = dict(os.environ)
    # Sanctioned no-bf16 flow defaults; bootstrap values (if set) win.
    env.setdefault("SFT_FP16", "1")
    env.setdefault("UNSLOTH_FORCE_FLOAT32", "1")
    env["BISECT_TRAIN_DIR"] = train_dir
    env["BISECT_MODEL"] = _env("MODEL", "scholzmx/sepalith-base-qwen35-2b-text")
    env["BISECT_STEPS"] = _env("BISECT_STEPS", "10")
    env["BISECT_DATA_DIR"] = _env("DATA_DIR", "/tmp/data/sft_v7")
    env["BISECT_OUT_DIR"] = os.path.join(_env("OUT_DIR", "/kaggle/working/run_sft"),
                                         f"bisect_{probe}")
    try:
        p = subprocess.run([sys.executable, wrapper], cwd=train_dir, env=env,
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        with open(log_path, "w") as f:
            f.write(out)
        return p.returncode, out, None
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") if isinstance(e.stdout, str) else "") + "\n" \
            + ((e.stderr or "") if isinstance(e.stderr, str) else "")
        with open(log_path, "w") as f:
            f.write(out)
        return 124, out, f"TIMEOUT after {timeout}s (no DONE marker)"


def _probe_offload_disable(train_dir, log_path, timeout):
    return _run_flow("offload_disable", train_dir, log_path, timeout)


def _probe_float_post_attach(train_dir, log_path, timeout):
    return _run_flow("float_post_attach", train_dir, log_path, timeout)


def _probe_unsloth_upgrade(train_dir, log_path, timeout):
    up = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U",
         "unsloth>=2026.9.2", "unsloth_zoo"],
        capture_output=True, text=True, timeout=1200)
    if up.returncode != 0:
        msg = ((up.stdout or "") + "\n" + (up.stderr or "")).strip()[-500:]
        return 1, msg, f"pip upgrade FAILED (rc={up.returncode}): {msg[:200]}"
    return _run_flow("unsloth_upgrade", train_dir, log_path, timeout)


PROBE_FNS = {
    "offload_disable": _probe_offload_disable,
    "float_post_attach": _probe_float_post_attach,
    "unsloth_upgrade": _probe_unsloth_upgrade,
}


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    cli_probes = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--probes=")]
    if cli_probes:
        wanted = [p.strip() for p in cli_probes[0].split(",") if p.strip()]
    else:
        wanted = [p.strip() for p in _env("BISECT_PROBES", "").split(",") if p.strip()] \
            or list(PROBE_ORDER)
    unknown = [p for p in wanted if p not in PROBE_FNS]
    if unknown:
        print(f"BISECT: unknown probes {unknown} (known: {list(PROBE_ORDER)})", flush=True)
        return 2
    # Fixed order regardless of request order (upgrade mutates shared pip env).
    probes = [p for p in PROBE_ORDER if p in wanted]
    for a in sys.argv:
        if a.startswith("--steps="):
            os.environ["BISECT_STEPS"] = a.split("=", 1)[1]
    steps = _env("BISECT_STEPS", "10")
    timeout = int(_env("BISECT_TIMEOUT", "2700"))
    train_dir = os.path.join(os.getcwd(), "experiments", "training")
    if not os.path.isfile(os.path.join(train_dir, "train_sft.py")):
        # Already at experiments/training (or elsewhere with train_sft.py).
        alt = os.path.join(os.getcwd(), "train_sft.py")
        train_dir = os.getcwd() if os.path.isfile(alt) else train_dir
    result_path = _result_path()
    print(f"BISECT: probes={probes} steps={steps} timeout={timeout}s", flush=True)
    print(f"BISECT: unsloth pre-run: {_unsloth_versions()}", flush=True)
    records = []
    for probe in probes:
        log_path = os.path.join(os.path.dirname(result_path) or ".",
                                f"train_bisect_{probe}.log")
        print(f"BISECT: --- probe {probe} (STEPS={steps}) ---", flush=True)
        rc, out, hard_err = PROBE_FNS[probe](train_dir, log_path, timeout)
        sig = hard_err or _signature(out, rc)
        status = "passed" if (hard_err is None and rc == 0 and "DONE" in out) else "crashed"
        rec = {"probe": probe, "status": status, "signature": sig[:500],
               "unsloth_version": _unsloth_versions()}
        records.append(rec)
        try:
            with open(result_path.replace(".json", f"_{probe}.json"), "w") as f:
                json.dump(rec, f, indent=1)
        except Exception as e:
            print(f"BISECT: WARN per-probe write failed ({e})", flush=True)
        print(f"BISECT: probe {probe}: {status}: {sig[:200]}", flush=True)
    with open(result_path, "w") as f:
        json.dump(records, f, indent=1)
    print(f"BISECT: wrote {result_path}", flush=True)
    winners = [r["probe"] for r in records if r["status"] == "passed"]
    if winners:
        print(f"BISECT DECISION: PASSING probe(s) {winners} — rerun at "
              f"STEPS=300 as the confirmation arm.", flush=True)
    else:
        print("BISECT DECISION: all probes crashed — door closed, document "
              "signatures in the research doc §gpu-blocker.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
