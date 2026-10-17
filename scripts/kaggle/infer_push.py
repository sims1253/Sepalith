#!/usr/bin/env python3
"""Push, poll and fetch a Kaggle T4 inference kernel (card C03).

Replaces ``scripts/cloud/kaggle_push.sh`` for inference. The kernel is a short
generated bootstrap with no secret in it:

1. installs the PC's pins (transformers 5.5.0, tokenizers 0.22.2,
   accelerate 1.14.0) next to Kaggle's own torch;
2. downloads this repository at ``--commit`` from GitHub (the repository is
   public, so the commit must be pushed) and checks a digest of every file it
   uses (``packages/sepalith/src`` and ``scripts/kaggle/infer_entry.py``)
   against the digest computed here from git;
3. runs ``scripts/kaggle/infer_entry.py`` with the arguments after ``--``.

Public HF sources need no token. A Kaggle dataset model source
(``--model kaggle:<owner>/<slug>[/dir]`` in the entry arguments) is attached
automatically. Outputs land in ``/kaggle/working/<run-id>/``; this driver
fetches them with ``kaggle kernels output`` and, with ``--upload-hf``, pushes
them to ``datasets/scholzmx/sepalith`` under
``campaign-20260915/inference/<run-id>/`` in one commit from the PC (the HF
token stays on the PC).

Quota (R7): ``kaggle quota`` is read before the push and after the run; the
push is refused (exit 3, stop-for-user) when the week's used GPU hours plus
``--estimate-hours`` would exceed ``--weekly-limit-hours`` (default 20). Every
run appends a line to ``~/.local/state/sepalith/kaggle/quota-ledger.jsonl``
and writes ``driver-receipt.json`` next to the fetched outputs.

Auth: the Kaggle CLI 2.x reads ``KAGGLE_API_TOKEN`` (a ``KGAT_`` access
token). If it is not in the environment, it is read from
``~/.config/sepalith/secrets.env`` into the CLI subprocess only. It is never
printed or written.

Example (the C03 smoke)::

    python3 scripts/kaggle/infer_push.py --run-id c03-smoke-20261012 --estimate-hours 0.5 -- \\
        --model hf:scholzmx/sepalith-2b-cpt/checkpoint-11586 \\
        --model-sha256 4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51 \\
        --panel dev75 --case-limit 10 --max-new-tokens 128 --dtype fp16 --batch-size 5 --workers 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = "sims1253/Sepalith"
KAGGLE_USER = "m0hawk"
KAGGLE_BIN = Path.home() / ".local/bin/kaggle"
SECRETS = Path.home() / ".config/sepalith/secrets.env"
LEDGER = Path.home() / ".local/state/sepalith/kaggle/quota-ledger.jsonl"
DEFAULT_FETCH_ROOT = Path.home() / ".local/state/sepalith/kaggle/runs"
HF_DATASET = "scholzmx/sepalith"
HF_PREFIX = "campaign-20260915/inference"
CODE_PATHS = ("packages/sepalith/src", "scripts/kaggle/infer_entry.py")
PIP_PINS = ("transformers==5.5.0", "tokenizers==0.22.2", "accelerate==1.14.0")
TERMINAL = ("complete", "error", "cancel")
SECRET_PATTERNS = (re.compile(r"KGAT_[A-Za-z0-9]{8,}"), re.compile(r"hf_[A-Za-z0-9]{20,}"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="output folder name; also the kernel slug suffix")
    p.add_argument("--commit", default="HEAD", help="repository commit the kernel runs (must be on GitHub)")
    p.add_argument("--estimate-hours", type=float, required=True, help="expected GPU hours (R7 check)")
    p.add_argument("--weekly-limit-hours", type=float, default=20.0, help="R7 stop-for-user limit per quota week")
    p.add_argument("--timeout-hours", type=float, default=11.5, help="kernel run-time limit (Kaggle max 12)")
    p.add_argument("--cpu", action="store_true", help="no accelerator (packaging checks; no GPU quota)")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--no-wait", action="store_true", help="push and return; fetch later with --fetch-only")
    p.add_argument("--fetch-only", action="store_true", help="skip the push; poll, fetch and record an existing kernel")
    p.add_argument("--fetch-root", type=Path, default=DEFAULT_FETCH_ROOT)
    p.add_argument("--upload-hf", action="store_true", help=f"upload the outputs to {HF_DATASET}/{HF_PREFIX}/<run-id>/")
    p.add_argument("--dry-run", action="store_true", help="write the kernel folder and print its path; no push")
    p.add_argument("entry_args", nargs=argparse.REMAINDER, help="-- then infer_entry.py arguments")
    a = p.parse_args(argv)
    if a.entry_args and a.entry_args[0] == "--":
        a.entry_args = a.entry_args[1:]
    if not a.fetch_only and not a.entry_args:
        p.error("pass the infer_entry.py arguments after --")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,40}", a.run_id):
        p.error("--run-id: 3-41 lowercase letters, digits and dashes")
    if "--run-id" in a.entry_args or "--output-dir" in a.entry_args:
        p.error("the driver sets --run-id and --output-dir")
    return a


def slug(run_id):
    return f"sepalith-infer-{run_id}"[:50].rstrip("-")


# Kaggle CLI ---------------------------------------------------------------------------------------

def kaggle_env():
    env = dict(os.environ)
    if not env.get("KAGGLE_API_TOKEN"):
        for line in SECRETS.read_text().splitlines() if SECRETS.is_file() else []:
            if line.startswith("export KAGGLE_API_TOKEN="):
                env["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")
    if not env.get("KAGGLE_API_TOKEN"):
        raise SystemExit(f"KAGGLE_API_TOKEN is not set and not in {SECRETS}")
    return env


def kaggle(*args, check=True, timeout=600):
    result = subprocess.run([str(KAGGLE_BIN), *args], env=kaggle_env(), capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed ({result.returncode}): {redact(result.stderr or result.stdout)[-2000:]}")
    return result


def redact(text):
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    token = os.environ.get("KAGGLE_API_TOKEN")
    return text.replace(token, "<redacted>") if token else text


def quota():
    """Parse ``kaggle quota``: {"GPU": {"used": h, "remaining": h, "total": h, "refresh": iso}, ...}."""
    out = kaggle("quota").stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in ("GPU", "TPU"):
            table[parts[0]] = {"used": float(parts[1].rstrip("h")), "remaining": float(parts[2].rstrip("h")),
                               "total": float(parts[3].rstrip("h")), "refresh": parts[4]}
    if "GPU" not in table:
        raise SystemExit(f"cannot parse kaggle quota output: {out!r}")
    return table


def status(kernel):
    text = kaggle("kernels", "status", kernel, check=False).stdout.strip()
    match = re.search(r'status "?([A-Za-z_.]+)"?', text)
    value = (match.group(1) if match else text).lower().rsplit(".", 1)[-1]
    return value, text


# Code identity ------------------------------------------------------------------------------------

def git(*args, text=True):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=text).stdout


def code_digest(commit):
    """sha256 over sorted ``path<TAB>sha256(blob)`` lines of every file the kernel uses."""
    lines = []
    for path in git("ls-tree", "-r", "--name-only", commit, "--", *CODE_PATHS).splitlines():
        blob = git("cat-file", "blob", f"{commit}:{path}", text=False)
        lines.append(f"{path}\t{hashlib.sha256(blob).hexdigest()}\n")
    if not any(line.startswith("scripts/kaggle/infer_entry.py\t") for line in lines):
        raise SystemExit(f"{commit} has no scripts/kaggle/infer_entry.py")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest(), len(lines)


def require_pushed(commit):
    remote = git("branch", "-r", "--contains", commit).split()
    if not remote:
        raise SystemExit(f"commit {commit} is not on any remote branch; push it first (the kernel fetches it from GitHub)")
    return remote


# Kernel -------------------------------------------------------------------------------------------

BOOTSTRAP = r'''"""Sepalith C03 inference bootstrap (generated by scripts/kaggle/infer_push.py; no secrets)."""
import hashlib, io, json, os, pathlib, subprocess, sys, tarfile, time, traceback, urllib.request
COMMIT = __COMMIT__
CODE_DIGEST = __DIGEST__
CODE_PATHS = __CODE_PATHS__
PIP_PINS = __PIP__
ENTRY_ARGS = __ENTRY_ARGS__
RUN_ID = __RUN_ID__
OUT = pathlib.Path("/kaggle/working") / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
boot = {"schema": "sepalith.c03.kaggle-bootstrap.v1", "commit": COMMIT, "code_digest": CODE_DIGEST, "status": "starting"}
T0 = time.monotonic()
def save():
    boot["elapsed_seconds"] = round(time.monotonic() - T0, 1)
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
def mark(step):
    boot.setdefault("steps", []).append([step, round(time.monotonic() - T0, 1)]); save(); print("[bootstrap]", step, flush=True)
try:
    mark("start")
    try:  # CPU kernels (--cpu) have no nvidia-smi
        boot["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        boot["nvidia_smi"] = []
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *PIP_PINS], check=True, timeout=900)
    mark("pip")
    data = urllib.request.urlopen(f"https://codeload.github.com/sims1253/Sepalith/tar.gz/{COMMIT}", timeout=600).read()
    root = pathlib.Path("/tmp/sepalith-code")
    lines = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.split("/", 1)[1] if "/" in member.name else ""
            if not member.isfile() or not any(name == p or name.startswith(p + "/") for p in CODE_PATHS):
                continue
            if name.startswith("/") or ".." in pathlib.PurePosixPath(name).parts:
                raise SystemExit(f"unsafe path in the archive: {name}")
            blob = archive.extractfile(member).read()
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            lines.append(f"{name}\t{hashlib.sha256(blob).hexdigest()}\n")
    digest = hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()
    boot["code_files"] = len(lines)
    if digest != CODE_DIGEST:
        raise SystemExit(f"code digest differs: {digest} != {CODE_DIGEST}")
    mark("code")
    env = dict(os.environ, PYTHONPATH=str(root / "packages/sepalith/src"), PYTHONDONTWRITEBYTECODE="1",
               HF_HOME="/tmp/hf-home", TOKENIZERS_PARALLELISM="false")
    code = subprocess.run([sys.executable, "-u", str(root / "scripts/kaggle/infer_entry.py"), "--run-id", RUN_ID,
                           "--output-dir", str(OUT), "--cache-dir", "/tmp/sepalith-inference", *ENTRY_ARGS], env=env).returncode
    boot["entry_exit"] = code
    boot["status"] = "complete" if code == 0 else "entry_failed"
    mark("entry")
except BaseException as error:
    boot.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    raise
finally:
    save()
if boot.get("entry_exit"):
    sys.exit(boot["entry_exit"])
'''


def kernel_files(args, commit, digest, kernel_id):
    source = BOOTSTRAP
    for key, value in {"__COMMIT__": commit, "__DIGEST__": digest, "__CODE_PATHS__": list(CODE_PATHS),
                       "__PIP__": list(PIP_PINS), "__ENTRY_ARGS__": list(args.entry_args), "__RUN_ID__": args.run_id}.items():
        source = source.replace(key, json.dumps(value))
    # Every kaggle:<owner>/<slug>[/...] source in the entry arguments is attached as a dataset.
    datasets = ["/".join(value.removeprefix("kaggle:").split("/")[:2]) for value in args.entry_args
                if value.startswith("kaggle:")]
    metadata = {"id": kernel_id, "title": kernel_id.split("/", 1)[1], "code_file": "kernel.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": not args.cpu, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": sorted(set(datasets)), "kernel_sources": [],
                "competition_sources": [], "model_sources": []}
    if not args.cpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    return source, metadata


def assert_no_secret(text):
    env = kaggle_env()
    for name in ("KAGGLE_API_TOKEN", "HF_TOKEN"):
        value = env.get(name)
        if value and value in text:
            raise SystemExit(f"refusing to push: the kernel source contains {name}")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise SystemExit("refusing to push: the kernel source contains a token-like string")


# Ledger and upload --------------------------------------------------------------------------------

def append_ledger(record):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def upload(run_dir, run_id):
    """One HF commit with the run's JSON, JSONL and log files (PC side; uses the local ``hf`` login)."""
    staging = Path(tempfile.mkdtemp(prefix="c03-upload-"))
    files = 0
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
            target = staging / path.relative_to(run_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files += 1
    result = subprocess.run(["hf", "upload", HF_DATASET, str(staging), f"{HF_PREFIX}/{run_id}", "--repo-type", "dataset",
                             "--commit-message", f"C03 inference outputs: {run_id}"],
                            capture_output=True, text=True, check=True, timeout=3600)
    return {"repo": f"datasets/{HF_DATASET}", "path": f"{HF_PREFIX}/{run_id}/", "files": files,
            "hf_output": result.stdout.strip()[-300:]}


# Main ---------------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    kernel_id = f"{KAGGLE_USER}/{slug(args.run_id)}"
    fetch_dir = args.fetch_root / args.run_id
    receipt = {"schema": "sepalith.c03.kaggle-driver-receipt.v1", "run_id": args.run_id, "kernel": kernel_id,
               "accelerator": None if args.cpu else "NvidiaTeslaT4", "entry_args": args.entry_args,
               "estimate_hours": args.estimate_hours}
    if not args.fetch_only:
        commit = git("rev-parse", args.commit).strip()
        digest, files = code_digest(commit)
        receipt.update(commit=commit, code_digest=digest, code_files=files)
        source, metadata = kernel_files(args, commit, digest, kernel_id)
        assert_no_secret(source)
        kernel_dir = Path(tempfile.mkdtemp(prefix=f"kaggle-{args.run_id}-"))
        (kernel_dir / "kernel.py").write_text(source)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.dry_run:
            print(json.dumps({"kernel_dir": str(kernel_dir), "kernel": kernel_id, "commit": commit, "digest": digest}))
            return 0
        receipt["remote_branches"] = require_pushed(commit)
        before = quota()
        receipt["quota_before"] = before
        projected = before["GPU"]["used"] + (0 if args.cpu else args.estimate_hours)
        if projected > args.weekly_limit_hours:
            receipt["refused"] = f"R7: {before['GPU']['used']:.2f} h used + {args.estimate_hours:.2f} h estimate > {args.weekly_limit_hours} h"
            print(json.dumps(receipt, indent=2))
            return 3
        push = ["kernels", "push", "-p", str(kernel_dir), "-t", str(int(args.timeout_hours * 3600))]
        if not args.cpu:
            push += ["--accelerator", "NvidiaTeslaT4"]
        receipt["pushed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        pushed = kaggle(*push)
        receipt["push_output"] = redact(pushed.stdout.strip())[-500:]
        print(receipt["push_output"], flush=True)
        # The CLI exits 0 on a refused push (e.g. "Kernel push error: Maximum batch GPU session count of 2
        # reached."); nothing was launched then, so stop here instead of polling a kernel that does not run.
        if "successfully pushed" not in pushed.stdout.lower() or "push error" in pushed.stdout.lower():
            raise SystemExit(f"kaggle refused the push of {kernel_id}: {receipt['push_output']}")
        if args.no_wait:
            print(f"pushed {kernel_id}; later: infer_push.py --run-id {args.run_id} --fetch-only --estimate-hours 0")
            return 0
    started = time.monotonic()
    last = None
    while True:
        value, text = status(kernel_id)
        if value != last:
            print(f"[{time.strftime('%H:%M:%S')}] {kernel_id}: {value}", flush=True)
            last = value
        if any(word in value for word in TERMINAL):
            break
        time.sleep(args.poll_seconds)
    receipt.update(final_status=value, status_text=redact(text), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   polled_seconds=round(time.monotonic() - started, 1))
    fetch_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id, "-p", str(fetch_dir), "-o", check=False, timeout=3600)
    run_dir = fetch_dir / args.run_id
    for name in ("summary.json", "bootstrap.json"):
        path = run_dir / name
        if path.is_file():
            receipt[name.removesuffix(".json")] = {k: v for k, v in json.loads(path.read_text()).items()
                                                   if k in ("status", "elapsed_seconds", "result", "error", "steps",
                                                            "nvidia_smi", "runtime", "gpus", "workers", "code_files")}
    after = quota()
    receipt["quota_after"] = after
    if "quota_before" in receipt:
        receipt["gpu_hours_used"] = round(after["GPU"]["used"] - receipt["quota_before"]["GPU"]["used"], 2)
    receipt["weekly_gpu_hours_used"] = after["GPU"]["used"]
    receipt["r7_stop_for_user"] = after["GPU"]["used"] > args.weekly_limit_hours
    if args.upload_hf and run_dir.is_dir() and value.endswith("complete"):
        receipt["hf_upload"] = upload(run_dir, args.run_id)
    fetch_dir.joinpath("driver-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    append_ledger({k: receipt.get(k) for k in ("run_id", "kernel", "commit", "accelerator", "pushed_at", "finished_at",
                                               "final_status", "gpu_hours_used", "weekly_gpu_hours_used")})
    print(json.dumps({k: receipt.get(k) for k in ("kernel", "final_status", "gpu_hours_used", "weekly_gpu_hours_used",
                                                   "r7_stop_for_user")}, indent=2))
    print(f"outputs: {fetch_dir}")
    return 0 if value.endswith("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
