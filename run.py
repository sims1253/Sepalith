#!/usr/bin/env python3
"""Sepalith A2 cluster bootstrap — the one-command rented-machine path.

    git clone <sepalith> && cd sepalith          # + stage data (see below)
    python3 run.py all --data-root /data         # everything else happens

`all` chains: doctor -> verify -> repack -> manifest -> train (13B staged,
detached, auto-resumes) -> gates (auto-run when training finishes; the
wrapper script also re-runs gates standalone: `python3 run.py gates`).

Data staging (the only manual step — ~100GB from the NAS):
    python3 run.py stage-list --nas-root /mnt/h/sepalith   # emits rsyncs
Every later step is idempotent: repack skips finished strata, train
resumes from latest.pt (step-pure data order makes resume exact), gates
re-score any checkpoint. A killed instance just needs `run.py all` again.

Sub-commands: doctor | verify | repack | manifest | train | gates | all |
stage-list. Stdlib-only (torch etc. only needed by the steps that use it).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "experiments/training/poc_twin/train_a2.py"
REPACK = REPO / "experiments/data-mining/r_repack_full.py"
ASSEMBLE = REPO / "experiments/data-mining/assemble_a2_mixture.py"

# the adopted recipe (runbook §3.2 — every value POC-verified)
RECIPE = ["--arm", "muon", "--lr", "0.01", "--lr-embed", "0.004",
          "--d-model", "2048", "--n-layers", "24", "--n-q", "16",
          "--n-kv", "2", "--head-dim", "128", "--ffn-hidden", "8192",
          "--vocab", "32768", "--exits", "8,16", "--tokens-per-step",
          "524288", "--micro-bs", "0", "--seed", "1273"]
TARGETS = {          # staged decision (runbook §4): 13B first, then 25B
    "13b": dict(steps=24_800, tag="a2_run1_13b"),
    "25b": dict(steps=47_700, tag="a2_run2_25b"),
}

# staged-data spec: (path under data-root, required, what it is)
SPEC = [
    ("datasets/a2_tokenizer_v1/tokenizer.json", True, "the frozen 32K tokenizer"),
    ("a2_transfers/english/blocks.npy", True, "english stratum"),
    ("a2_transfers/python_v2/blocks.npy", False, "python stratum (fetching on NAS)"),
    ("a2_transfers/c_cpp_v2/blocks.npy", True, "c/cpp stratum"),
    ("a2_transfers/js_ts_v2/blocks.npy", True, "js/ts stratum"),
    ("a2_transfers/sql_v2/blocks.npy", True, "sql stratum"),
    ("a2_transfers/julia_v2/blocks.npy", True, "julia stratum"),
    ("a2_transfers/matlab_v2/blocks.npy", True, "matlab stratum"),
    ("a2_transfers/python/blocks.npy", True, "curated python stratum"),
    ("a2/r/so_r_qa.npy", False, "SO r-tag keep set (packed; provisional share)"),
    ("a2/r/bioc.npy", False, "Bioconductor R/tests (packed; provisional share)"),
    ("datasets/astfim_v1/fixed/train.jsonl", False, "R raw (or stage a2/r/ packed)"),
    ("datasets/astfim_v1/fixed/eval.jsonl", False, "R eval raw"),
    ("datasets/astfim_random_v1/train-000.jsonl", False, "random-cut raw"),
    ("datasets/scenarios_v1/no_op.jsonl", False, "no_op raw"),
    ("a2/r/stats.json", False, "pre-packed R strata (NAS-side repack)"),
]
MIN_FREE_GB = 150          # block files + ckpts (~11GB each) + headroom


def sh(cmd, env=None, **kw):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, env=e, **kw)


def data_root_arg(ap):
    ap.add_argument("--data-root", default="/data",
                    help="staged-data root (default /data)")


# ---------------------------------------------------------------- doctor
def cmd_doctor(args):
    ok = True
    print(f"python {sys.version.split()[0]}")
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        print(f"torch {torch.__version__} + CUDA {torch.version.cuda} OK: "
              f"{name}, {total/2**30:.0f}GB ({free/2**30:.0f} free)")
        if free / 2**30 < 60:
            print("  WARN: <60GB free VRAM — 1.5B run wants an 80GB card")
    except Exception as e:
        ok = False
        print(f"torch/CUDA UNAVAILABLE ({e}). Install:")
        print("  pip install torch --index-url "
              "https://download.pytorch.org/whl/cu126")
    for mod in ("numpy", "tokenizers"):
        try:
            __import__(mod)
            print(f"{mod} OK")
        except ImportError:
            ok = False
            print(f"{mod} MISSING — pip install {mod}")
    du = shutil.disk_usage(args.data_root if os.path.ismount(args.data_root)
                           or Path(args.data_root).exists() else "/")
    print(f"disk at {args.data_root}: {du.free/2**30:.0f}GB free "
          f"(need {MIN_FREE_GB}GB+)")
    if du.free / 2**30 < MIN_FREE_GB:
        print("  WARN: low disk — block files + checkpoints need room")
    if os.environ.get("POC_MEM_FRACTION") != "0.95":
        print("note: run.py train sets POC_MEM_FRACTION=0.95 itself")
    return 0 if ok else 1


# ---------------------------------------------------------------- verify
def cmd_verify(args):
    root = Path(args.data_root)
    missing_req, warned = [], []
    for rel, required, why in SPEC:
        p = root / rel
        if p.exists() and (not str(p).endswith(".npy") or p.stat().st_size > 1_000_000):
            continue
        line = f"  {'MISSING' if required else 'absent'}: {rel} — {why}"
        (missing_req if required else warned).append(line)
    for l in warned:
        print(l)
    if missing_req:
        print("REQUIRED staged data missing:")
        print("\n".join(missing_req))
        print("stage it (NAS side): python3 run.py stage-list "
              "--nas-root /mnt/h/sepalith --dst " + str(root))
        sys.exit(1)
    contam = root / "a2/r/contamination.json"
    if contam.exists():
        v = json.loads(contam.read_text()).get("VERDICT")
        print(f"contamination gate: {v}")
        if v != "PASS":
            sys.exit("REFUSING: contamination verdict is not PASS")
    elif (root / "a2/r/stats.json").exists():
        print("note: packed R strata present but no contamination.json "
              "(pre-packed without gate?)")
    print(f"verify OK — staged tree complete at {root}")
    return 0


# ---------------------------------------------------------------- repack
def cmd_repack(args):
    cmd = [sys.executable, str(REPACK), "--root", args.data_root,
           "--out", str(Path(args.data_root) / "a2" / "r"),
           "--skip-existing"]
    if args.nas_mirror:
        cmd += ["--mirror", args.nas_mirror]
    print(" ".join(cmd), flush=True)
    return sh(cmd).returncode


# --------------------------------------------------------------- manifest
def cmd_manifest(args):
    cmd = [sys.executable, str(ASSEMBLE), "--data-root", args.data_root]
    print(" ".join(cmd), flush=True)
    return sh(cmd).returncode


# ----------------------------------------------------------------- train
def cmd_train(args):
    root = Path(args.data_root)
    tgt = TARGETS[args.target]
    tag = args.tag or tgt["tag"]
    ckpt_dir = root / f"checkpoints/{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    final = ckpt_dir / "final.pt"

    train_cmd = [sys.executable, str(TRAIN)] + RECIPE + [
        "--steps", str(args.steps or tgt["steps"]), "--mixture",
        str(root / "a2_mixture_manifest.json"), "--tag", tag,
        "--ckpt-dir", str(ckpt_dir), "--ckpt-every", str(args.ckpt_every)]
    if latest.exists() and not final.exists():
        train_cmd += ["--resume", str(latest)]
        print(f"[train] resuming from {latest}")
    elif final.exists():
        print(f"[train] {final} exists — run complete; use `run.py gates`")
        return 0

    # wrapper: train, then gates on success (the run-and-report loop)
    script = log_dir / f"run_{tag}.sh"
    gates = [sys.executable, str(REPO / "run.py"), "gates",
             "--data-root", str(root), "--ckpt", str(final), "--tag", tag]
    script.write_text(
        "#!/bin/sh\nset -e\n" + " ".join(f"'{c}'" for c in train_cmd) + "\n"
        + " ".join(f"'{c}'" for c in gates) + "\n")
    script.chmod(0o755)
    log = log_dir / f"{tag}.log"
    env = {"POC_MEM_FRACTION": "0.95",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print(f"[train] launching detached: {script} -> {log}")
    if args.dry_run:
        print(" ".join(train_cmd))
        return 0
    out = open(log, "a")
    pid = subprocess.Popen(["setsid", "nice", "-n", "10", str(script)],
                           stdout=out, stderr=out, env={**os.environ, **env},
                           start_new_session=True).pid
    (log_dir / f"{tag}.pid").write_text(str(pid))
    print(f"[train] pid {pid} (kill by PID only). watch: "
          f"tail -f {log}")
    return 0


# ----------------------------------------------------------------- gates
def cmd_gates(args):
    root = Path(args.data_root)
    ckpt = Path(args.ckpt) if args.ckpt else root / (
        f"checkpoints/{args.tag or TARGETS['13b']['tag']}/final.pt")
    if not ckpt.exists():
        sys.exit(f"no checkpoint at {ckpt} — train first")
    stats = json.loads((root / "a2/r/stats.json").read_text())
    results = {}
    for slice_name in ("eval_causal", "eval_rc"):
        s = stats["streams"][slice_name]
        cmd = [sys.executable, str(REPO / "experiments/training/poc_twin/"
                                   "ladder/eval_a2_exits.py"),
               "--ckpt", str(ckpt), "--blocks",
               str(root / f"a2/r/{slice_name}.npy"),
               "--bytes", str(s["bytes"])]
        print(" ".join(cmd), flush=True)
        r = sh(cmd, capture_output=True, text=True)
        try:
            results[slice_name] = json.loads(
                r.stdout.strip().splitlines()[-1])["bpb_per_exit"]
        except Exception:
            print(r.stdout, r.stderr)
            sys.exit(f"gate eval failed on {slice_name}")
    # pre-registered reads (runbook §4)
    causal = results["eval_causal"]
    keys = [k for k in causal if k.startswith("exit_")]
    exit24 = "top" if "top" in causal else None
    gap = None
    if "exit_16" in causal and exit24:
        gap = (causal["exit_16"] - causal[exit24]) / causal[exit24]
    report = dict(ckpt=str(ckpt), when=time.strftime("%FT%T"),
                  bpb=results,
                  exit16_vs_top_gap_pct=round(100 * gap, 2) if gap else None,
                  ship_M_signal=(gap is not None and gap <= 0.02),
                  note="exit_16 within 2% of top = the ship-M signal "
                       "(§4); causal floor vs twin anchor = manual read")
    out = root / f"gates_{ckpt.parent.name}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"GATES_DONE -> {out}  ship_M_signal={report['ship_M_signal']}")
    return 0


# ------------------------------------------------------------ stage-list
def cmd_stage_list(args):
    nas = Path(args.nas_root)
    dst = args.dst or "/data"
    rsyncs = []
    for rel, required, why in SPEC:
        src = nas / rel
        if not src.exists():
            print(f"# absent on NAS, skipped: {rel} ({why})")
            continue
        line = (f"rsync -a --info=progress2 \"{src}\" "
                f"\"{dst}/{str(rel).rsplit('/', 1)[0]}/\"")
        rsyncs.append(line)
    # stats.json companions for the strata (token accounting)
    rsyncs.append(f"rsync -a --include='*/' --include='stats.json' "
                  f"--exclude='*' \"{nas}/a2_transfers/\" "
                  f"\"{dst}/a2_transfers/\"")
    print("# --- stage to the instance (run on the NAS-reachable box) ---")
    print("# (or, if pretraining/ is on the HF dataset: "
          "hf download scholzmx/sepalith --repo-type dataset "
          f"--include 'pretraining/*' --local-dir {dst})")
    print("\n".join(rsyncs))
    print(f"# then ON THE INSTANCE: python3 run.py all --data-root {dst}")
    print("# NAS-side pre-flight (recommended, uses the mirror):")
    print(f"#   python3 {REPACK} --root {nas} --mirror {nas}/a2_code_mirror")


# ------------------------------------------------------------------ all
def cmd_all(args):
    if cmd_doctor(args):
        sys.exit("doctor failed — fix the environment before the run "
                 "(see messages above)")
    if cmd_verify(args):
        sys.exit(1)
    if args.dry_run:
        print("[all --dry-run] would run: repack (idempotent) -> manifest "
              "-> train detached (target "
              f"{args.target}, tag {args.tag or TARGETS[args.target]['tag']}) "
              "-> gates on completion")
        return 0
    for step in (cmd_repack, cmd_manifest, cmd_train):
        rc = step(args)
        if rc:
            sys.exit(rc)
    print("\n[all] pipeline live: training detached, gates will run on "
          "completion. Re-run `run.py all` after any interruption — "
          "every step resumes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="env check (torch/CUDA/deps/disk)")
    data_root_arg(d)
    d.set_defaults(fn=cmd_doctor)

    v = sub.add_parser("verify", help="staged-data completeness gate")
    data_root_arg(v)
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser("repack", help="R-side re-pack (idempotent)")
    data_root_arg(r)
    r.add_argument("--nas-mirror", default=None,
                   help="run the mirror contamination gate (NAS-side)")
    r.set_defaults(fn=cmd_repack)

    m = sub.add_parser("manifest", help="build a2_mixture_manifest.json")
    data_root_arg(m)
    m.set_defaults(fn=cmd_manifest)

    t = sub.add_parser("train", help="launch the adopted recipe (detached)")
    data_root_arg(t)
    t.add_argument("--target", choices=TARGETS, default="13b")
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--tag", default=None)
    t.add_argument("--ckpt-every", type=int, default=2000)
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("gates", help="pre-registered 13B gate evals")
    data_root_arg(g)
    g.add_argument("--ckpt", default=None)
    g.add_argument("--tag", default=None)
    g.set_defaults(fn=cmd_gates)

    s = sub.add_parser("stage-list", help="emit the NAS->instance rsyncs")
    s.add_argument("--nas-root", default="/mnt/h/sepalith")
    s.add_argument("--dst", default=None)
    s.set_defaults(fn=cmd_stage_list)

    a = sub.add_parser("all", help="verify -> repack -> manifest -> train "
                                   "(the one command)")
    data_root_arg(a)
    a.add_argument("--nas-mirror", default=None)
    a.add_argument("--target", choices=TARGETS, default="13b")
    a.add_argument("--tag", default=None)
    a.add_argument("--steps", type=int, default=None)
    a.add_argument("--ckpt-every", type=int, default=2000)
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(fn=cmd_all)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
