#!/usr/bin/env python3
"""Build the type-conditioning ablation pair from finish-block records.

For N sampled packages: run ry dump-types once per package, attach a
<filename>types section (bindings of the record's function scope) to each
record's prompt, and write TWO datasets identical except for that section:
  /mnt/h/sepalith/datasets/sft_ablation/{types,plain}/{train,eval}.jsonl
"""
import json, random, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_sft_v1 import render

RY = "/home/m0hawk/Documents/ry-worktrees/dump-types/target/release/ry"
SRC = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/finish_block_sample.jsonl")
OUT = Path("/mnt/h/sepalith/datasets/sft_ablation")
N_PKGS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAX_BINDINGS = 16

records = [json.loads(l) for l in open(SRC)]
pkgs = sorted({r["package"] for r in records})
random.Random(23).shuffle(pkgs)
sampled = pkgs[:N_PKGS]
records = [r for r in records if r["package"] in set(sampled)]
print(f"{len(records)} records from {len(sampled)} packages", flush=True)

_pkg_scopes = {}
def pkg_scope_map(pkg):
    """Run dump-types once per package; map fn-name -> bindings."""
    if pkg in _pkg_scopes:
        return _pkg_scopes[pkg]
    m = {}
    rdir = next((Path("/mnt/h/sepalith/normalized") / pkg).glob(f"*/{pkg}/R"), None)
    if rdir is not None:
        try:
            out = subprocess.run([RY, "dump-types", str(rdir), "--format", "json"],
                                 capture_output=True, text=True, timeout=180)
            for f in json.loads(out.stdout).get("files", []):
                for sc in f.get("scopes", []):
                    if sc.get("kind") == "function" and sc.get("name") and sc["name"] not in m:
                        binds = sc.get("bindings", [])
                        kept = [b for b in binds if b["kind"] == "param"]
                        kept += [b for b in binds if b["kind"] != "param" and b["type"] != "unknown"]
                        kept += [b for b in binds if b["kind"] != "param" and b["type"] == "unknown"
                                 and len(kept) < 4]
                        m[sc["name"]] = kept[:MAX_BINDINGS]
        except Exception:
            pass
    _pkg_scopes[pkg] = m
    return m

def types_section(r):
    binds = pkg_scope_map(r["package"]).get(r["fn"])
    if not binds:
        return None
    return [f"{b['name']}: {b['type']}  # {b['kind']}" for b in binds]

# package-level split for eval
eval_pkgs = set(sampled[:max(1, len(sampled) // 20)])
for variant in ("types", "plain"):
    d = OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    n = {"tr": 0, "ev": 0}
    with open(d / "train.jsonl", "w") as tr, open(d / "eval.jsonl", "w") as ev:
        for r in records:
            prompt, target = render(r)
            sec = types_section(r) if variant == "types" else None
            if variant == "types" and sec:
                lines = prompt.splitlines()
                # insert as its own section right after <[fim-prefix]> header line
                for i, l in enumerate(lines):
                    if l.startswith("<[fim-prefix]>"):
                        lines[i] = "<[fim-prefix]><filename>types"
                        lines[i+1:i+1] = sec + [f"<filename>{r['package']}/{r['path']}"]
                        break
                prompt = "\n".join(lines)
            row = dict(text=prompt + target, prompt=prompt, target=target,
                       kind=r["kind"], gated=r["gated"], package=r["package"],
                       has_types=bool(sec))
            (ev if r["package"] in eval_pkgs else tr).write(json.dumps(row) + "\n")
            n["ev" if r["package"] in eval_pkgs else "tr"] += 1
    print(variant, n, flush=True)

covered = sum(1 for r in records[:500] if types_by.get((r["package"], r["fn"])))
print(f"types coverage (first 500 records): {covered}/500", flush=True)
