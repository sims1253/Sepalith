#!/usr/bin/env python3
"""Per-layer ffn_down / attn_v quant types of GGUF files (gguf-py GGUFReader).

Checks the llama-quantize ``--tensor-type`` counter-shift hazard (quant report
2026-10-17 §5): a partial ffn_down override (layers 0,1,40,41 -> q6_K) on a
Q4_K_M base should leave every other layer at the base Q4_K_M mix. The
"intended" mix is the plain Q4_K_M file's mix with the overridden layers
replaced by q6_K; the script reports layers where a file deviates from it.

Usage: ffn_down_mix.py --plain PLAIN.gguf --override 0,1,40,41=Q6_K FILE.gguf [...]
"""
from __future__ import annotations

import argparse
import json
import re

from gguf import GGUFReader

PAT = re.compile(r"^blk\.(\d+)\.(ffn_down|attn_v)\.weight$")


def mix(path: str) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {"ffn_down": {}, "attn_v": {}}
    for t in GGUFReader(path).tensors:
        m = PAT.match(t.name)
        if m:
            out[m.group(2)][int(m.group(1))] = t.tensor_type.name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plain", required=True, help="plain Q4_K_M file (no overrides)")
    ap.add_argument("--override", default="0,1,40,41=Q6_K")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    layers_s, qtype = a.override.split("=")
    layers = {int(x) for x in layers_s.split(",")}
    plain = mix(a.plain)["ffn_down"]
    intended = {i: (qtype if i in layers else q) for i, q in plain.items()}
    report: dict = {
        "plain": a.plain,
        "plain_ffn_down": "".join("6" if q == "Q6_K" else "4" if q == "Q4_K" else "?"
                                  for _, q in sorted(plain.items())),
        "intended_ffn_down": "".join("6" if q == "Q6_K" else "4" if q == "Q4_K" else "?"
                                     for _, q in sorted(intended.items())),
        "files": {},
    }
    for f in a.files:
        got = mix(f)["ffn_down"]
        dev = sorted(i for i in intended if got.get(i) != intended[i])
        report["files"][f] = {
            "ffn_down": "".join("6" if q == "Q6_K" else "4" if q == "Q4_K" else "?"
                                for _, q in sorted(got.items())),
            "n_q6k": sum(q == "Q6_K" for q in got.values()),
            "deviating_layers_vs_intended": dev,
            "matches_intended": not dev,
        }
    print(json.dumps(report, indent=1))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
