#!/usr/bin/env python3
"""MTP-preserving GGUF export for the qwen3.5-2b b4 checkpoint (S1 arm c).

`export_gguf.py` passes `--no-nextn` for qwen archs because the LOCAL base
(`models/qwen3.5-2b-base-text-hf`, the 2026-08-20 text-only strip) has NO
`mtp.*` tensors — the strip kept only embed/layers/norm/rotary children, so
every GGUF on disk is MTP-less. The UPSTREAM release
(`Qwen/Qwen3.5-2B-Base`, the VLM wrapper the strip was made from) DOES ship
the MTP head: 15 `mtp.*` tensors, contiguous at the tail of
`model.safetensors-00001-of-00001.safetensors` (bytes 4426488512..4548144832,
121.7 MB, bf16). This script range-fetches exactly those bytes (cached at
`models/qwen3.5-2b-mtp-head.safetensors`), grafts them onto the b4 LoRA
merge, and runs the b10453 converter WITHOUT `--no-nextn` so the output
carries `blk.24.nextn.*` tensors + `nextn_predict_layers=1` — the
`--spec-type draft-mtp` serve artifact (W13). Also emits the two-file
variant (converter `--mtp`: head-only draft GGUF for `-md`).

The b4 LoRA never saw the MTP head (it was absent from the training base),
so the grafted head is exactly the upstream base MTP layer — the honest
"base MTP head + b4 body" composite. Acceptance measured with it is a
BASE-family-MTP datapoint, not a b4-trained-MTP one; flag carried into
S1_RIG.md.

Stages (each skippable if its output exists; --fresh to redo):
  fetch   range-fetch upstream mtp.* -> models/qwen3.5-2b-mtp-head.safetensors
  graft   merged HF dir (unsloth/PEFT, run separately per export_gguf.py
          flow; --merged-dir) + mtp head -> single model.safetensors +
          config with mtp_num_hidden_layers=1 in a prepared export dir
  convert b10453 convert_hf_to_gguf.py WITHOUT --no-nextn (f16)
          + --mtp head-only run (f16)
  quant   llama-quantize (b10453) -> Q8_0 both
  verify  tensor-list + KV check on the Q8 artifacts (assert nextn tensors
          survived quantize)

Usage (run from repo root, .venv-sft):
  .venv-sft/bin/python experiments/training/export_gguf_mtp.py \
      --merged-dir /tmp/merged_mtp-b4_qwen35_2b --stem mtp-b4_qwen35_2b
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/training
EXP = HERE.parent                               # experiments
MODELS = EXP / "models"                         # experiments/models

UPSTREAM_URL = ("https://huggingface.co/Qwen/Qwen3.5-2B-Base/resolve/main/"
                "model.safetensors-00001-of-00001.safetensors")
# from the upstream safetensors header (fetched + parsed at runtime; these
# are the expected values, asserted so a repo-side change fails loudly)
EXPECTED_MTP_TENSORS = 15
HEAD_CACHE = MODELS / "qwen3.5-2b-mtp-head.safetensors"

CONVERT = Path("/tmp/llamacpp-convert-b10453/convert_hf_to_gguf.py")
QUANT = EXP / "bin" / "llama" / "llama-b10453" / "llama-quantize"
PY = sys.executable


def log(msg):
    print(f"[export_gguf_mtp] {msg}", flush=True)


def fetch_header(url, n_bytes=200_000):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read(8 + n_bytes)  # urlopen follows redirects
    (hlen,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8:8 + hlen])


def fetch_mtp_head(fresh=False):
    """Range-fetch the contiguous mtp.* span -> HEAD_CACHE (safetensors)."""
    if HEAD_CACHE.exists() and not fresh:
        log(f"fetch: cached {HEAD_CACHE}")
        return HEAD_CACHE
    hdr = fetch_header(UPSTREAM_URL)
    mtp = {k: v for k, v in hdr.items() if k != "__metadata__" and k.startswith("mtp.")}
    assert len(mtp) == EXPECTED_MTP_TENSORS, \
        f"upstream mtp tensor count changed: {len(mtp)} != {EXPECTED_MTP_TENSORS}"
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k)
                   for k, v in mtp.items())
    # assert contiguity (single range request)
    for (b0, e0, _), (b1, _, _) in zip(spans, spans[1:]):
        assert e0 == b1, "mtp.* span no longer contiguous upstream"
    lo, hi = spans[0][0], spans[-1][1]
    log(f"fetch: {len(spans)} tensors, bytes {lo}..{hi} ({(hi - lo) / 1e6:.1f} MB)")
    req = urllib.request.Request(UPSTREAM_URL,
                                 headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        blob = r.read()
    assert len(blob) == hi - lo, f"short fetch {len(blob)} != {hi - lo}"
    import torch
    from safetensors.torch import save_file
    out = {}
    for b, e, k in spans:
        t = hdr[k]
        assert t["dtype"] == "BF16", f"{k}: unexpected dtype {t['dtype']}"
        buf = bytearray(blob[b - lo:e - lo])  # bytearray: writable for frombuffer
        out[k] = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(t["shape"])
    HEAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(HEAD_CACHE))
    log(f"fetch: wrote {HEAD_CACHE} ({len(out)} tensors)")
    return HEAD_CACHE


def graft(merged_dir, head_file, export_dir):
    """merged HF dir + mtp head -> export dir with a single grafted
    model.safetensors and config asserting mtp_num_hidden_layers=1."""
    import torch
    from safetensors.torch import load_file, save_file
    export_dir.mkdir(parents=True, exist_ok=True)
    for f in merged_dir.iterdir():
        if f.name == "model.safetensors" or f.suffix in (".json", ".jinja"):
            if f.name != "model.safetensors":
                shutil.copy2(f, export_dir / f.name)
    merged = load_file(str(merged_dir / "model.safetensors"))
    head = load_file(str(head_file))
    overlap = set(merged) & set(head)
    assert not overlap, f"merged model already has mtp tensors: {sorted(overlap)}"
    merged.update(head)
    save_file(merged, str(export_dir / "model.safetensors"))
    cfg_p = export_dir / "config.json"
    cfg = json.loads(cfg_p.read_text())
    n = cfg.get("mtp_num_hidden_layers")
    if n != 1:
        log(f"graft: config mtp_num_hidden_layers={n} -> forcing 1 "
            f"(we graft exactly the 1 upstream MTP layer)")
        cfg["mtp_num_hidden_layers"] = 1
        cfg_p.write_text(json.dumps(cfg, indent=2))
    log(f"graft: {len(merged)} tensors -> {export_dir}")
    return export_dir


def convert(export_dir, stem):
    f16 = MODELS / f"{stem}-f16.gguf"
    head_f16 = MODELS / f"{stem}-head-f16.gguf"
    if not f16.exists():
        # NOTE: NO --no-nextn — this is the whole point (MTP tensors survive)
        subprocess.run([PY, str(CONVERT), str(export_dir),
                        "--outfile", str(f16), "--outtype", "f16"], check=True)
        log(f"convert: {f16}")
    if not head_f16.exists():
        subprocess.run([PY, str(CONVERT), str(export_dir), "--mtp",
                        "--outfile", str(head_f16), "--outtype", "f16"],
                       check=True)
        log(f"convert (head-only): {head_f16}")
    return f16, head_f16


def quantize(f16, head_f16, stem):
    q8 = MODELS / f"{stem}-Q8_0.gguf"
    head_q8 = MODELS / f"{stem}-head-Q8_0.gguf"
    if not q8.exists():
        subprocess.run([str(QUANT), str(f16), str(q8), "Q8_0"], check=True)
    if not head_q8.exists():
        subprocess.run([str(QUANT), str(head_f16), str(head_q8), "Q8_0"], check=True)
    log(f"quant: {q8} + {head_q8}")
    return q8, head_q8


def verify(gguf_path, expect_nextn=True):
    from gguf import GGUFReader
    r = GGUFReader(str(gguf_path))
    names = [t.name for t in r.tensors]
    nextn = sorted(n for n in names if ".nextn." in n)
    kvs = {}
    for k in r.fields:
        if "nextn" in k:
            v = r.fields[k]
            try:
                val = v.parts_encoding, [x for x in r._getField(k, v.types)]
            except Exception:
                val = "<unreadable>"
            kvs[k] = val
    n_blk = sorted({int(n.split(".")[1]) for n in names if n.startswith("blk.")})
    print(json.dumps(dict(file=str(gguf_path), tensors=len(names),
                          blocks=[n_blk[0], n_blk[-1]], n_nextn_tensors=len(nextn),
                          nextn_tensors=nextn[:6], nextn_kvs=str(kvs)), indent=1))
    if expect_nextn:
        last = n_blk[-1]
        assert any(f"blk.{last}.nextn." in n for n in names), \
            f"NO nextn tensors in blk.{last} — MTP did not survive!"
    return nextn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", default="/tmp/merged_mtp-b4_qwen35_2b",
                    help="unsloth/PEFT LoRA merge output (export_gguf.py flow)")
    ap.add_argument("--stem", default="mtp-b4_qwen35_2b")
    ap.add_argument("--export-dir", default="/tmp/mtp_export-b4_qwen35_2b")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["all", "fetch", "graft", "convert", "quant",
                             "verify"])
    a = ap.parse_args()
    merged = Path(a.merged_dir)
    assert (merged / "model.safetensors").exists(), \
        f"{merged} has no model.safetensors — run the merge first"
    if a.stage in ("all", "fetch"):
        fetch_mtp_head(fresh=a.fresh)
    if a.stage in ("all", "graft"):
        graft(merged, HEAD_CACHE, Path(a.export_dir))
    if a.stage in ("all", "convert"):
        f16, head_f16 = convert(Path(a.export_dir), a.stem)
    else:
        f16 = MODELS / f"{a.stem}-f16.gguf"
        head_f16 = MODELS / f"{a.stem}-head-f16.gguf"
    if a.stage in ("all", "quant"):
        q8, head_q8 = quantize(f16, head_f16, a.stem)
    else:
        q8 = MODELS / f"{a.stem}-Q8_0.gguf"
        head_q8 = MODELS / f"{a.stem}-head-Q8_0.gguf"
    if a.stage in ("all", "verify"):
        nextn = verify(q8, expect_nextn=True)
        print(f"VERIFY-OK: {len(nextn)} nextn tensors in {q8.name}")
        verify(head_q8, expect_nextn=True)
        f16.unlink(missing_ok=True)
        head_f16.unlink(missing_ok=True)
        log(f"done -> {q8} + {head_q8} (f16 intermediates removed)")


if __name__ == "__main__":
    main()
