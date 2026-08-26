#!/usr/bin/env python3
"""Pack the R-adjacent pretraining strata (2026-08-26, post R-inventory map).

  so_r_qa : stack_staging/files_v3/shard-R-*.jsonl — the license-clean
            (ODC-By-1.0), deduped Stack Exchange R code keep set (v3).
            Plain docs (content as-is), eos-separated. NOTE: this is
            answer-CODE, not Q&A prose — the design's "English×R bridge"
            slice remains unmaterialized; documented in the manifest.
  bioc    : normalized_bioc/<pkg>/<ver>/<pkg>/{R,tests}/**/*.R —
            Bioconductor current, normalized. Doc = package/relpath
            header + file body (the r_causal convention). man/ excluded
            (roxygen-derived, the house double-count rule); vignettes/
            and src/ deferred (prose slice / non-R, GO-time decisions).

Both stream-tokenized (32K a2_tokenizer_v1, eos id 5), memmap +
as_strided 1025-blocks, eval tail slice + byte counts — the r_repack
discipline. so_r_qa holds out its LAST 200 rows as eval_so.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from array import array
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SEQ = 1025


def stream_tokens(texts, tok, eos, label):
    stream = array("i")
    n_docs = n_bytes = 0
    batch = []
    t0 = time.time()
    for t in texts:
        batch.append(t)
        if len(batch) >= 2000:
            for e in tok.encode_batch(batch):
                stream.extend(e.ids)
                stream.append(eos)
            n_docs += len(batch)
            n_bytes += sum(len(x.encode("utf-8", "replace")) for x in batch)
            batch = []
            if n_docs % 100_000 == 0:
                print(f"  [{label}] {n_docs} docs {len(stream)/1e6:.0f}M tok "
                      f"{(time.time()-t0)/60:.0f}m", flush=True)
    if batch:
        for e in tok.encode_batch(batch):
            stream.extend(e.ids)
            stream.append(eos)
        n_docs += len(batch)
        n_bytes += sum(len(x.encode("utf-8", "replace")) for x in batch)
    return stream, n_docs, n_bytes


def write_blocks(stream, out_path):
    n_blocks = (len(stream) - 1) // SEQ
    if n_blocks == 0:
        raise SystemExit(f"EMPTY stream for {out_path}")
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int32,
                                   shape=(n_blocks, SEQ))
    flat = np.frombuffer(stream, dtype=np.int32)
    for b0 in range(0, n_blocks, 4096):
        b1 = min(b0 + 4096, n_blocks)
        seg = flat[b0 * 1024:(b1 - 1) * 1024 + SEQ]
        view = np.lib.stride_tricks.as_strided(
            seg, shape=(b1 - b0, SEQ),
            strides=(seg.strides[0] * 1024, seg.strides[0]))
        mm[b0:b1] = view
    mm.flush()
    del mm
    return n_blocks


def pack(texts, out_dir, name, tok, eos, eval_tail=0):
    train, eval_texts = texts, []
    if eval_tail:
        eval_texts = texts[-eval_tail:]
        train = texts[:-eval_tail]
    s, nd, nb = stream_tokens(train, tok, eos, name)
    stats = {name: dict(docs=nd, bytes=nb, tokens=len(s),
                        blocks=write_blocks(s, out_dir / f"{name}.npy"))}
    del s
    if eval_texts:
        s2, nd2, nb2 = stream_tokens(eval_texts, tok, eos, f"{name}_eval")
        stats[f"{name}_eval"] = dict(
            docs=nd2, bytes=nb2, tokens=len(s2),
            blocks=write_blocks(s2, out_dir / f"{name}_eval.npy"))
        del s2
    gc.collect()
    return stats


def so_texts(root):
    import glob as _g
    for p in sorted(_g.glob(str(root / "stack_staging/files_v3/"
                                   "shard-R-*.jsonl"))):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["content"]


def bioc_texts(root):
    base = root / "normalized_bioc"
    pkgs = sorted(p.name for p in base.iterdir() if p.is_dir())
    print(f"[bioc] {len(pkgs)} packages", flush=True)
    for i, pkg in enumerate(pkgs):
        for ver_dir in sorted((base / pkg).iterdir()):
            tree = ver_dir / pkg
            if not tree.is_dir():
                continue
            for area in ("R", "tests"):
                for rF in sorted((tree / area).rglob("*.R")):
                    try:
                        txt = rF.read_text(encoding="utf-8",
                                           errors="replace")
                    except OSError:
                        continue
                    if txt.strip():
                        rel = f"{pkg}/{ver_dir.name}/{tree.name}" \
                              f"/{rF.relative_to(tree)}"
                        yield f"{rel}\n{txt}"
        if (i + 1) % 300 == 0:
            print(f"  [bioc] walked {i + 1}/{len(pkgs)} pkgs", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/h/sepalith")
    ap.add_argument("--strata", default="so_r_qa,bioc")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "a2" / "r"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(root / "datasets/a2_tokenizer_v1/"
                                        "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    assert eos is not None

    stats_path = out / "stats.json"
    stats = {"streams": {}, "seq": SEQ, "eos_id": eos,
             "tokenizer": "a2_tokenizer_v1"}
    if stats_path.exists():
        stats.update(json.loads(stats_path.read_text()))

    for name in args.strata.split(","):
        if name in stats["streams"]:
            print(f"[{name}] already packed — skip", flush=True)
            continue
        t0 = time.time()
        if name == "so_r_qa":
            texts = list(so_texts(root))
            got = pack(texts, out, "so_r_qa", tok, eos, eval_tail=200)
        elif name == "bioc":
            got = pack(bioc_texts(root), out, "bioc", tok, eos)
        else:
            raise SystemExit(f"unknown stratum {name}")
        stats["streams"].update(got)
        stats_path.write_text(json.dumps(stats, indent=1))
        for k, v in got.items():
            print(f"[{k}] {v['docs']} docs {v['tokens']/1e6:.0f}M tok "
                  f"{v['blocks']} blocks ({(time.time()-t0)/60:.0f}m)",
                  flush=True)
    print("R_ADJACENT_PACK_DONE", flush=True)


if __name__ == "__main__":
    main()
