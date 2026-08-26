#!/usr/bin/env python3
"""Full-depth R-side re-pack for the A2 cluster run (the runbook §3.1 step).

Re-renders the SAME astfim_v1 corpus the twin/ladder POCs validated, at
full depth and with the REAL 32K a2_tokenizer_v1 (the POCs ran MiniCPM —
structure-validation only). Streams, mix ratios and renderings are copied
verbatim from the POC packers (data_prep.py / data_prep_ladder.py /
a2_data_prep.py — do not diverge):

  r_causal  : every train row re-rendered as a plain document
              (path header + prefix + span + suffix, PSM markers stripped)
  r_fim_mix : every train row's PSM text + astfim_random_v1 random-cursor
              cuts (12% of ast doc count) + no_op PSM renders (all) —
              seeded shuffle, so dose ~0.30-equivalent at R scale
  r_noop    : the no_op PSM renders alone (the manifest's noop stratum)

Also packs held-out eval slices + their utf-8 byte counts (BPB reads):
eval_causal (fixed/eval.jsonl rows, plain rendering) and eval_rc (the
LAST 400 random-cursor rows — same holdout convention as the POC).

Contamination check (the protocol the mirror is EVAL-PROTECTED for):
sampled word-8-gram shingles of train renderings are checked against the
git mirror (hard fail above 0.1% — aborts the pack) and against the
internal eval slices (package-disjoint by construction; warn-only, some
boilerplate overlap is expected).

Discipline (scars): stream-tokenize into array('i'), memmap +
as_strided block writes ONLY, no whole-stream copies, bounded shingle
memory, and LEAN source loading — the v1 of this script held all 276k
row dicts at once and earlyoom killed it mid-load (strings-only pools,
one stratum's working set at a time, gc between strata). Output layout
matches the manifest builder's cluster paths:
  <out>/{r_causal,r_fim_mix,r_noop}.npy + eval slices + stats.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import random
import re
import time
from array import array
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SEQ = 1025
CTX, HIST, SUF, END = "<|context|>", "<|history|>", "<|suffix|>\n", "\n<|end|>"
SHINGLE_N = 8                 # word 8-grams
SHINGLE_CAP = 2_000_000       # bounded protected-side memory
TRAIN_PROBE_DOCS = 30_000     # contamination probe depth (docs per stream)
MIRROR_FAIL = 0.001           # mirror shingle hit rate above this = hard fail


def causal_from_row(r):
    """Plain-document rendering (data_prep_ladder.py, verbatim)."""
    p, t = r["prompt"], r["target"]
    i_ctx = p.find(CTX)
    i_hist = p.find(HIST)
    i_sf = p.find(SUF)
    if i_ctx != 0 or i_hist == -1 or i_sf == -1:
        return None
    ctx = p[len(CTX):i_hist]
    i_end = p.rfind(END)
    s0 = i_sf + len(SUF)
    suffix = p[s0:i_end] if i_end != -1 else p[s0:]
    span = t
    if span.endswith(END):
        span = span[:-len(END)]
    elif span.endswith("<|end|>"):
        span = span[:-len("<|end|>")]
    return ctx + span + "\n" + suffix


def noop_psm_text(row):
    """no_op PSM render with EMPTY span (a2_data_prep.py, verbatim)."""
    pre = "\n".join(list(row["prefix"]) + list(row["region_old"])) \
        .replace("\r\n", "\n").rstrip("\n")
    suf = "\n".join(row.get("suffix") or []).replace("\r\n", "\n") \
        .strip("\n")
    prompt = (f"<|context|>{row['package']}/{row['path']}\n{pre}\n"
              f"<|history|>\n\n<|cursor|><|suffix|>\n{suf}\n<|end|>\n")
    target = f"\n<|end|>"
    return prompt + target


def jsonl(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield json.loads(line)


def count_rows(path):
    """Fast pre-pass (no json parse) — pool sizing for the rc ratio."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def causal_stream(path):
    """PSM rows -> plain documents, one at a time (never materialized)."""
    for r in jsonl(path):
        c = causal_from_row(r)
        if c is not None and "<|" not in c:
            yield c


def stream_tokens(texts, tok, eos, label):
    """Stream-tokenize (2k batches — the earlyoom-hardened pattern)."""
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
            if n_docs % 50_000 == 0:
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
    """memmap + chunked as_strided (ingest_stackv2 pattern; no copies)."""
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


def shingles(text):
    """Word SHINGLE_N-grams -> 64-bit hashes (bounded, order-free)."""
    words = re.findall(r"\S+", text)
    if len(words) < SHINGLE_N:
        return ()
    return tuple(
        hashlib.blake2b(" ".join(words[i:i + SHINGLE_N]).encode(),
                        digest_size=8).hexdigest()
        for i in range(len(words) - SHINGLE_N + 1))


def protected_shingles(sources):
    """Bounded shingle set over protected content (eval slice, mirror)."""
    prot = set()
    for label, texts in sources:
        for t in texts:
            for h in shingles(t):
                prot.add(int(h, 16))
                if len(prot) >= SHINGLE_CAP * 2:
                    break
        print(f"  [contam] protected '{label}': "
              f"{len(prot)} shingles total", flush=True)
    if len(prot) > SHINGLE_CAP:
        prot = set(random.Random(20260826).sample(sorted(prot),
                                                  SHINGLE_CAP))
    return prot


def contamination_check(train_streams, protected, out_dir):
    """Sampled probe of train renderings vs protected shingle sets.

    train_streams: (label, factory) where factory() yields a FRESH text
    iterable per protected source (generators exhaust).

    The git mirror is the EVAL-PROTECTED asset (protocol: never in
    training) — mirror hits above MIRROR_FAIL abort the pack. The
    internal eval slices are package-disjoint BY CONSTRUCTION; boiler-
    plate overlap there is expected and reported warn-only.
    """
    report = {}
    for label, make_texts in train_streams:
        for src, (prot, hard) in protected.items():
            hits = total = 0
            for i, t in enumerate(make_texts()):
                if i >= TRAIN_PROBE_DOCS:
                    break
                for h in shingles(t):
                    total += 1
                    if int(h, 16) in prot:
                        hits += 1
            rate = hits / max(1, total)
            report[f"{label}_vs_{src}"] = dict(
                hits=hits, shingles=total, rate=round(rate, 6))
            verdict = ("FAIL" if hard and rate > MIRROR_FAIL
                       else "warn" if rate > MIRROR_FAIL else "ok")
            report[f"{label}_vs_{src}"]["verdict"] = verdict
            print(f"  [contam] {label} vs {src}: {hits}/{total} "
                  f"({rate:.4%}) [{verdict}]", flush=True)
            if verdict == "FAIL":
                report["VERDICT"] = "FAIL"
                (out_dir / "contamination.json").write_text(
                    json.dumps(report, indent=1))
                raise SystemExit(
                    f"CONTAMINATION GATE FAILED ({label} vs {src} "
                    f"{rate:.4%} > {MIRROR_FAIL:.4%}) — do not train")
    report["VERDICT"] = "PASS"
    (out_dir / "contamination.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/h/sepalith",
                    help="NAS staging root (datasets/ lives under it); on "
                         "the instance point at the staged copy")
    ap.add_argument("--out", default=None,
                    help="default <root>/a2/r/ (the manifest's cluster path)")
    ap.add_argument("--mirror", default=None,
                    help="a2_code_mirror dir for the contamination gate "
                         "(NAS-side run; skipped if absent)")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="idempotent repack: skip finished strata")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "a2" / "r"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(root / "datasets/a2_tokenizer_v1/"
                                        "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    assert eos is not None, "tokenizer missing <|endoftext|>"

    train_path = root / "datasets/astfim_v1/fixed/train.jsonl"
    eval_path = root / "datasets/astfim_v1/fixed/eval.jsonl"
    rc_path = root / "datasets/astfim_random_v1/train-000.jsonl"
    noop_path = root / "datasets/scenarios_v1/no_op.jsonl"

    stats_path = out / "stats.json"
    stats = {"streams": {}, "seq": SEQ, "eos_id": eos,
             "tokenizer": "a2_tokenizer_v1"}
    if stats_path.exists():
        stats.update(json.loads(stats_path.read_text()))

    def done(name):
        return args.skip_existing and name in stats.get("streams", {})

    # ---- sources: LEAN loading (earlyoom discipline — strings only,
    # one stratum's working set at a time, never row dicts en masse) ----
    n_ast = count_rows(train_path)
    print(f"[load] astfim rows: {n_ast} (counted, never held)", flush=True)

    rc_rows = list(jsonl(rc_path))           # small file (~190MB)
    rc_eval = [r["prompt"] + r["target"] for r in rc_rows[-400:]]
    del rc_rows[-400:]                       # POC holdout convention
    rng = random.Random(20260826)
    rng.shuffle(rc_rows)
    n_rc = round(0.12 * n_ast)
    rc_texts = [r["prompt"] + r["target"] for r in rc_rows[:n_rc]]
    del rc_rows
    noop_texts = [noop_psm_text(r) for r in jsonl(noop_path)]  # small
    print(f"[load] rc mix {len(rc_texts)} docs, no_op {len(noop_texts)}",
          flush=True)

    # ---- r_causal (pure stream: no doc list ever materialized) ----
    if not done("r_causal"):
        s, nd, nb = stream_tokens(causal_stream(train_path), tok, eos,
                                  "r_causal")
        stats["streams"]["r_causal"] = dict(
            docs=nd, bytes=nb, tokens=len(s),
            blocks=write_blocks(s, out / "r_causal.npy"))
        del s
        gc.collect()
        print(f"[r_causal] done "
              f"{stats['streams']['r_causal']['tokens']/1e6:.0f}M tok",
              flush=True)

    # ---- r_fim_mix (lean pool: text strings only, POC shuffle seed) ----
    if not done("r_fim_mix"):
        pool = [r["text"] for r in jsonl(train_path)] + rc_texts \
            + noop_texts
        random.Random(20260823).shuffle(pool)
        shares = dict(ast=n_ast, rc=len(rc_texts), noop=len(noop_texts))
        s, nd, nb = stream_tokens(iter(pool), tok, eos, "r_fim_mix")
        stats["streams"]["r_fim_mix"] = dict(
            docs=nd, bytes=nb, tokens=len(s),
            doc_shares={k: round(v / len(pool), 4) for k, v in
                        shares.items()},
            blocks=write_blocks(s, out / "r_fim_mix.npy"))
        del pool, s
        gc.collect()
        print(f"[r_fim_mix] done "
              f"{stats['streams']['r_fim_mix']['tokens']/1e6:.0f}M tok "
              f"{shares}", flush=True)

    # ---- r_noop ----
    if not done("r_noop"):
        s, nd, nb = stream_tokens(iter(noop_texts), tok, eos, "r_noop")
        stats["streams"]["r_noop"] = dict(
            docs=nd, bytes=nb, tokens=len(s),
            blocks=write_blocks(s, out / "r_noop.npy"))
        del s
        gc.collect()
        print(f"[r_noop] done "
              f"{stats['streams']['r_noop']['tokens']/1e6:.0f}M tok",
              flush=True)

    # ---- eval slices + byte counts (BPB denominators) ----
    for name, texts in (("eval_causal", causal_stream(eval_path)),
                        ("eval_rc", iter(rc_eval))):
        if not done(name):
            s, nd, nb = stream_tokens(texts, tok, eos, name)
            stats["streams"][name] = dict(
                docs=nd, bytes=nb, tokens=len(s),
                blocks=write_blocks(s, out / f"{name}.npy"))
            del s
            gc.collect()
            print(f"[{name}] done {nd} docs {nb} bytes", flush=True)

    # ---- contamination gate (fresh streams per protected source) ----
    if not args.skip_check:
        print("[contam] building protected shingle sets", flush=True)
        protected = {  # name -> (shingles, hard-fail?)
            "eval_slices": (protected_shingles([
                ("eval_causal", causal_stream(eval_path)),
                ("eval_rc", iter(rc_eval))]), False),
        }
        if args.mirror and Path(args.mirror).exists():
            mirror_files = (sorted(Path(args.mirror).rglob("*.R"))[:2000]
                            + sorted(Path(args.mirror).rglob("*.r"))[:2000])
            protected["git_mirror"] = (protected_shingles([(
                "git_mirror",
                (Path(p).read_text(encoding="utf-8",
                                   errors="replace")[:400_000]
                 for p in mirror_files))]), True)
            print(f"  [contam] mirror: {len(mirror_files)} files sampled",
                  flush=True)
        elif args.mirror is None:
            print("  [contam] NOTE: no --mirror — the EVAL-PROTECTED "
                  "mirror check must run NAS-side before staging",
                  flush=True)

        def ast_texts():
            return (r["text"] for r in itertools.islice(
                jsonl(train_path), 200_000))
        contamination_check(
            [("r_causal", lambda: causal_stream(train_path)),
             ("r_fim_mix", ast_texts)],
            protected, out)
    stats["repacked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("tokens", "blocks", "bytes", "docs")}
                      for k, v in stats["streams"].items()}), flush=True)
    print(f"R_REPACK_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
