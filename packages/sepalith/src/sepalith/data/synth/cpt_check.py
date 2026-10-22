"""Verbatim-overlap check of the normalised seed sources against the CPT corpus.

Decodes every CPT row (token IDs, pinned tokenizer) and records which seed
fingerprints occur in CPT:

- formatting-invariant token 12-grams (``dedup.text_tokens``/``kgram_hashes``);
- line 4-grams (``dedup.line_4grams``), as in v1.

Writes hashes and counts only, never CPT text. The result is bound to the
seed source tree, the CPT corpus file and the tokenizer (sha256 each) and to
the fingerprint parameters; ``build.py`` refuses to admit a build whose
binding does not match.

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.data.synth.cpt_check \\
        --sources /mnt/e/.../SYNTH-v2/sources --out /mnt/e/.../SYNTH-v2/cpt-hits.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sepalith.data.synth.dedup import FINGERPRINT, kgram_hashes, line_4grams, text_tokens, token_ids

CPT = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl")
# Pinned in receipts/C05-cpt-overlap.json (inputs.cpt_rows.sha256).
CPT_SHA256 = "7b3edfd052de4be5d1180144f75497aef5e862eeed816b372c5d751a855425bf"
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")
SCHEMA = "sepalith.synth.cpt-hits.v2"
_STATE: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Deterministic digest of every ``*.R`` file (relative path and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.R")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def binding(sources: Path) -> dict:
    return {"source_tree_sha256": tree_sha256(sources), "cpt_sha256": CPT_SHA256,
            "tokenizer_sha256": sha256_file(TOKENIZER), "fingerprint": dict(FINGERPRINT)}


def _init(grams: frozenset[int], tokens: np.ndarray) -> None:
    from tokenizers import Tokenizer
    os.nice(10)
    _STATE.update(tok=Tokenizer.from_file(str(TOKENIZER)), grams=grams, tokens=tokens)


def _scan(span: tuple[int, int]) -> tuple[int, list[int], list[int]]:
    tok, grams, tokens = _STATE["tok"], _STATE["grams"], _STATE["tokens"]
    line_hits: set[int] = set()
    token_hits: set[int] = set()
    rows = 0
    with CPT.open("rb") as stream:
        stream.seek(span[0])
        if span[0]:
            stream.readline()
        while stream.tell() <= span[1]:
            line = stream.readline()
            if not line:
                break
            text = tok.decode(json.loads(line)["input_ids"], skip_special_tokens=True)
            line_hits |= line_4grams(text) & grams
            hashes = kgram_hashes(token_ids(text_tokens(text.split("\n"))))
            if hashes.size:
                found = hashes[np.isin(hashes, tokens)]
                token_hits.update(int(h) for h in np.unique(found))
            rows += 1
    return rows, sorted(line_hits), sorted(token_hits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    by_file: dict[str, tuple[set[int], np.ndarray]] = {}
    for path in sorted(args.sources.rglob("*.R")):
        text = path.read_text(encoding="utf-8")
        hashes = np.unique(kgram_hashes(token_ids(text_tokens(text.split("\n")))))
        by_file[str(path.relative_to(args.sources))] = (line_4grams(text), hashes)
    grams = frozenset().union(*(g for g, _ in by_file.values()))
    tokens = np.unique(np.concatenate([h for _, h in by_file.values()]))
    with ThreadPoolExecutor(max_workers=1) as hasher:
        corpus_sha = hasher.submit(sha256_file, CPT)
        size = CPT.stat().st_size
        spans = [(size * i // 64, size * (i + 1) // 64) for i in range(64)]
        line_hits: set[int] = set()
        token_hits: set[int] = set()
        rows = 0
        with ProcessPoolExecutor(max_workers=min(args.workers, 8), initializer=_init,
                                 initargs=(grams, tokens)) as pool:
            for n, lh, th in pool.map(_scan, spans):
                rows += n
                line_hits.update(lh)
                token_hits.update(th)
        corpus = corpus_sha.result()
    if corpus != CPT_SHA256:
        raise SystemExit(f"CPT corpus sha256 {corpus} differs from the pinned {CPT_SHA256}")
    hit_array = np.array(sorted(token_hits), dtype=np.uint64)
    per_file = Counter()
    for name, (g, h) in by_file.items():
        n = len(g & line_hits) + int(np.isin(h, hit_array).sum())
        if n:
            per_file[name] = n
    result = {"schema": SCHEMA, "binding": binding(args.sources), "cpt_rows_scanned": rows,
              "seed_line_grams": len(grams), "seed_token_grams": int(tokens.size),
              "hit_line_gram_count": len(line_hits), "hit_token_gram_count": len(token_hits),
              "seed_files_with_hits": dict(per_file), "hit_line_grams": sorted(line_hits),
              "hit_token_grams": [int(h) for h in hit_array]}
    args.out.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("hit_") or k.endswith("count")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
