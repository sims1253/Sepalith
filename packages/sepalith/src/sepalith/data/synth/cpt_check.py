"""Verbatim-overlap check of the normalised seed sources against the CPT corpus.

Decodes every CPT row (token IDs, pinned tokenizer), hashes its line 4-grams
(``dedup.line_4grams``) and records which seed 4-grams occur in CPT. Writes
only hashes and counts, never CPT text.

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.data.synth.cpt_check \
        --sources /mnt/e/.../SYNTH-v1/sources --out /mnt/e/.../SYNTH-v1/cpt-hits.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from sepalith.data.synth.dedup import line_4grams

CPT = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl")
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")
_STATE: dict = {}


def _init(grams: frozenset[int]) -> None:
    from tokenizers import Tokenizer
    os.nice(10)
    _STATE["tok"] = Tokenizer.from_file(str(TOKENIZER))
    _STATE["grams"] = grams


def _scan(span: tuple[int, int]) -> tuple[int, list[int]]:
    tok, grams = _STATE["tok"], _STATE["grams"]
    hits: set[int] = set()
    rows = 0
    with CPT.open("rb") as stream:
        stream.seek(span[0])
        if span[0]:
            stream.readline()
        while stream.tell() <= span[1]:
            line = stream.readline()
            if not line:
                break
            ids = json.loads(line)["input_ids"]
            text = tok.decode(ids, skip_special_tokens=True)
            hits |= line_4grams(text) & grams
            rows += 1
    return rows, sorted(hits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    by_file = {}
    for path in sorted(args.sources.rglob("*.R")):
        by_file[str(path.relative_to(args.sources))] = line_4grams(path.read_text(encoding="utf-8"))
    grams = frozenset().union(*by_file.values())
    size = CPT.stat().st_size
    pieces = 64
    spans = [(size * i // pieces, size * (i + 1) // pieces) for i in range(pieces)]
    hits: set[int] = set()
    rows = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, 8), initializer=_init, initargs=(grams,)) as pool:
        for n, found in pool.map(_scan, spans):
            rows += n
            hits.update(found)
    per_file = Counter({name: len(g & hits) for name, g in by_file.items() if g & hits})
    args.out.write_text(json.dumps({"cpt_rows_scanned": rows, "seed_grams": len(grams), "hit_grams": sorted(hits),
                                    "hit_gram_count": len(hits), "seed_files_with_hits": dict(per_file)},
                                   indent=1) + "\n")
    print(json.dumps({"rows": rows, "seed_grams": len(grams), "hits": len(hits), "files_with_hits": len(per_file)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
