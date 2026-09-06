#!/usr/bin/env python3
"""Sample rendered training text by family for llama-imatrix (no tokenizer dependency)."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def build(source, output, per_family=64, seed=3407):
    if source.name != 'train.jsonl':
        raise ValueError('Expected an assembled train.jsonl, never an evaluation split')
    if per_family < 1:
        raise ValueError('per-family must be positive')
    rng = random.Random(seed)
    pools, counts = {}, Counter()
    digest = hashlib.sha256()
    with source.open('rb', buffering=4 * 1024 * 1024) as rows:
        for raw in rows:
            digest.update(raw)
            row = json.loads(raw)
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Every training row must have nonempty rendered text')
            family = row.get('family', 'unknown')
            counts[family] += 1
            pool = pools.setdefault(family, [])
            if len(pool) < per_family:
                pool.append(text)
            else:
                index = rng.randrange(counts[family])
                if index < per_family:
                    pool[index] = text
    selected = [text for family in sorted(pools) for text in pools[family]]
    if not selected:
        raise ValueError('Training split is empty')
    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n\n'.join(selected) + '\n', encoding='utf-8')
    manifest = dict(source=str(source.resolve()), source_sha256=digest.hexdigest(),
                    seed=seed, per_family=per_family, available=dict(counts),
                    selected={k: len(v) for k, v in pools.items()},
                    corpus_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--per-family', type=int, default=64)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    print(json.dumps(build(args.source, args.output, args.per_family, args.seed), indent=2))
