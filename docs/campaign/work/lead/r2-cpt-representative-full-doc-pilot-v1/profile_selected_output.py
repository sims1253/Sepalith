#!/usr/bin/env python3
"""Compute immutable geometry statistics for a selected pilot output stream."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from collections import Counter
from pathlib import Path


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb', buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bucket(n: int) -> str:
    if n <= 512: return '1-512'
    if n <= 1024: return '513-1024'
    if n <= 2048: return '1025-2048'
    if n <= 4096: return '2049-4096'
    if n <= 8192: return '4097-8192'
    if n <= 16384: return '8193-16384'
    if n <= 32768: return '16385-32768'
    if n <= 65536: return '32769-65536'
    if n <= 131072: return '65537-131072'
    return '131073+'


def q(values: list[int], fraction: float) -> int:
    values = sorted(values)
    return values[math.ceil(fraction * len(values)) - 1]


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows_path = a.output / 'cpt_train_ctx16384.jsonl'
    provenance_path = a.output / 'selected-document-provenance.jsonl'
    result_path = a.output / 'result.json'
    provenance = [json.loads(line) for line in provenance_path.open(encoding='utf-8')]
    expected_ids = {row['document_id'] for row in provenance}
    lengths = [int(row['document_token_count']) for row in provenance]
    row_lengths: list[int] = []
    row_payload_lengths: list[int] = []
    row_seen: set[str] = set()
    docs: dict[str, int] = Counter()
    terminal = 0
    with rows_path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            ident = row['document_id']
            row_seen.add(ident)
            row_lengths.append(len(row['input_ids']))
            owned = row['source_token_end'] - row['source_token_start']
            row_payload_lengths.append(owned)
            docs[ident] += 1
            terminal += int(row['is_document_end'])
    if row_seen != expected_ids or terminal != len(expected_ids):
        raise ValueError('output_document_or_terminal_mismatch')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    profile = {
        'schema': 'sepalith.sft11.representative_full_document_pilot_length_profile.v1',
        'status': 'complete_diagnostic_only',
        'output_rows': {
            'path': str(rows_path), 'bytes': rows_path.stat().st_size, 'sha256': sha_file(rows_path),
            'rows': len(row_lengths), 'documents': len(row_seen), 'terminal_rows': terminal,
            'input_tokens': sum(row_lengths), 'payload_tokens': sum(row_payload_lengths),
            'input_length_min': min(row_lengths), 'input_length_median': q(row_lengths, .5),
            'input_length_p90': q(row_lengths, .9), 'input_length_p99': q(row_lengths, .99),
            'input_length_max': max(row_lengths), 'length_buckets': dict(sorted(Counter(bucket(x) for x in row_lengths).items())),
            'payload_length_min': min(row_payload_lengths), 'payload_length_median': q(row_payload_lengths, .5),
            'payload_length_p90': q(row_payload_lengths, .9), 'payload_length_p99': q(row_payload_lengths, .99),
            'payload_length_max': max(row_payload_lengths),
        },
        'documents': {
            'count': len(lengths), 'payload_tokens_from_provenance': sum(lengths),
            'length_min': min(lengths), 'length_median': q(lengths, .5), 'length_p90': q(lengths, .9),
            'length_p99': q(lengths, .99), 'length_max': max(lengths),
            'length_buckets': dict(sorted(Counter(bucket(x) for x in lengths).items())),
            'chunks_min': min(docs.values()), 'chunks_median': q(list(docs.values()), .5),
            'chunks_p90': q(list(docs.values()), .9), 'chunks_max': max(docs.values()),
            'chunk_count_distribution': dict(sorted(Counter(docs.values()).items())),
            'documents_over_1024_tokens': sum(x > 1024 for x in lengths),
            'documents_over_16384_tokens': sum(x > 16384 for x in lengths),
        },
        'source_binding': {
            'result_path': str(result_path), 'result_status': result['status'],
            'frozen_rows_sha256': result['source']['frozen_rows_sha256'],
            'frozen_provenance_sha256': result['source']['frozen_document_provenance_sha256'],
        },
        'training_admission': False,
        'final_global_dedup_pending': True,
    }
    atomic_json(a.output / 'length-profile.json', profile)
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
