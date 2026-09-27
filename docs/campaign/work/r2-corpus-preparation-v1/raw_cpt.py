"""TRAIN-only raw-R profile materializer. No model import or training execution."""
from __future__ import annotations
import argparse, hashlib, json, os, re, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
TOKENIZER_SHA = '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
BOS, EOS, PAD = 0, 1, 1
MAX_FILE_BYTES = 4 * 1024 * 1024

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def partition(group):
    return 'cpt_validation' if int(hashlib.sha256(('DAT10-CPT-validation-v1\0' + group).encode()).hexdigest(), 16) % 20 == 0 else 'cpt_train'

def chunks(ids, size=4096):
    """Each code token supervised once; only actual document end receives EOS.

    Every chunk starts BOS. Continuations retain the prior code token with its
    label masked, so the first new code token has its true immediate predecessor.
    No padding here. EOS may require its own terminal three-token chunk.
    """
    if size < 4: raise ValueError('chunk size must be at least four')
    if any(x in (BOS, EOS) for x in ids): raise ValueError('raw code contains a special token ID')
    start = 0
    while True:
        carry = ids[start - 1:start] if start else []
        capacity = size - 1 - len(carry)
        end = min(len(ids), start + capacity)
        terminal = end == len(ids) and end - start < capacity
        seq = [BOS] + carry + ids[start:end] + ([EOS] if terminal else [])
        labels = [-100] * (1 + len(carry)) + ids[start:end] + ([EOS] if terminal else [])
        yield {'input_ids': seq, 'labels': labels, 'attention_mask': [1] * len(seq),
               'source_token_start': start, 'source_token_end': end,
               'overlap_context_tokens': len(carry), 'is_document_end': terminal,
               'supervised_tokens': sum(x != -100 for x in labels)}
        if terminal: return
        start = end

def allowed_license(value):
    # Conservative recognized license families; custom/file-only terms remain excluded.
    return bool(re.search(r'(^|[ |+,(])(?:A?GPL|LGPL|MIT|BSD|Apache|Artistic|MPL|CC0|Unlimited)(?:[- (]|$)', value))

def fingerprints(raw):
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'sha1': hashlib.sha1(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}

def read_file(row):
    p = Path(row['path'])
    if row['split'] != 'train_group' or row['cpt_partition'] != partition(row['group_id']):
        raise ValueError('TRAIN split or CPT partition mismatch')
    if not allowed_license(row['license']): return row, None, 'license_not_in_recognized_families'
    if p.is_symlink(): raise ValueError('symlink source')
    before = p.stat()
    expected = (row['bytes'], row['mtime_ns'], row['inode'], row['device'])
    actual = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
    if actual != expected or not 0 < before.st_size <= MAX_FILE_BYTES: raise ValueError('source stat changed or size out of bounds')
    with p.open('rb') as f: raw = f.read(MAX_FILE_BYTES + 1)
    after = p.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) != expected or len(raw) != before.st_size:
        raise ValueError('source changed during read')
    try: text = raw.decode('utf-8')
    except UnicodeDecodeError: return row, None, 'non_utf8'
    if '\0' in text: return row, None, 'NUL_in_source'
    return row, (raw, text, fingerprints(raw)), None

def select(rows, train_bytes, validation_bytes):
    """Hash-order package groups, then round-robin files; never truncate a file."""
    by_partition = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['split'] != 'train_group' or partition(r['group_id']) != r['cpt_partition']: raise ValueError('bad split')
        by_partition[r['cpt_partition']][r['group_id']].append(r)
    selected = []
    for part, budget in [('cpt_validation', validation_bytes), ('cpt_train', train_bytes)]:
        groups = by_partition[part]
        keys = sorted(groups, key=lambda g: hashlib.sha256(('DAT10-profile-package-v1\0' + g).encode()).digest())
        for g in keys: groups[g].sort(key=lambda r: hashlib.sha256(r['path'].encode()).digest())
        consumed = 0
        for depth in range(max((len(x) for x in groups.values()), default=0)):
            for g in keys:
                if depth >= len(groups[g]): continue
                r = groups[g][depth]
                if consumed + r['bytes'] > budget: continue
                selected.append(r); consumed += r['bytes']
    return selected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=Path, default=HERE/'initial-raw-file-snapshot.jsonl')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-bytes', type=int, default=12*1024*1024)
    ap.add_argument('--validation-bytes', type=int, default=2*1024*1024)
    ap.add_argument('--chunk-size', type=int, default=4096)
    args = ap.parse_args()
    if hasattr(os, 'sched_setaffinity'): os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if sha(TOKENIZER) != TOKENIZER_SHA: raise ValueError('tokenizer identity mismatch')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER)); tok.encode_special_tokens = True
    rows = [json.loads(x) for x in args.metadata.open()]
    selected = select(rows, args.train_bytes, args.validation_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'selected-source-metadata.jsonl').write_text(''.join(json.dumps(x, separators=(',', ':'))+'\n' for x in selected))
    parent_hashes = set(json.loads((HERE/'known-nontrain-parent-hashes.json').read_text()))
    seen = {}; counts = defaultdict(Counter); packages = defaultdict(set); excluded = Counter(); started = time.monotonic()
    files = {part: (args.output/(part+'.jsonl')).open('w') for part in ('cpt_train', 'cpt_validation')}
    docs = (args.output/'documents.jsonl').open('w'); rejects = (args.output/'exclusions.jsonl').open('w')
    try:
        # Validation first, so cross-partition exact duplicates are removed from TRAIN.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row, result, reason in pool.map(read_file, selected):
                if not reason:
                    raw, text, hashes = result
                    if parent_hashes.intersection(hashes.values()): reason = 'known_nontrain_parent_hash_match'
                    elif hashes['sha256'] in seen: reason = 'exact_duplicate_of_' + seen[hashes['sha256']]
                if reason:
                    excluded[reason] += 1; rejects.write(json.dumps({'path': row['path'], 'reason': reason})+'\n'); continue
                ids = tok.encode(text, add_special_tokens=False).ids
                if tok.decode(ids, skip_special_tokens=False) != text: raise ValueError('tokenizer roundtrip mismatch')
                if not ids: raise ValueError('empty tokenization')
                part = row['cpt_partition']; seen[hashes['sha256']] = part
                source = {**row, **hashes, 'source_code_tokens': len(ids), 'source_utf8_bytes': len(raw)}
                doc_id = hashes['sha256']
                chunk_count = 0
                for chunk_index, chunk in enumerate(chunks(ids, args.chunk_size)):
                    record = {'schema': 1, 'row_id': doc_id+':'+str(chunk_index), 'document_id': doc_id,
                              'package': row['package'], 'group_id': row['group_id'], 'cpt_partition': part,
                              'source_path': row['path'], 'source_sha256': doc_id, 'chunk_index': chunk_index, **chunk}
                    files[part].write(json.dumps(record, separators=(',', ':'))+'\n')
                    counts[part]['rows'] += 1; counts[part]['input_tokens'] += len(chunk['input_ids'])
                    counts[part]['supervised_tokens'] += chunk['supervised_tokens']; chunk_count += 1
                docs.write(json.dumps({**source, 'document_id': doc_id, 'chunks': chunk_count}, separators=(',', ':'))+'\n')
                counts[part]['documents'] += 1; counts[part]['raw_bytes'] += len(raw); counts[part]['code_tokens'] += len(ids)
                packages[part].add(row['package'])
                if sum(v['documents'] for v in counts.values()) % 100 == 0:
                    print(json.dumps({'documents': sum(v['documents'] for v in counts.values()), 'counts': dict(counts), 'seconds': time.monotonic()-started}), flush=True)
    finally:
        for f in [*files.values(), docs, rejects]: f.close()
    for part in counts:
        counts[part]['packages'] = len(packages[part])
        if counts[part]['supervised_tokens'] != counts[part]['code_tokens'] + counts[part]['documents']: raise ValueError('loss denominator mismatch')
    manifest = {'schema': 1, 'status': 'CPU_materialized_candidate_not_training_admission', 'counts': dict(counts),
        'tokenizer_path': str(TOKENIZER), 'tokenizer_sha256': TOKENIZER_SHA, 'metadata_path': str(args.metadata),
        'metadata_sha256': sha(args.metadata), 'partition_sha256': sha(HERE/'cpt-train-group-partition.json'),
        'parent_hash_guard_sha256': sha(HERE/'known-nontrain-parent-hashes.json'), 'source_sha256': sha(__file__),
        'max_length': args.chunk_size, 'bos_id': BOS, 'eos_id': EOS, 'pad_id': PAD,
        'sampling_scope': 'Hash-ordered groups from the immutable initial partial metadata inventory; not representative of all normalized packages.',
        'boundaries': 'Sequential source chunks; one masked prior-code-token overlap on continuation; BOS on every chunk; EOS only actual document end; no discarded code tokens.',
        'padding': 'None stored. Collator must pad attention with 0 and labels with -100, without masking actual EOS by token ID.',
        'heldout_scope': 'All campaign non-TRAIN groups excluded by registry before source traversal. CPT validation reserves TRAIN groups before reads. Exact TRAIN dedup and metadata SHA1/SHA256/git-blob identity guard; no complete heldout near-duplicate claim.',
        'exclusions': dict(excluded), 'seconds': time.monotonic()-started,
        'artifacts': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__': main()
