"""Independent geometry verification of committed preparation shards."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load_rows(folder, manifest, name):
    raw = (folder / name).read_bytes()
    expected = manifest['outputs'][name]
    assert sha(raw) == expected['sha256'] and len(raw) == expected['bytes']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == expected['rows']
    result = {row['row_id']: row for row in rows}
    assert len(result) == len(rows)
    return result

def check(shard):
    folder = ROOT / f'shard-{shard:04d}'
    manifest = json.loads((folder / 'preparation-manifest.json').read_text())
    predictions = load_rows(folder, manifest, 'prediction-inputs.jsonl')
    targets = load_rows(folder, manifest, 'training-sidecar.jsonl')
    holds = load_rows(folder, manifest, 'preparation-holds.jsonl')
    assert set(predictions) == set(targets)
    assert not set(predictions) & set(holds)
    assert len(predictions) + len(holds) == manifest['semantic_supported']
    expected_keys = {'schema', 'row_id', 'path', 'preedit_text', 'preedit_sha256', 'cursor', 'document_eol', 'required_helper_names', 'external_import_dependencies'}
    eols = collections.Counter()
    nonascii = 0
    for rid, prediction in predictions.items():
        assert set(prediction) == expected_keys
        target = targets[rid]
        before = prediction['preedit_text']
        assert sha(before.encode()) == prediction['preedit_sha256']
        assert prediction['document_eol'] in ('lf', 'crlf')
        separator = '\r\n' if prediction['document_eol'] == 'crlf' else '\n'
        lines = before.split(separator)
        cursor = prediction['cursor']
        assert cursor['character'] == 0 and 0 <= cursor['line'] < len(lines)
        assert lines[cursor['line']] == ''
        assert target['target_lines'] and all(isinstance(line, str) and '\r' not in line and '\n' not in line for line in target['target_lines'])
        assert sha(('\n'.join(target['target_lines']) + '\n').encode()) == target['target_sha256']
        lines[cursor['line']] = separator.join(target['target_lines'])
        assert sha(separator.join(lines).encode()) == target['postedit_source_sha256']
        assert target['postedit_source_sha256'] == target['identity']['source_sha256']
        assert target['identity']['split'] == 'train_group'
        eols[prediction['document_eol']] += 1
        nonascii += not before.isascii()
    return {'shard': shard, 'prepared_verified': len(predictions), 'holds': len(holds), 'eols': dict(eols), 'nonascii': nonascii, 'prediction_fields_target_free': True, 'exact_source_reapplication': True, 'training_admission': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', nargs='+', type=int, required=True)
    args = parser.parse_args()
    for shard in args.shards:
        print(json.dumps(check(shard)), flush=True)
