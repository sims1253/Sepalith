#!/usr/bin/env python3
"""Bind derived length-profile metadata into the pilot result atomically."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb', buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    result_path = a.output / 'result.json'; profile_path = a.output / 'length-profile.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete_diagnostic_candidate_pending_final_dedup_root_admission':
        raise ValueError('unexpected_result_status')
    if json.loads(profile_path.read_text(encoding='utf-8')).get('training_admission') is not False:
        raise ValueError('profile_is_not_diagnostic_only')
    result.setdefault('artifacts', {}).pop('result.json', None)
    result['artifacts']['length-profile.json'] = {'bytes': profile_path.stat().st_size, 'sha256': sha_file(profile_path)}
    result['artifact_manifest_policy'] = 'result.json self-hash is omitted; receipt binds the final result hash.'
    result['length_profile'] = {'path': str(profile_path), 'bytes': profile_path.stat().st_size, 'sha256': sha_file(profile_path)}
    atomic_json(result_path, result)
    for name, binding in result['artifacts'].items():
        path = a.output / name
        if path.stat().st_size != binding['bytes'] or sha_file(path) != binding['sha256']:
            raise ValueError(f'artifact_binding_mismatch:{name}')
    print(json.dumps({'result_path': str(result_path), 'result_bytes': result_path.stat().st_size,
                      'result_sha256': sha_file(result_path), 'artifact_count': len(result['artifacts'])}, sort_keys=True))


if __name__ == '__main__':
    main()
