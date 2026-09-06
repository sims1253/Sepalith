#!/usr/bin/env python3
"""Fetch one indexed archive object and verify its bytes before publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def verify(path, entry):
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Downloaded artifact does not match the Git index')


def fetch(index, original_path, output=None, downloader=None):
    matches = [r for r in index['artifacts'] if r['original_path'] == original_path]
    if len(matches) != 1:
        raise ValueError('Expected one indexed artifact for this original path')
    entry = matches[0]
    target = output or (Path.home() / '.cache/sepalith/artifacts' /
                        entry['sha256'] / Path(original_path).name)
    if target.exists():
        verify(target, entry)
        return target
    if downloader is None:
        from huggingface_hub import download_bucket_files
        downloader = download_bucket_files
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fetch-', dir=target.parent) as temporary:
        staged = Path(temporary) / 'artifact'
        downloader(index['bucket'], [(entry['object'], staged)], raise_on_missing_files=True)
        verify(staged, entry)
        # Same-filesystem hard link publishes complete bytes atomically and
        # refuses to replace a concurrently created destination.
        os.link(staged, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_path', help='Original repository path from the archive index')
    parser.add_argument('--output', type=Path, help='Explicit destination; defaults to the local cache')
    args = parser.parse_args()
    index = json.loads((ROOT / 'docs/artifacts/index.json').read_text())
    print(fetch(index, args.original_path, args.output))


if __name__ == '__main__':
    main()
