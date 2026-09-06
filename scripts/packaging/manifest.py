#!/usr/bin/env python3
"""Assemble a release manifest from reviewed runtime bundles and one GGUF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

if __package__:
    from .release_manifest import validate_manifest
else:
    from release_manifest import validate_manifest


def write_manifest(manifest, output):
    """Validate first, then atomically replace the output from a sibling file."""
    validate_manifest(manifest)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix=f'.{output.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(manifest, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', action='append', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--model-url', required=True)
    ap.add_argument('--model-revision', required=True, help='Immutable trained model revision identity')
    ap.add_argument('--tokenizer-revision', required=True, help='Immutable tokenizer revision identity')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    digest = hashlib.sha256()
    with args.model.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    bundles = [json.loads(p.read_text()) for p in args.bundle]
    manifest = dict(schema=1, build='b10453', bundles=bundles,
                    modelProfile=dict(renderer='zeta2-v1', task='r-next-edit',
                                      tokenizerRevision=args.tokenizer_revision,
                                      modelRevision=args.model_revision),
                    model=dict(name=args.model.name, url=args.model_url,
                               sha256=digest.hexdigest(), bytes=args.model.stat().st_size))
    write_manifest(manifest, args.output)


if __name__ == '__main__':
    main()
