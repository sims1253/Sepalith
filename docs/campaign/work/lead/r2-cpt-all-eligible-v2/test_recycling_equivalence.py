#!/usr/bin/env python3
"""Actual-source equivalence and atomic recovery tests for the v2 recycler."""

import fcntl
import importlib.util
import json
import tempfile
from pathlib import Path


PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
V1 = PLAN / 'docs/campaign/work/lead/r2-cpt-all-eligible-v1/materialize_all_eligible.py'
V2 = PLAN / 'docs/campaign/work/lead/r2-cpt-all-eligible-v2/materialize_all_eligible.py'
INDEX = 2151  # gasfluxes: multiple real R files and multiple tokenizer chunks.


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def initialize(path: Path, source: dict) -> None:
    (path / 'groups').mkdir(parents=True)
    (path / '.staging').mkdir()
    (path / 'run-manifest.json').write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')


def main() -> None:
    old = module('dat10_v1_reference', V1)
    new = module('dat10_v2_under_test', V2)
    entries, registry, parts, source = new.preflight(new.DEFAULT_OUTPUT)
    entry = entries[INDEX - new.START_INDEX]
    assert entry['name'] == 'gasfluxes'
    seen, protected = new.base_seen()
    raw_cpt = old.load_module('dat10_test_raw_cpt', old.BASE / 'raw_cpt_broader.py')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(old.TOKENIZER))
    tokenizer.encode_special_tokens = True

    with tempfile.TemporaryDirectory(prefix='dat10-v2-equivalence-') as td:
        root = Path(td)
        direct = root / 'direct-v1'
        recycled = root / 'recycled-v2'
        initialize(direct, source)
        initialize(recycled, source)
        direct_receipt = old.process_group(
            direct, INDEX, entry, tokenizer, raw_cpt, registry, parts,
            set(seen), set(protected),
        )
        lock = (recycled / 'process.lock').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recycled_receipt, worker_peak = new.invoke_worker(recycled, INDEX, lock.fileno(), set(seen))

        assert direct_receipt['counts'] == recycled_receipt['counts']
        assert direct_receipt['status'] == recycled_receipt['status'] == 'complete'
        assert direct_receipt['package'] == recycled_receipt['package'] == 'gasfluxes'
        assert direct_receipt['group_id'] == recycled_receipt['group_id']
        assert direct_receipt['source_categories'] == recycled_receipt['source_categories']
        assert direct_receipt['artifacts'] == recycled_receipt['artifacts']
        assert recycled_receipt['counts']['documents'] > 1
        assert recycled_receipt['counts']['rows'] > recycled_receipt['counts']['documents']

        # A crash residue is not a commit. Rebuilding state sees the exact committed group once.
        residue = recycled / '.staging' / 'interrupted-worker-residue'
        residue.mkdir()
        (residue / 'partial').write_text('not committed')
        rebuilt_seen, _ = new.base_seen()
        commits, totals = new.committed(recycled, rebuilt_seen)
        assert set(commits) == {INDEX}
        assert totals == recycled_receipt['counts']
        assert residue.exists()
        try:
            new.invoke_worker(recycled, INDEX, lock.fileno(), rebuilt_seen)
        except RuntimeError as error:
            assert 'already committed group' in str(error)
        else:
            raise AssertionError('duplicate worker launch was not rejected')

        print(json.dumps({
            'status': 'pass', 'representative_seeded_index': INDEX,
            'package': entry['name'], 'artifact_hashes_exact': True,
            'counts': recycled_receipt['counts'],
            'worker_peak_rss_kib': worker_peak,
            'atomic_commit_reconciles_once': True,
            'staging_residue_ignored': True,
            'duplicate_worker_rejected': True,
        }, sort_keys=True))


if __name__ == '__main__':
    main()
