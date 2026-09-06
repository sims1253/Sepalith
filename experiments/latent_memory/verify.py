"""Run CPU contracts and retain a machine-readable receipt; no pretrained weights."""
import argparse
import io
import time
import unittest

from .provenance import backend_identity, source_identity, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromName('experiments.latent_memory.test_contracts')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_json(args.output, {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
               'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'elapsed_seconds': time.monotonic() - start, 'backend': backend_identity(),
               'source_files': source_identity(), 'output': stream.getvalue(),
               'scope': 'CPU tiny random hybrid model, not banked B4 numerical validation',
               'learned_information_recovery': 'NOT TESTED', 'editing_improvement': 'NOT TESTED'})
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
