"""Small jobs exercise the complete candidate recipe; never load a real model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'packages/sepalith/src'), str(ROOT / 'scripts/migration'),
                str(ROOT / 'experiments/eval')]
from sepalith.runner import Runner
from prepare_v1c import INCLUDES, recipe, freeze
from v1c_artifacts import digest
import spec_bench
import latency_load

FAKE = '''import latency_load
class Server:
    def __init__(self, *a, **kw):
        kw['log_path'].write_text('fake server; no model loaded\\n')
    def start(self): pass
    def stop(self): pass
latency_load.spec_bench.SpecServer = Server
def stream(*a, **kw):
    return dict(ttft_ms=10.0, tokens=3, completed=True, aborted=False,
                starved=False, wall_ms=30.0, prompt_n=20, prompt_ms=10.0,
                predicted_ms=20.0, n_predicted=3, error=None)
latency_load.stream_once = stream
latency_load.main()
'''


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
        for rel in INCLUDES:
            dst = self.repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if (ROOT / rel).is_dir():
                shutil.copytree(ROOT / rel, dst, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copyfile(ROOT / rel, dst)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        traces = [{'ctx_class': cc, 'trace_id': f'{i}-{cc}', 'prompt': f'prompt {i}',
                   'target': 'answer>>>>>>> UPDATED'} for cc in ('2k', '8k') for i in range(64)]
        (self.assets / 'traces.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in traces))
        self.runner = Runner(self.root / 'state')

    def run_fake(self, code=FAKE):
        (self.repo / 'fake.py').write_text(code)
        snapshot = self.runner.snapshot(self.repo, [*INCLUDES, 'fake.py'])
        candidate = recipe(snapshot, self.assets, self.root / 'archive', sys.executable,
                           [{'path': str(self.assets / 'traces.jsonl'),
                             'sha256': digest(self.assets / 'traces.jsonl')}])
        candidate['steps'][1]['argv'][1] = '{source}/fake.py'
        self.runner.enqueue(candidate)
        self.assertIsNone(self.runner.run_next())
        self.runner.resume()
        attempt = self.runner.run_next()
        return self.runner.root / 'attempts' / attempt

    def test_complete_recipe_and_verified_archive(self):
        run = self.run_fake()
        self.assertEqual('succeeded', self.runner.plan()['jobs'][0]['status'])
        verdict = json.loads((run / 'verdict.json').read_text())
        self.assertEqual('MEASUREMENT-COMPLETE', verdict['verdict'])
        self.assertEqual('NOT-ASSESSED', verdict['adoption'])
        receipt = json.loads((run / 'archive.json').read_text())
        for entry in receipt['files']:
            self.assertEqual(entry['sha256'], digest(Path(receipt['path']) / entry['path']))
        self.assertEqual({'ttft': 100, 'sweep': 140},
                         json.loads((run / 'evaluation.json').read_text())['counts'])
        self.assertIsNone(self.runner.run_next())

    def test_negative_science_still_archives_successful_execution(self):
        run = self.run_fake(FAKE.replace('starved=False', 'starved=True'))
        self.assertEqual('succeeded', self.runner.plan()['jobs'][0]['status'])
        self.assertEqual('INCONCLUSIVE', json.loads((run / 'verdict.json').read_text())['verdict'])
        self.assertTrue((run / 'archive.json').exists())

    def test_failure_blocks_evaluation_archive_and_requires_retry(self):
        run = self.run_fake('raise SystemExit(9)')
        self.assertEqual('failed', self.runner.plan()['jobs'][0]['status'])
        self.assertFalse((run / 'archive.json').exists())
        self.assertIsNone(self.runner.run_next())
        self.runner.retry('v1c-v7-runner-20260906')
        second = self.runner.run_next()
        self.assertNotEqual(run.name, second)
        self.assertEqual(2, len(self.runner.plan()['attempts']))

    def test_missing_rows_fail_even_with_successful_measurement_command(self):
        run = self.run_fake(FAKE + "\nfrom pathlib import Path\np=Path(__file__).parents[1]/'measurement/results_v1c_ttft.jsonl'\np.write_text(p.read_text().splitlines()[0]+'\\n')\n")
        self.assertEqual('failed', self.runner.plan()['jobs'][0]['status'])
        self.assertFalse((run / 'archive.json').exists())

    def test_freeze_preserves_uncommitted_input_and_library_names(self):
        source = self.assets / 'traces.jsonl'
        alias = self.assets / 'alias'
        alias.symlink_to(source)
        bundle, inputs = freeze(self.root / 'frozen', {'data': source, 'runtime/lib.so.0': alias})
        source.write_text('later mutable bytes')
        for item in inputs:
            self.assertEqual(item['sha256'], digest(Path(item['path'])))
        self.assertFalse((bundle / 'runtime/lib.so.0').is_symlink())

    def test_preparation_cli_binds_inputs_without_dispatch(self):
        runtime = self.root / 'runtime'
        runtime.mkdir()
        shutil.copyfile('/usr/bin/true', runtime / 'llama-server')
        (runtime / 'llama-server').chmod(0o755)
        model = self.root / 'fake.gguf'
        model.write_text('fake model; never executed')
        output = self.root / 'bound.json'
        state = self.root / 'prepared-state'
        subprocess.run([sys.executable, str(ROOT / 'scripts/migration/prepare_v1c.py'),
                        '--state', str(state), '--assets', str(self.root / 'frozen'),
                        '--archive', str(self.root / 'archive'), '--model', str(model),
                        '--traces', str(self.assets / 'traces.jsonl'),
                        '--server-dir', str(runtime), '--output', str(output)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bound = json.loads(output.read_text())
        self.assertTrue(bound['inputs'])
        for item in bound['inputs']:
            self.assertEqual(item['sha256'], digest(Path(item['path'])))
        self.assertNotIn(str(model), json.dumps(bound))
        plan = Runner(state).plan()
        self.assertTrue(plan['paused'])
        self.assertEqual([], plan['jobs'])
        self.assertEqual([], plan['attempts'])

    def test_foreground_server_does_not_start_new_session(self):
        server = spec_bench.SpecServer('model', 18431, [], log_path=self.root / 'server.log',
                                       server='/explicit/server', foreground=True)
        self.assertEqual('/explicit/server', server.cmd()[0])
        with patch.object(spec_bench, 'port_open', return_value=False), \
                patch.object(spec_bench.subprocess, 'Popen') as launch:
            launch.return_value.poll.return_value = 9
            with self.assertRaises(RuntimeError):
                server.start(ready_timeout=1)
            self.assertFalse(launch.call_args.kwargs['start_new_session'])

    def test_main_stops_server_when_readiness_fails(self):
        with patch.object(sys, 'argv', ['latency_load', '--traces', str(self.assets / 'traces.jsonl'),
                                       '--out', str(self.root / 'out'), '--foreground']), \
                patch.object(spec_bench, 'SpecServer') as server:
            server.return_value.start.side_effect = RuntimeError('readiness failed')
            with self.assertRaisesRegex(RuntimeError, 'readiness failed'):
                latency_load.main()
            server.return_value.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
