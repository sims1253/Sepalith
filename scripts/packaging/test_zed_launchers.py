import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PACKAGING = Path(__file__).resolve().parent
ROOT = PACKAGING.parents[1]
_old_path = list(sys.path)
try:
    sys.path.insert(0, str(PACKAGING))
    spec = importlib.util.spec_from_file_location('launchers', PACKAGING / 'zed_launchers.py')
    launchers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launchers)
    producer_spec = importlib.util.spec_from_file_location('manifest_producer', PACKAGING / 'manifest.py')
    producer = importlib.util.module_from_spec(producer_spec)
    producer_spec.loader.exec_module(producer)
finally:
    sys.path[:] = _old_path

PROFILE = dict(renderer='zeta2-v1', task='r-next-edit',
               tokenizerRevision='tokenizer:fixture', modelRevision='model:fixture')


def shared_cases():
    return json.loads((ROOT / 'tests/fixtures/release-manifests.json').read_text())['cases']


class LauncherTests(unittest.TestCase):
    def test_download_cache_and_native_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'; source.mkdir()
            server = source / 'llama-server'
            server.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_ARGS"\n')
            model = source / 'model.gguf'; model.write_bytes(b'fixture')
            def asset(p):
                return dict(name=p.name, url='https://example.com/'+p.name,
                            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), executable=p==server)
            manifest = dict(schema=1, build='b10453', modelProfile=PROFILE, model=asset(model), bundles=[dict(platform='linux',arch='x64',backend='cpu',server=server.name,files=[asset(server)])])
            out = root / 'out'
            launchers.generate(manifest, out)
            bin_dir = root / 'bin'; bin_dir.mkdir()
            curl = bin_dir / 'curl'
            curl.write_text('#!/usr/bin/env python3\nimport os,sys,shutil\nfrom pathlib import Path\nshutil.copyfile(Path(os.environ["TEST_SOURCE"])/sys.argv[-1].rsplit("/",1)[-1],sys.argv[sys.argv.index("--output")+1])\nwith open(os.environ["TEST_CALLS"],"a") as f:f.write("download\\n")\n')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], SEPALITH_HOME=str(root/'cache'), SEPALITH_BACKEND='cpu', TEST_SOURCE=str(source), TEST_ARGS=str(root/'args'), TEST_CALLS=str(root/'calls'))
            script = out / 'sepalith-linux-x64-cpu.sh'
            for executable in [script, out / 'sepalith.sh']:
                subprocess.run(['bash', str(executable)], env=env, check=True)
            self.assertEqual((root/'calls').read_text().count('download'), 2)
            args = (root/'args').read_text().splitlines()
            self.assertEqual(args[args.index('--alias')+1], 'sepalith')
            self.assertEqual(args[args.index('-ngl')+1], '0')
            self.assertEqual(args[args.index('--host')+1], '127.0.0.1')
            self.assertFalse(list((root/'cache').rglob('*.partial.*')))

    def test_shared_validation_fixtures_before_any_output_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in shared_cases():
                with self.subTest(name=case['name']):
                    output = root / case['name']
                    if case['valid']:
                        launchers.generate(case['manifest'], output)
                        self.assertTrue((output / 'sepalith.sh').is_file())
                    else:
                        with self.assertRaises(ValueError):
                            launchers.generate(case['manifest'], output)
                        self.assertFalse(output.exists(), 'Validation must precede output creation')

    def test_invalid_later_bundle_preserves_existing_output(self):
        case = next(case for case in shared_cases() if case['name'] == 'late_invalid_bundle')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            script = output / 'sepalith-linux-x64-cpu.sh'
            script.write_bytes(b'existing reviewed launcher')
            with self.assertRaises(ValueError):
                launchers.generate(case['manifest'], output)
            self.assertEqual(script.read_bytes(), b'existing reviewed launcher')
            self.assertEqual(list(output.iterdir()), [script])

    def test_corrupt_download_is_not_published_or_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'launchers'
            launchers.generate(shared_cases()[0]['manifest'], output)
            binary = root / 'bin'
            binary.mkdir()
            curl = binary / 'curl'
            curl.write_text('#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
                            'Path(sys.argv[sys.argv.index("--output")+1]).write_bytes(b"corrupt")\n')
            curl.chmod(0o755)
            cache = root / 'cache'
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], SEPALITH_HOME=str(cache))
            result = subprocess.run(['bash', str(output / 'sepalith-linux-x64-cpu.sh')],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Download verification failed', result.stderr)
            self.assertFalse(list(cache.rglob('*.partial.*')))
            self.assertFalse(list(cache.rglob('llama-server')))

    def test_generated_network_deadlines_and_bash_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in shared_cases():
                if not case['valid']:
                    continue
                output = root / case['name']
                launchers.generate(case['manifest'], output)
                for script in output.iterdir():
                    text = script.read_text()
                    if script.name != 'sepalith.sh' or case['manifest']['bundles'][0]['platform'] != 'win32':
                        self.assertIn('--connect-timeout 15 --max-time 1800 --max-redirs 5', text)
                    if script.suffix == '.sh':
                        subprocess.run(['bash', '-n', str(script)], check=True, timeout=5)

    def test_manifest_producer_requires_revisions_and_validates_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.gguf'
            model.write_bytes(b'tiny fake GGUF')
            bundle = root / 'bundle.json'
            valid = shared_cases()[0]['manifest']
            bundle.write_text(json.dumps(valid['bundles'][0]))
            output = root / 'release.json'
            command = [sys.executable, '-B', str(PACKAGING / 'manifest.py'), '--bundle', str(bundle),
                       '--model', str(model), '--model-url', 'https://example.com/model.gguf',
                       '--output', str(output)]
            missing = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse(output.exists())
            revisions = ['--model-revision', 'model:fixture', '--tokenizer-revision', 'tokenizer:fixture']
            subprocess.run(command + revisions, check=True, capture_output=True, timeout=5)
            produced = json.loads(output.read_text())
            self.assertEqual(produced['modelProfile'], PROFILE)
            self.assertEqual(produced['model']['sha256'], hashlib.sha256(model.read_bytes()).hexdigest())
            original = output.read_bytes()
            bad = dict(valid['bundles'][0], files=[])
            bundle.write_text(json.dumps(bad))
            invalid = subprocess.run(command + revisions, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_manifest_publication_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'release.json'
            output.write_bytes(b'previous reviewed manifest')
            with mock.patch.object(producer.os, 'replace', side_effect=OSError('simulated write failure')):
                with self.assertRaises(OSError):
                    producer.write_manifest(shared_cases()[0]['manifest'], output)
            self.assertEqual(output.read_bytes(), b'previous reviewed manifest')
            self.assertEqual(list(root.iterdir()), [output])

    def test_reject_unsafe_filename(self):
        with self.assertRaises(ValueError):
            launchers.validate_asset(dict(name='../escape', url='https://example.com', sha256='a'*64,bytes=1))


if __name__ == '__main__':
    unittest.main()
