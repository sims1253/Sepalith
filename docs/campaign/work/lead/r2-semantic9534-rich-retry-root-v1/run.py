import datetime, hashlib, json, os, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parent
PACKET = ROOT.parent / 'r2-provider-parse-unavailable-preparation-v1/semantic9534-retry'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
def write(name, value):
    value['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

admission = json.loads((ROOT / 'admission.json').read_text())
for item in admission['frozen_files']:
    assert sha(pathlib.Path(item['path'])) == item['sha256']
manifest = json.loads((PACKET / 'source-manifest.json').read_text())
inputs = pathlib.Path(manifest['input_root'])
assert sha(inputs / 'manifest.json') == manifest['input_manifest_sha256']
data = json.loads((inputs / 'manifest.json').read_text())
assert data['provider_rows'] == 9534
assert [x['shard'] for x in data['shards']] == list(range(27, 41))
for item in data['shards']:
    info = item['provider']
    path = inputs / info['path']
    assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
assert sum(x['provider_rows'] for x in data['shards']) == 9534
write('preflight.json', {'status': 'pass', 'input_rows': 9534, 'input_shards': 14, 'cuda': False})
command = ['timeout', '--signal=TERM', '--kill-after=30s', '7200', 'bash', str(PACKET / 'commands/retry_14_shards.sh')]
with (ROOT / 'process.log').open('x') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    write('launch.json', {'controller_pid': os.getpid(), 'child_pid': child.pid, 'command': command, 'cores': [8, 10], 'cuda': False})
    code = child.wait()
write('terminal.json', {'exit_code': code, 'status': 'command_complete_requires_root_artifact_review' if code == 0 else 'failed_preserve_partial_outputs', 'training_admitted': False})
raise SystemExit(code)
