import datetime, hashlib, json, os, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parent
PACKET = ROOT.parent / 'r2-noop4100-parse-retry-preparation-v1'
def write(name, value):
    value['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

admission = json.loads((ROOT / 'admission.json').read_text())
assert admission['status'] == 'root_admitted_cpu_retry'
for item in admission['frozen_files']:
    assert hashlib.sha256(pathlib.Path(item['path']).read_bytes()).hexdigest() == item['sha256']
command = ['timeout', '--signal=TERM', '--kill-after=30s', '7200', 'bash', str(PACKET / 'commands/run_noop4100_all41.sh')]
with (ROOT / 'process.log').open('x') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    write('launch.json', {'controller_pid': os.getpid(), 'child_pid': child.pid, 'command': command, 'cores': [4, 6], 'cuda': False})
    code = child.wait()
write('terminal.json', {'exit_code': code, 'status': 'command_complete_requires_root_artifact_review' if code == 0 else 'failed_preserve_partial_outputs', 'training_admitted': False})
raise SystemExit(code)
