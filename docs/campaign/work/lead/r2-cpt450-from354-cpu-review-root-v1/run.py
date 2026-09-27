import datetime, hashlib, json, os, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parent
LEAD = ROOT.parent
PACKET = LEAD / 'r2-cpt450-from354-review-preparation-v2'
ACTIVE = LEAD / 'r2-cpt354-to450-continuation-root-v1'
MANIFEST = '7dec3555636919d0b77edb3d90b57e1eb78c277fc1e2fa7056e3d1f98c1b49c4'

def write(name, value):
    value['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def verify_packet():
    raw = (PACKET / 'artifact-manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST
    for item in json.loads(raw)['files']:
        path = PACKET / item['path']
        assert path.stat().st_size == item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']

verify_packet()
write('launch.json', {'pid': os.getpid(), 'status': 'waiting_for_exact_training_terminal', 'training_controller_pid': 2699989, 'cuda_authorized': False})
deadline = time.monotonic() + 10800
try:
    while not (ACTIVE / 'terminal.json').exists():
        assert time.monotonic() < deadline, 'training observation deadline expired; do not restart'
        time.sleep(15)
    terminal = json.loads((ACTIVE / 'terminal.json').read_text())
    assert terminal['exit_code'] == 0, 'training failed; no verification or retry launched'
    while any(pathlib.Path(f'/proc/{pid}').exists() for pid in (2699989, 2700261, 2700314, 2700542)):
        assert time.monotonic() < deadline, 'training processes have not exited'
        time.sleep(2)
    verify_packet()
    command = ['timeout', '--signal=TERM', '--kill-after=30s', '1800', 'taskset', '-c', '12,14'] + json.loads((PACKET / 'root-commands.json').read_text())['prepare_manual_after_terminal']
    with (ROOT / 'verification.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write('verification-launch.json', {'pid': child.pid, 'command': command, 'cuda_authorized': False})
        code = child.wait()
    write('terminal.json', {'exit_code': code, 'status': 'cpu_review_complete_requires_root_admission' if code == 0 else 'cpu_review_failed', 'cuda_launched': False})
except Exception as error:
    write('terminal.json', {'exit_code': 1, 'status': 'stopped_requires_root_review', 'error': str(error), 'cuda_launched': False})
    raise
