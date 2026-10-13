import datetime, hashlib, json, os, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parent
LEAD = ROOT.parent
PLAN = ROOT.parents[4]
PACKET = LEAD / 'r2-cpt450-from354-review-preparation-v2'
VERIFIER = LEAD / 'r2-cpt450-from354-cpu-review-root-v1'
ARCHIVE = pathlib.Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-recovery354-to450-cadence24-v1/full/checkpoint-450')
GUARD = pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt450-matched-eval-root-v1-host-supervision')

def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
def record(name, value):
    value['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(ROOT / name, value)

phase = 'waiting_for_cpu_verification'
record('launch.json', {'pid': os.getpid(), 'phase': phase, 'verifier_pid': 2721744, 'promotion_authorized': False})
try:
    deadline = time.monotonic() + 10800
    while not (VERIFIER / 'terminal.json').exists():
        assert time.monotonic() < deadline, 'verification observation deadline expired'
        time.sleep(15)
    assert read(VERIFIER / 'terminal.json')['exit_code'] == 0, 'checkpoint verification failed'
    while any(pathlib.Path(f'/proc/{pid}').exists() for pid in (2699989, 2700261, 2700314, 2700542, 2721744)):
        assert time.monotonic() < deadline, 'prior processes have not exited'
        time.sleep(2)
    assert sha(PACKET / 'artifact-manifest.json') == '7dec3555636919d0b77edb3d90b57e1eb78c277fc1e2fa7056e3d1f98c1b49c4'
    for item in read(PACKET / 'artifact-manifest.json')['files']:
        path = PACKET / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    review = read(PACKET / 'checkpoint-review.json')
    assert review['step'] == 450 and review['updates_verified'] == 96 and review['draws_verified'] == 1536 and review['cursor'] == 6144
    assert review['native_and_durable_payloads_verified'] is True
    binding = read(PACKET / 'binding.review.json')
    assert binding['checkpoint_step'] == 450 and binding['training_authorized'] is False
    model = pathlib.Path(binding['model_path'])
    manifest = read(model / 'campaign-manifest.json')
    assert sha(model / 'campaign-manifest.json') == sha(ARCHIVE / 'campaign-manifest.json') == review['checkpoint_manifest_sha256']
    assert all(binding['model_files'][name] == manifest['files'][name]['sha256'] for name in binding['model_files'])
    binding['status'] = 'admitted'
    write(PACKET / 'binding.admitted.json', binding)
    released = []
    for base in (model, ARCHIVE):
        for name in ('model.safetensors', 'optimizer.pt'):
            path = base / name
            before = path.stat()
            assert before.st_size == manifest['files'][name]['bytes']
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
            released.append(str(path))
    record('verified-checkpoint-cache-release.json', {'files': released, 'content_unchanged': True})
    phase = 'matched_cuda_evaluation'
    command = ['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python', '-B', str(LEAD / 'host-memory-guard-v4/cuda_host_guard.py'), '--command-json', str(PACKET / 'command.json'), '--output', str(GUARD), '--seconds', '2400', '--minimum-free-mib', '6144', '--admission-free-mib', '12288']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UNSLOTH_RETURN_LOGITS='0')
    with (ROOT / 'evaluation.log').open('x') as log:
        child = subprocess.Popen(command, cwd=PLAN, env=env, stdout=log, stderr=subprocess.STDOUT)
        record('evaluation-launch.json', {'guard_pid': child.pid, 'command': command, 'binding_sha256': sha(PACKET / 'binding.admitted.json')})
        code = child.wait()
    assert code == 0, 'matched evaluation guard failed'
    phase = 'metric_verification'
    command = read(PACKET / 'root-commands.json')['review_eval_manual_after_evaluation']
    with (ROOT / 'metric-review.log').open('x') as log:
        code = subprocess.run(command, cwd=PLAN, stdout=log, stderr=subprocess.STDOUT).returncode
    assert code == 0, 'exact fixture metric verification failed'
    output = pathlib.Path('/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-cpt450-from354-matched-root-v1')
    for name in ('anchor2k', '8k', '16k'):
        assert read(output / (name + '.json'))['saved_precision_audit']['fp32_tensors_restored'] == 85
    record('terminal.json', {'exit_code': 0, 'status': 'metrics_verified_requires_root_scientific_decision', 'promotion_authorized': False})
except Exception as error:
    record('terminal.json', {'exit_code': 1, 'phase': phase, 'status': 'stopped_without_retry', 'error': str(error), 'promotion_authorized': False})
    raise
