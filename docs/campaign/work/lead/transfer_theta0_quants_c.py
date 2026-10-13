"""Stream accepted quants to the authorized notebook with bounded clean cache."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import time

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
HOST = 'm0hawk@192.168.178.40'
DEST = '/home/m0hawk/.local/share/sepalith-campaign-20260915/models/SFT-primary-step1000-quant-candidates-c'
review = json.loads((PLAN / 'docs/campaign/receipts/RUN-01-theta0-quant-c-lead-review.json').read_text())
assert review['status'] == 'quant_artifact_integrity_accepted_native_quality_pending'
started = time.monotonic()
results = []
for artifact in review['artifacts']:
    source = Path(artifact['path'])
    spec = {'path': DEST + '/' + source.name, 'sha256': artifact['sha256'], 'bytes': artifact['bytes']}
    remote = '''from pathlib import Path
import sys,os,json,hashlib,shutil
spec = SPEC
p=Path(spec['path']); partial=p.with_name(p.name+'.transfer-partial')
assert not p.exists() and not partial.exists(), 'Fresh destination required'
assert shutil.disk_usage(p.parents[1]).free > 20*(1<<30)
p.parent.mkdir(parents=True,exist_ok=True)
d=hashlib.sha256(); count=0
with partial.open('xb') as out:
 while True:
  block=sys.stdin.buffer.read(1<<20)
  if not block: break
  count+=len(block); assert count<=spec['bytes']; d.update(block); out.write(block)
 out.flush(); os.fsync(out.fileno())
assert count==spec['bytes'] and d.hexdigest()==spec['sha256'], 'Transfer identity mismatch'
with partial.open('rb') as inp:
 assert inp.read(4)==b'GGUF'
 os.posix_fadvise(inp.fileno(),0,0,os.POSIX_FADV_DONTNEED)
os.link(partial,p); partial.unlink()
fd=os.open(p.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)
print(json.dumps({'status':'transferred_hash_verified','path':str(p),'bytes':count,'sha256':d.hexdigest()}))
'''.replace('SPEC', repr(spec))
    proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
                             'python3', '-c', shlex.quote(remote)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = hashlib.sha256()
    copied = 0
    stage_start = time.monotonic()
    try:
        with source.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(1 << 20), b''):
                assert time.monotonic() - started < 580, 'Transfer deadline'
                proc.stdin.write(block)
                d.update(block)
                os.posix_fadvise(stream.fileno(), copied, len(block), os.POSIX_FADV_DONTNEED)
                copied += len(block)
                delay = copied / (16 * (1 << 20)) - (time.monotonic() - stage_start)
                if delay > 0:
                    time.sleep(delay)
            after = os.fstat(stream.fileno())
        proc.stdin.close()
        proc.stdin = None
        stdout, stderr = proc.communicate(timeout=max(1, 590 - (time.monotonic() - started)))
        assert proc.returncode == 0, stderr.decode(errors='replace')[-1000:]
        assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        assert copied == artifact['bytes'] and d.hexdigest() == artifact['sha256']
        result = json.loads(stdout)
        assert result['sha256'] == artifact['sha256'] and result['bytes'] == copied
        results.append(result)
        print(json.dumps(result), flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
receipt = {'task': 'RUN-01', 'owner': 'lead', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
           'status': 'notebook_transfer_verified_native_validation_pending', 'host': HOST,
           'seconds': time.monotonic() - started, 'artifacts': results,
           'scope': 'No model inference; fresh paths; old model artifacts preserved; 16MiB/s source read limit.'}
p = PLAN / 'docs/campaign/receipts/RUN-01-theta0-quant-c-notebook-transfer.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
