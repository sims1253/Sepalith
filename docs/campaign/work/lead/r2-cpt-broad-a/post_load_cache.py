"""Release clean pages for one verified model after its owned loader closes it."""
from pathlib import Path
import os
import subprocess


def owned_processes(root_pid):
    pending, found = [root_pid], []
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        proc = Path('/proc') / str(pid)
        if not proc.exists():
            continue
        found.append(pid)
        try:
            pending.extend(int(x) for x in (proc / 'task' / str(pid) / 'children').read_text().split())
        except FileNotFoundError:
            pass
    return found


def model_still_open(path, root_pid):
    target = str(path.resolve())
    for pid in owned_processes(root_pid):
        proc = Path('/proc') / str(pid)
        try:
            if any(line.rstrip().endswith(target) for line in (proc / 'maps').read_text().splitlines()):
                return True
            for fd in (proc / 'fd').iterdir():
                try:
                    if str(fd.resolve(strict=True)) == target:
                        return True
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            continue
    return False


def release_after_load(path, process_log, root_pid, *, compact=True, attempts=None):
    """Return None until the marker exists and owned processes closed the file."""
    path = Path(path)
    process_log = Path(process_log)
    with process_log.open('rb') as stream:
        stream.seek(max(0, process_log.stat().st_size - 32768))
        tail = stream.read()
    if b'Unsloth 2026.8.18 patched 42 layers' not in tail:
        return None
    if model_still_open(path, root_pid):
        return None
    if attempts is not None:
        if attempts:
            return None
        attempts.add(str(path))
    before = path.stat()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('model file metadata changed during advisory cache release')
    result = {'path': str(path), 'bytes': before.st_size, 'operation': 'POSIX_FADV_DONTNEED',
              'no_owned_process_mapping_or_descriptor': True, 'file_metadata_unchanged': True,
              'scope': 'One clean model file after GPU load; no file content or persistent setting changed. One-shot VM compaction can affect other WSL processes latency.'}
    if compact:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        code = ("from pathlib import Path\n"
                "if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != " + repr(boot)
                + ": raise SystemExit('WSL boot identity mismatch')\n"
                "Path('/proc/sys/vm/compact_memory').write_text('1\\n')")
        completed = subprocess.run(['/mnt/c/Windows/System32/wsl.exe', '--distribution', 'Ubuntu-22.04',
                                    '--user', 'root', '--exec', '/usr/bin/python3', '-c', code],
                                   capture_output=True, text=True, timeout=5)
        result['one_shot_compaction_exit_code'] = completed.returncode
        if completed.returncode:
            raise RuntimeError('post-load one-shot compaction failed')
    return result
