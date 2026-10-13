"""Bound clean file cache for owned campaign data and checkpoint payloads only."""
import os, pathlib, threading, time

def start(native, archive, data):
    stop = threading.Event()
    errors = []
    def loop():
        try:
            while not stop.is_set():
                paths = list(data)
                for root in (native, archive / 'full'):
                    for directory in root.glob('*checkpoint-*'):
                        if directory.is_dir() and not directory.is_symlink():
                            paths.extend(directory / name for name in ('model.safetensors', 'optimizer.pt'))
                for p in paths:
                    try:
                        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
                    except FileNotFoundError:
                        continue
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                stop.wait(1)
        except BaseException as exc:
            errors.append(repr(exc))
    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    return stop, worker, errors
