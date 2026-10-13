"""Keep the WSL page cache below a cap so Windows keeps free memory.

WSL's ``autoMemoryReclaim`` only reclaims while the VM is idle: user CPU below
0.5% of all cores for ten minutes (``dropCache``) or three minutes
(``gradual``). With agent sessions or a training job running, that never
happens. Meanwhile, tens of GB of Linux page cache count as used memory on
the Windows side, and the campaign memory guard refuses or stops GPU work.

This service does what ``gradual`` does without the idle condition. Every few
seconds it asks the kernel to reclaim file cache above the cap through the
root cgroup's ``memory.reclaim``, which evicts the coldest pages first. After
a large reclaim it compacts memory, so free-page reporting can hand whole
blocks back to Windows. It needs root and runs as a system service; see
``scripts/night/install_cache_trim.sh``. This file is standard library only
and is installed as a standalone script.

Usage::

    sepalith-cache-trim [--cap-gib 8] [--step-gib 2] [--interval 10]
    sepalith-cache-trim --status        # no root needed; prints what it would do
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import errno
from pathlib import Path
import time

GIB = 1024**3
MIB = 1024**2
RECLAIM = Path("/sys/fs/cgroup/memory.reclaim")
COMPACT = Path("/proc/sys/vm/compact_memory")
MEMINFO = Path("/proc/meminfo")


def read_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo values in bytes."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields:
            values[name] = int(fields[0]) * (1024 if fields[1:] == ["kB"] else 1)
    return values


def file_cache(meminfo: Mapping[str, int]) -> int:
    return meminfo["Active(file)"] + meminfo["Inactive(file)"]


def reclaim_request(cache: int, cap: int, step: int, slack: int = 256 * MIB) -> int:
    """Bytes to ask the kernel to reclaim; 0 while the cache is within cap + slack."""
    if cache <= cap + slack:
        return 0
    return min(cache - cap, step)


def request_reclaim(amount: int, path: Path = RECLAIM) -> bool:
    """Write to memory.reclaim. EAGAIN means the kernel reclaimed less than asked."""
    try:
        path.write_text(f"{amount}\n")
    except OSError as error:
        if error.errno == errno.EAGAIN:
            return False
        raise
    return True


class Trimmer:
    def __init__(self, cap: int, step: int, *, compact_after: int = GIB, compact_every: float = 60.0,
                 read: Callable[[], str] | None = None, reclaim: Callable[[int], bool] | None = None,
                 compact: Callable[[], None] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.cap, self.step = cap, step
        self.compact_after, self.compact_every = compact_after, compact_every
        self.read = read or MEMINFO.read_text
        self.reclaim = reclaim or request_reclaim
        self.compact = compact or (lambda: COMPACT.write_text("1\n"))
        self.clock = clock
        self.pending_compaction = 0
        self.last_compaction = float("-inf")

    def tick(self) -> dict[str, int | bool]:
        before = file_cache(read_meminfo(self.read()))
        amount = reclaim_request(before, self.cap, self.step)
        complete = self.reclaim(amount) if amount else True
        after = file_cache(read_meminfo(self.read())) if amount else before
        self.pending_compaction += max(0, before - after)
        compacted = False
        if self.pending_compaction >= self.compact_after and self.clock() - self.last_compaction >= self.compact_every:
            self.compact()
            self.pending_compaction, self.last_compaction, compacted = 0, self.clock(), True
        return {"before": before, "requested": amount, "after": after, "complete": complete, "compacted": compacted}


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sepalith-cache-trim", description=__doc__.split("\n\n")[0])
    parser.add_argument("--cap-gib", type=float, default=8.0, help="page cache to keep (default: %(default)s)")
    parser.add_argument("--step-gib", type=float, default=2.0, help="largest reclaim per tick (default: %(default)s)")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between ticks (default: %(default)s)")
    parser.add_argument("--status", action="store_true", help="print the current cache and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    cap, step = int(args.cap_gib * GIB), int(args.step_gib * GIB)
    if args.status:
        cache = file_cache(read_meminfo(MEMINFO.read_text()))
        print(f"file cache {cache // MIB} MiB, cap {cap // MIB} MiB, "
              f"next request {reclaim_request(cache, cap, step) // MIB} MiB")
        return 0
    trimmer = Trimmer(cap, step)
    print(f"keeping page cache at or below {cap // MIB} MiB, up to {step // MIB} MiB per {args.interval:g} s",
          flush=True)
    while True:
        result = trimmer.tick()
        if result["requested"]:
            print(f"cache {int(result['before']) // MIB} -> {int(result['after']) // MIB} MiB"
                  f"{' (partial)' if not result['complete'] else ''}{', compacted' if result['compacted'] else ''}",
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
