"""Page-cache trimmer logic with a fake /proc/meminfo; no root needed."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sepalith.ops.cache_trim import GIB, MIB, Trimmer, file_cache, read_meminfo, reclaim_request, request_reclaim


def meminfo(cache_bytes):
    half = cache_bytes // 2 // 1024
    return f"MemTotal: 49327760 kB\nActive(file): {half} kB\nInactive(file): {half} kB\nHugePages_Total: 0\n"


class CacheTrimTests(unittest.TestCase):
    def test_meminfo_parsing(self):
        values = read_meminfo(meminfo(40 * GIB))
        self.assertEqual(file_cache(values), 40 * GIB)
        self.assertEqual(values["HugePages_Total"], 0)

    def test_request_is_bounded_by_step_and_slack(self):
        self.assertEqual(reclaim_request(40 * GIB, 8 * GIB, 2 * GIB), 2 * GIB)
        self.assertEqual(reclaim_request(9 * GIB, 8 * GIB, 2 * GIB), GIB)
        self.assertEqual(reclaim_request(8 * GIB + 100 * MIB, 8 * GIB, 2 * GIB), 0)

    def test_trimmer_reaches_the_cap_and_compacts_after_a_gib(self):
        state = {"cache": 12 * GIB, "now": 0.0, "compactions": 0}

        def reclaim(amount):
            state["cache"] -= amount
            return True

        def compact():
            state["compactions"] += 1

        trimmer = Trimmer(8 * GIB, 2 * GIB, read=lambda: meminfo(state["cache"]), reclaim=reclaim, compact=compact,
                          clock=lambda: state["now"])
        first = trimmer.tick()
        self.assertEqual((first["requested"], first["compacted"]), (2 * GIB, True))
        state["now"] = 10
        second = trimmer.tick()
        self.assertEqual((second["requested"], second["compacted"]), (2 * GIB, False))  # 60 s rate limit
        self.assertEqual(trimmer.tick()["requested"], 0)
        self.assertEqual(state["cache"], 8 * GIB)
        state["now"] = 70
        self.assertTrue(trimmer.tick()["compacted"])
        self.assertEqual(state["compactions"], 2)

    def test_eagain_is_a_partial_reclaim(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "memory.reclaim"
            self.assertTrue(request_reclaim(123, target))
            self.assertEqual(target.read_text(), "123\n")
            with patch.object(Path, "write_text", side_effect=OSError(errno.EAGAIN, "again")):
                self.assertFalse(request_reclaim(1, target))
            with patch.object(Path, "write_text", side_effect=OSError(errno.EACCES, "denied")), \
                    self.assertRaises(PermissionError):
                request_reclaim(1, target)


if __name__ == "__main__":
    unittest.main()
