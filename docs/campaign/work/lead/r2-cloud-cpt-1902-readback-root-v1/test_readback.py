import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import readback


class ReadbackTests(unittest.TestCase):
    def make_tar(self, path, members):
        with tarfile.open(path, "w") as archive:
            for name, kind, payload in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.type = kind
                archive.addfile(info, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)

    def test_safe_extract_exact_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = b"opaque-state"
            tar_path = root / "good.tar"
            self.make_tar(tar_path, [("checkpoint-1902/optimizer.pt", tarfile.REGTYPE, payload)])
            expected = {"optimizer.pt": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
            self.assertEqual(len(readback.validate_member_table(tar_path, expected)), 1)
            self.assertEqual(readback.extract_and_verify(tar_path, root / "out", expected), expected)

    def test_rejects_traversal_link_unexpected_and_duplicate(self):
        cases = [
            [("checkpoint-1902/../bad", tarfile.REGTYPE, b"x")],
            [("checkpoint-1902/link", tarfile.SYMTYPE, b"")],
            [("checkpoint-1902/nope", tarfile.REGTYPE, b"x")],
            [("checkpoint-1902/a", tarfile.REGTYPE, b"x"), ("checkpoint-1902/a", tarfile.REGTYPE, b"x")],
        ]
        expected = {"a": {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}}
        with tempfile.TemporaryDirectory() as td:
            for index, members in enumerate(cases):
                path = Path(td) / f"bad-{index}.tar"
                self.make_tar(path, members)
                with self.assertRaises(ValueError):
                    readback.validate_member_table(path, expected)

    def test_safetensors_header_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.safetensors"
            header = json.dumps({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, separators=(",", ":")).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"12345678")
            result = readback.safetensors_header(path)
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["payload_bytes"], 8)
            self.assertEqual(result["dtype_counts"], {"F32": 1})


if __name__ == "__main__":
    unittest.main()
