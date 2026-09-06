import hashlib
from pathlib import Path
import tempfile
import unittest

from fetch import fetch


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'result'
        self.data = b'archived evidence'
        self.index = {'bucket': 'test/private', 'artifacts': [{
            'original_path': 'logs/example.log', 'object': 'sha256/example',
            'bytes': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}]}

    def download(self, bucket, files, **kwargs):
        files[0][1].write_bytes(self.data)

    def test_verified_download_and_existing_cache(self):
        self.assertEqual(self.target, fetch(self.index, 'logs/example.log', self.target, self.download))
        def unexpected(*args, **kwargs):
            self.fail('Verified cache should not download again')
        fetch(self.index, 'logs/example.log', self.target, unexpected)

    def test_bad_download_does_not_publish(self):
        def corrupt(bucket, files, **kwargs):
            files[0][1].write_bytes(b'wrong bytes')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            fetch(self.index, 'logs/example.log', self.target, corrupt)
        self.assertFalse(self.target.exists())

    def test_existing_unrelated_file_is_preserved(self):
        self.target.write_bytes(b'keep this')
        with self.assertRaises(ValueError):
            fetch(self.index, 'logs/example.log', self.target, self.download)
        self.assertEqual(b'keep this', self.target.read_bytes())

    def test_unknown_path_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch(self.index, 'not indexed', self.target, self.download)


if __name__ == '__main__':
    unittest.main()
