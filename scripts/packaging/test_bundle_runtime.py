"""Runtime bundler tests use tiny archives and mocked HTTP; no real downloads."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

import bundle_runtime as runtime

BASE_URL = 'https://github.com/example/runtime/releases/download/runtime-b10453-v1'


class Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url
        self.read_sizes = []

    def geturl(self):
        return self.url

    def read1(self, size):
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError('Unbounded network read')
        return super().read1(size)


class BundleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'

    def tar(self, entries=None):
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode='w:gz') as archive:
            for name, content, kind in entries or [('bin/llama-server', b'server', 'file'),
                                                  ('bin/libllama.so.1', b'library', 'file'),
                                                  ('bin/libllama.so', 'libllama.so.1', 'link'),
                                                  ('LICENSE', b'license', 'file')]:
                entry = tarfile.TarInfo(name)
                if kind == 'link':
                    entry.type, entry.linkname = tarfile.SYMTYPE, content
                    archive.addfile(entry)
                elif kind == 'hardlink':
                    entry.type, entry.linkname = tarfile.LNKTYPE, content
                    archive.addfile(entry)
                else:
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
        return out.getvalue()

    def zip(self, entries=None):
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            for name, content, mode in entries or [('bin/llama-server.exe', b'server', stat.S_IFREG),
                                                   ('bin/llama.dll', b'library', stat.S_IFREG)]:
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = (mode | 0o644) << 16
                archive.writestr(entry, content)
        return out.getvalue()

    def serve(self, archive, target='linux-x64-cpu', change=None):
        name = f'llama-{runtime.PIN}-bin-{runtime.ASSETS[target]}'
        url = f'https://github.com/ggml-org/llama.cpp/releases/download/{runtime.PIN}/{name}'
        release = {'tag_name': runtime.PIN, 'assets': [{'name': name, 'browser_download_url': url,
                   'digest': 'sha256:' + hashlib.sha256(archive).hexdigest(), 'size': len(archive)}]}
        if change:
            change(release)
        metadata_url = f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{runtime.PIN}'
        responses = {metadata_url: json.dumps(release).encode(), url: archive}
        calls, streams = [], []

        def open_url(request, *, timeout):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 30)
            calls.append(request.full_url)
            stream = Response(responses[request.full_url], request.full_url)
            streams.append(stream)
            return stream

        opener = unittest.mock.Mock()
        opener.open.side_effect = open_url
        return patch('bundle_runtime.urllib.request.build_opener', return_value=opener), calls, streams

    def test_tar_bundle_verifies_streams_and_materializes_safe_library_links(self):
        mock, calls, streams = self.serve(self.tar())
        with mock, patch.object(Path, 'read_bytes', side_effect=AssertionError('Unbounded file read')):
            result = runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
        self.assertEqual(2, len(calls))
        self.assertTrue(all(size == runtime.CHUNK for stream in streams for size in stream.read_sizes))
        self.assertEqual({'llama-server', 'libllama.so', 'libllama.so.1', 'LICENSE'}, {a['name'] for a in result['files']})
        self.assertFalse((self.output / 'libllama.so').is_symlink())
        self.assertEqual(b'library', (self.output / 'libllama.so').read_bytes())
        for asset in result['files']:
            payload = (self.output / asset['name']).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), asset['sha256'])
            self.assertEqual(len(payload), asset['bytes'])
            self.assertEqual(payload, (self.output / 'release-assets' / ('linux-x64-cpu--' + asset['name'])).read_bytes())
        self.assertEqual(result, json.loads((self.output / 'bundle.json').read_text()))
        self.assertEqual(runtime.PIN, json.loads((self.output / 'receipt.json').read_text())['build'])
        self.assertEqual([], list(self.root.glob('.runtime-bundle-*')))

    def test_zip_bundle_and_executable(self):
        mock, _, _ = self.serve(self.zip(), 'win32-x64-cpu')
        with mock:
            result = runtime.bundle('win32-x64-cpu', self.output, BASE_URL)
        self.assertEqual('llama-server.exe', result['server'])
        self.assertTrue(next(a['executable'] for a in result['files'] if a['name'] == result['server']))

    def test_missing_digest_wrong_pin_url_and_size_fail_before_archive_download(self):
        changes = [lambda r: r['assets'][0].pop('digest'),
                   lambda r: r['assets'][0].update(digest='md5:abc'),
                   lambda r: r.update(tag_name='new-release'),
                   lambda r: r['assets'][0].update(browser_download_url='https://example.com/unpinned'),
                   lambda r: r['assets'][0].update(size=True),
                   lambda r: r['assets'].append(dict(r['assets'][0]))]
        for change in changes:
            with self.subTest(change=change):
                mock, calls, _ = self.serve(self.tar(), change=change)
                with mock, self.assertRaises(ValueError):
                    runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
                self.assertEqual(1, len(calls))
                self.assertFalse(self.output.exists())
                self.assertEqual([], list(self.root.glob('.runtime-bundle-*')))

    def test_digest_and_archive_size_mismatch_never_publish_output(self):
        for change in (lambda r: r['assets'][0].update(digest='sha256:' + '0' * 64),
                       lambda r: r['assets'][0].update(size=r['assets'][0]['size'] - 1),
                       lambda r: r['assets'][0].update(size=r['assets'][0]['size'] + 1)):
            mock, _, _ = self.serve(self.tar(), change=change)
            with mock, self.assertRaises(ValueError):
                runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
            self.assertFalse(self.output.exists())

    def test_archive_traversal_links_and_colliding_names_fail_atomically(self):
        bad_tars = [[('../escape', b'bad', 'file')],
                    [('/absolute', b'bad', 'file')],
                    [('bin/libbad.so', '../../outside', 'link')],
                    [('bin/libbad.so', 'missing', 'link')],
                    [('bin/libbad.so', 'libbad.so', 'link')],
                    [('a/llama-server', b'one', 'file'), ('b/llama-server', b'two', 'file')],
                    [('bin/llama-server', b'one', 'file'), ('bin/llama-server', b'two', 'file')],
                    [('bin/llama-server', b'one', 'file'), ('bin/LIB.so', b'a', 'file'), ('bin/lib.so', b'b', 'file')]]
        for entries in bad_tars:
            with self.subTest(entries=entries):
                mock, _, _ = self.serve(self.tar(entries))
                with mock, self.assertRaises(ValueError):
                    runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
                self.assertFalse(self.output.exists())
                self.assertEqual([], list(self.root.glob('.runtime-bundle-*')))
        for entries in [[('bin/lib.dll', '../outside', stat.S_IFLNK)],
                        [('..\\escape', b'bad', stat.S_IFREG)]]:
            mock, _, _ = self.serve(self.zip(entries), 'win32-x64-cpu')
            with mock, self.assertRaises(ValueError):
                runtime.bundle('win32-x64-cpu', self.output, BASE_URL)
            self.assertFalse(self.output.exists())
        self.assertFalse((self.root / 'escape').exists())

    def test_missing_server_and_extraction_limit_never_publish_output(self):
        mock, _, _ = self.serve(self.tar([('lib.so', b'library', 'file')]))
        with mock, self.assertRaisesRegex(ValueError, 'no llama-server'):
            runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
        mock, _, _ = self.serve(self.tar())
        with mock, patch.object(runtime, 'MAX_BUNDLE_BYTES', 3), self.assertRaisesRegex(ValueError, 'size limit'):
            runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
        self.assertFalse(self.output.exists())
        # Limits also apply to ignored members before tar iteration skips
        # their compressed bodies, not just files chosen for the bundle.
        mock, _, _ = self.serve(self.tar([('ignored.dat', b'oversized', 'file')]))
        with mock, patch.object(runtime, 'MAX_BUNDLE_BYTES', 3), self.assertRaisesRegex(ValueError, 'size limit'):
            runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
        self.assertFalse(self.output.exists())

    def test_invalid_target_urls_timeout_and_existing_output_require_no_network(self):
        with patch('bundle_runtime.urllib.request.build_opener') as network:
            for target, url, timeout in [('linux-mips-cpu', BASE_URL, 30),
                                         ('linux-x64-cpu', 'http://github.com/a/b/releases/download/x', 30),
                                         ('linux-x64-cpu', BASE_URL + '?x=1', 30),
                                         ('linux-x64-cpu', BASE_URL + '#tag', 30),
                                         ('linux-x64-cpu', 'https://user@github.com/a/b/releases/download/x', 30),
                                         ('linux-x64-cpu', BASE_URL, float('nan'))]:
                with self.subTest(target=target, url=url, timeout=timeout), self.assertRaises(ValueError):
                    runtime.bundle(target, self.output, url, timeout=timeout)
            self.output.mkdir()
            marker = self.output / 'preserved'
            marker.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
            self.assertEqual('existing', marker.read_text())
            network.assert_not_called()

    def test_deadline_and_metadata_limit_abort_before_publication(self):
        mock, _, _ = self.serve(self.tar())
        with mock, patch.object(runtime.time, 'monotonic', side_effect=[0, 0, 2]), self.assertRaises(TimeoutError):
            runtime.bundle('linux-x64-cpu', self.output, BASE_URL, timeout=1)
        self.assertFalse(self.output.exists())
        mock, _, _ = self.serve(self.tar())
        with mock, patch.object(runtime, 'MAX_METADATA_BYTES', 1), self.assertRaisesRegex(ValueError, 'transfer limit'):
            runtime.bundle('linux-x64-cpu', self.output, BASE_URL)
        self.assertFalse(self.output.exists())

    def test_redirect_to_http_is_rejected(self):
        handler = runtime._HTTPSRedirect()
        request = urllib.request.Request('https://github.com/a')
        with self.assertRaises(ValueError):
            handler.redirect_request(request, None, 302, 'redirect', {}, 'http://example.com/asset')


if __name__ == '__main__':
    unittest.main()
