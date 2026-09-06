#!/usr/bin/env python3
"""Normalize a verified b10453 release into an atomically published runtime bundle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import time
from urllib.parse import urlsplit
import urllib.request
import zipfile

if __package__:
    from .release_manifest import validate_asset, validate_filename, validate_https_url
else:
    from release_manifest import validate_asset, validate_filename, validate_https_url

PIN = 'b10453'
ASSETS = {
    'linux-x64-cpu': 'ubuntu-x64.tar.gz',
    'linux-arm64-cpu': 'ubuntu-arm64.tar.gz',
    'linux-x64-vulkan': 'ubuntu-vulkan-x64.tar.gz',
    'linux-arm64-vulkan': 'ubuntu-vulkan-arm64.tar.gz',
    'win32-x64-cpu': 'win-cpu-x64.zip',
    'win32-arm64-cpu': 'win-cpu-arm64.zip',
    'win32-x64-vulkan': 'win-vulkan-x64.zip',
    'darwin-x64-metal': 'macos-x64.tar.gz',
    'darwin-arm64-metal': 'macos-arm64.tar.gz',
}
CHUNK = 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 ** 3
MAX_BUNDLE_BYTES = 8 * 1024 ** 3
MAX_MEMBERS = 10000


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, destination, limit, deadline):
    """Bound memory, total transfer time and bytes; hash bytes as they arrive."""
    validate_https_url(url)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Runtime download deadline exceeded')
    opener = urllib.request.build_opener(_HTTPSRedirect())
    request = urllib.request.Request(url, headers={'User-Agent': f'sepalith-runtime-bundler/{PIN}'})
    digest, total = hashlib.sha256(), 0
    with opener.open(request, timeout=min(30, remaining)) as response, destination.open('xb') as output:
        validate_https_url(response.geturl())
        # HTTPResponse.read1 performs at most one underlying buffered read,
        # allowing the overall deadline to be checked during slow transfers.
        read = getattr(response, 'read1', response.read)
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Runtime download deadline exceeded')
            chunk = read(CHUNK)
            if time.monotonic() >= deadline:
                raise TimeoutError('Runtime download deadline exceeded')
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError('Download exceeds its declared size or transfer limit')
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest(), total


def _base_url(value):
    validate_https_url(value)
    parsed = urlsplit(value)
    if (parsed.netloc != 'github.com' or parsed.query or parsed.fragment
            or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/releases/download/[A-Za-z0-9_.-]+/?', parsed.path)):
        raise ValueError('base-url must name a pinned GitHub Release without a query or fragment')
    if any(part in ('.', '..') for part in parsed.path.split('/')):
        raise ValueError('base-url contains an invalid path segment')
    return value.rstrip('/')


def _archive_path(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or '\0' in name or path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def _wanted(name, server):
    return name == server or '.so' in name or Path(name).suffix in ('.dll', '.dylib') or name.startswith('LICENSE')


def _members(archive, is_zip):
    """Validate all archive entries; resolve safe tar links without extracting them."""
    entries = archive.infolist() if is_zip else archive
    indexed = {}
    unpacked_bytes = 0
    for index, entry in enumerate(entries):
        if index >= MAX_MEMBERS:
            raise ValueError('Archive contains too many entries')
        path = _archive_path(entry.filename if is_zip else entry.name)
        if is_zip:
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError('ZIP links and special files are not supported')
            directory = entry.is_dir()
        else:
            if not (entry.isfile() or entry.isdir() or entry.issym() or entry.islnk()):
                raise ValueError('Archive special files are not supported')
            directory = entry.isdir()
        unpacked_bytes += entry.file_size if is_zip else entry.size
        if unpacked_bytes > MAX_BUNDLE_BYTES:
            raise ValueError('Archive exceeds its unpacked size limit')
        if path in indexed:
            raise ValueError(f'Duplicate archive path: {path}')
        indexed[path] = (entry, directory)

    def regular(path, seen):
        if path in seen or path not in indexed:
            raise ValueError('Archive link is cyclic or has a missing target')
        entry, directory = indexed[path]
        if directory:
            raise ValueError('Archive link points to a directory')
        if not is_zip and (entry.issym() or entry.islnk()):
            target = _archive_path(entry.linkname)
            if entry.issym():
                target = path.parent / target
            return regular(target, seen | {path})
        return entry

    result = []
    for path, (entry, directory) in indexed.items():
        if not directory:
            resolved = regular(path, set())
            result.append((path.name, resolved))
    return result


def _materialize(archive_path, name, stage, key, base_url, server):
    is_zip = name.endswith('.zip')
    factory = zipfile.ZipFile if is_zip else tarfile.open
    files, names, total = [], set(), 0
    release_assets = stage / 'release-assets'
    release_assets.mkdir()
    with factory(archive_path) as archive:
        for filename, entry in sorted(_members(archive, is_zip), key=lambda pair: pair[0]):
            if not _wanted(filename, server):
                continue
            validate_filename(filename)
            if filename.lower() in names:
                raise ValueError(f'Duplicate archive basename: {filename}')
            names.add(filename.lower())
            size = entry.file_size if is_zip else entry.size
            total += size
            if size <= 0 or total > MAX_BUNDLE_BYTES:
                raise ValueError('Runtime file is empty or bundle exceeds its size limit')
            dest = stage / filename
            digest, copied = hashlib.sha256(), 0
            source = archive.open(entry) if is_zip else archive.extractfile(entry)
            with source, dest.open('xb') as output:
                while chunk := source.read(CHUNK):
                    copied += len(chunk)
                    if copied > size:
                        raise ValueError('Archive entry exceeds its declared size')
                    digest.update(chunk)
                    output.write(chunk)
            if copied != size:
                raise ValueError('Truncated archive entry')
            if filename == server:
                dest.chmod(0o755)
            os.link(dest, release_assets / f'{key}--{filename}')
            asset = dict(name=filename, url=f'{base_url}/{key}--{filename}', sha256=digest.hexdigest(),
                         bytes=copied, executable=filename == server)
            validate_asset(asset)
            files.append(asset)
    if not any(f['name'] == server for f in files):
        raise ValueError('Archive has no llama-server')
    return files


def bundle(key, output, base_url, *, timeout=1800):
    if not isinstance(key, str) or key not in ASSETS:
        raise ValueError('Unsupported runtime target')
    base_url = _base_url(base_url)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be a positive finite number of seconds')
    output = Path(output)
    if os.path.lexists(output):
        raise ValueError('Output already exists; choose a new bundle directory')
    platform, arch, backend = key.split('-')
    name = f'llama-{PIN}-bin-{ASSETS[key]}'
    expected_url = f'https://github.com/ggml-org/llama.cpp/releases/download/{PIN}/{name}'
    deadline = time.monotonic() + timeout
    # Stage on the destination filesystem so only a complete bundle is published.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-bundle-', dir=output.parent) as tmp:
        tmp = Path(tmp)
        metadata = tmp / 'release.json'
        _download(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{PIN}',
                  metadata, MAX_METADATA_BYTES, deadline)
        release = json.loads(metadata.read_text())
        if not isinstance(release, dict) or release.get('tag_name') != PIN or not isinstance(release.get('assets'), list):
            raise ValueError('Release metadata does not match the pinned build')
        matches = [a for a in release['assets'] if isinstance(a, dict) and a.get('name') == name]
        if len(matches) != 1:
            raise ValueError('Pinned release must contain exactly one matching runtime asset')
        asset = matches[0]
        if not isinstance(asset.get('digest'), str) or not re.fullmatch(r'sha256:[a-f0-9]{64}', asset['digest']):
            raise ValueError('Pinned upstream asset requires a SHA256 digest')
        if asset.get('browser_download_url') != expected_url:
            raise ValueError('Upstream asset URL does not match the pinned build and asset')
        size = asset.get('size')
        if type(size) is not int or not 0 < size <= MAX_ARCHIVE_BYTES:
            raise ValueError('Upstream archive size is missing or exceeds the transfer limit')
        archive = tmp / name
        digest, downloaded = _download(expected_url, archive, size, deadline)
        if downloaded != size or asset['digest'] != 'sha256:' + digest:
            raise ValueError('Upstream archive size or checksum mismatch')
        stage = tmp / 'bundle'
        stage.mkdir()
        server = 'llama-server.exe' if platform == 'win32' else 'llama-server'
        files = _materialize(archive, name, stage, key, base_url, server)
        result = dict(platform=platform, arch=arch, backend=backend, server=server, files=files)
        (stage / 'bundle.json').write_text(json.dumps(result, indent=2) + '\n')
        (stage / 'receipt.json').write_text(json.dumps(dict(build=PIN, upstream=expected_url, sha256=digest), indent=2) + '\n')
        if os.path.lexists(output):
            raise ValueError('Output appeared during bundling; refusing to replace it')
        stage.rename(output)
        return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('target', choices=ASSETS)
    ap.add_argument('output', type=Path)
    ap.add_argument('--base-url', required=True, help='Pinned GitHub Release download URL, e.g. https://github.com/OWNER/REPO/releases/download/runtime-b10453-v1')
    ap.add_argument('--timeout', type=float, default=1800, help='Total network deadline in seconds (socket waits capped at 30 seconds)')
    args = ap.parse_args()
    bundle(args.target, args.output, args.base_url, timeout=args.timeout)
