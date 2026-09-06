"""Strict schema-1 release validation shared by manifest and launcher producers."""
import ipaddress
import math
import re
from urllib.parse import urlsplit

MAX_SAFE_INTEGER = 9007199254740991
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
_HASH = re.compile(r'[a-f0-9]{64}\Z')
_DNS_LABEL = re.compile(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z')
_DEVICES = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
            *(f'LPT{i}' for i in range(1, 10))}


def _safe_positive_integer(value):
    # JSON numbers such as 7 and 7.0 are indistinguishable to the TS consumer.
    return (type(value) in (int, float) and 0 < value <= MAX_SAFE_INTEGER
            and (type(value) is int or (math.isfinite(value) and value.is_integer())))


def _fields(value, required, optional=frozenset(), *, label):
    if type(value) is not dict:
        raise ValueError(f'{label} must be an object')
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f'{label} has missing or unknown fields')


def validate_filename(value):
    if (not isinstance(value, str) or not _NAME.fullmatch(value)
            or value.endswith('.') or '..' in value
            or value.split('.')[0].upper() in _DEVICES):
        raise ValueError('Unsafe asset filename or build identifier')


def validate_https_url(value):
    if (not isinstance(value, str) or not value.startswith('https://')
            or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
            or '\\' in value):
        raise ValueError('Asset URL must be an ASCII HTTPS URL without whitespace')
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (parsed.scheme != 'https' or not host or parsed.username is not None
                or parsed.password is not None or port == 0):
            raise ValueError('Invalid HTTPS authority')
        if ':' in host:
            ipaddress.IPv6Address(host)
        elif re.fullmatch(r'[0-9.]+', host):
            ipaddress.IPv4Address(host)
        elif (len(host) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in host.split('.'))):
            raise ValueError('Invalid HTTPS hostname')
        # Empty port text is ambiguous across URL parsers.
        if parsed.netloc.endswith(':'):
            raise ValueError('Empty HTTPS port')
    except ValueError as error:
        raise ValueError('Asset URL requires a valid HTTPS hostname and port') from error


def validate_asset(asset):
    _fields(asset, {'name', 'url', 'sha256', 'bytes'}, {'executable'}, label='Asset')
    validate_filename(asset['name'])
    validate_https_url(asset['url'])
    if not isinstance(asset['sha256'], str) or not _HASH.fullmatch(asset['sha256']):
        raise ValueError('Invalid asset SHA256')
    if not _safe_positive_integer(asset['bytes']):
        raise ValueError('Asset bytes must be a positive safe integer')
    if 'executable' in asset and type(asset['executable']) is not bool:
        raise ValueError('Asset executable must be a boolean')


def validate_model_profile(profile):
    _fields(profile, {'renderer', 'task', 'tokenizerRevision', 'modelRevision'}, label='Model profile')
    if profile['renderer'] != 'zeta2-v1' or profile['task'] != 'r-next-edit':
        raise ValueError('Unsupported model renderer/task')
    for key in ('tokenizerRevision', 'modelRevision'):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f'{key} must be a nonempty revision identity')


def validate_manifest(manifest):
    """Validate the entire object before any output directory/file is created."""
    _fields(manifest, {'schema', 'build', 'model', 'modelProfile', 'bundles'}, label='Release manifest')
    if not _safe_positive_integer(manifest['schema']) or manifest['schema'] != 1:
        raise ValueError('Unsupported manifest schema')
    validate_filename(manifest['build'])
    validate_model_profile(manifest['modelProfile'])
    validate_asset(manifest['model'])
    if type(manifest['bundles']) is not list or not manifest['bundles']:
        raise ValueError('Release manifest requires nonempty bundles')
    targets = set()
    for bundle in manifest['bundles']:
        _fields(bundle, {'platform', 'arch', 'backend', 'server', 'files'}, label='Runtime bundle')
        if (bundle['platform'] not in ('linux', 'darwin', 'win32')
                or bundle['arch'] not in ('x64', 'arm64')
                or bundle['backend'] not in ('cpu', 'vulkan', 'metal')):
            raise ValueError('Invalid runtime target')
        target = (bundle['platform'], bundle['arch'], bundle['backend'])
        if target in targets:
            raise ValueError('Duplicate runtime target')
        targets.add(target)
        validate_filename(bundle['server'])
        if type(bundle['files']) is not list or not bundle['files']:
            raise ValueError('Runtime bundle requires nonempty files')
        names = set()
        server_found = False
        for asset in bundle['files']:
            validate_asset(asset)
            # Portable bundles must also be unambiguous on case-insensitive disks.
            name = asset['name'].lower()
            if name in names:
                raise ValueError('Duplicate runtime filename')
            names.add(name)
            if asset['name'] == bundle['server'] and asset.get('executable') is True:
                server_found = True
        if not server_found:
            raise ValueError('Runtime bundle requires its declared executable server')
