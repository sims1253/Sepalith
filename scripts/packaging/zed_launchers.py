#!/usr/bin/env python3
"""Generate native-shell Zed launchers from a reviewed, hash-pinned release manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

if __package__:
    from .release_manifest import validate_asset, validate_manifest
else:
    from release_manifest import validate_asset, validate_manifest

SH = '''#!/usr/bin/env bash
set -euo pipefail
SEPALITH_ROOT="${SEPALITH_HOME:-$HOME/.local/share/sepalith}"
hash_file() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d ' ' -f 1
  else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}
download() (
  local url="$1" target="$2" digest="$3" bytes="$4"
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]] && [[ "$(wc -c < "$target" | tr -d ' ')" == "$bytes" ]] && [[ "$(hash_file "$target")" == "$digest" ]]; then return; fi
  local temporary
  temporary="$(mktemp "$target.partial.XXXXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  curl --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 1800 --max-redirs 5 --output "$temporary" "$url"
  [[ "$(wc -c < "$temporary" | tr -d ' ')" == "$bytes" ]] && [[ "$(hash_file "$temporary")" == "$digest" ]] || { echo "Download verification failed" >&2; rm -f "$temporary"; return 1; }
  mv -f "$temporary" "$target"
  trap - EXIT
)
'''
PS = '''$ErrorActionPreference = 'Stop'
$SepalithRoot = if ($env:SEPALITH_HOME) { $env:SEPALITH_HOME } else { Join-Path $env:LOCALAPPDATA 'Sepalith' }
function Download-Asset($Url, $Target, $Digest, $Bytes) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
  if ((Test-Path $Target) -and ((Get-Item $Target).Length -eq $Bytes) -and ((Get-FileHash $Target -Algorithm SHA256).Hash.ToLower() -eq $Digest)) { return }
  $Temporary = "$Target.partial.$([Guid]::NewGuid().ToString('N'))"
  try {
    & curl.exe --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 1800 --max-redirs 5 --output $Temporary $Url
    if ($LASTEXITCODE -ne 0) { throw 'Download failed' }
    if (((Get-Item $Temporary).Length -ne $Bytes) -or ((Get-FileHash $Temporary -Algorithm SHA256).Hash.ToLower() -ne $Digest)) { throw 'Download verification failed' }
    Move-Item -Force $Temporary $Target
  } finally { Remove-Item -ErrorAction SilentlyContinue $Temporary }
}
'''


def psquote(s):
    return "'" + s.replace("'", "''") + "'"


def generate(manifest, output):
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    for b in manifest['bundles']:
        key = f"{b['platform']}-{b['arch']}-{b['backend']}"
        identity = hashlib.sha256(json.dumps(b, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()[:16]
        runtime = f"runtimes/{manifest['build']}/{identity}"
        model = manifest['model']
        model_path = f"models/{model['sha256']}/{model['name']}"
        assets = [(a, f"{runtime}/{a['name']}") for a in b['files']] + [(model, model_path)]
        gpu = '0' if b['backend'] == 'cpu' else '99'
        if b['platform'] == 'win32':
            text = PS
            for a, dest in assets:
                text += f"Download-Asset {psquote(a['url'])} (Join-Path $SepalithRoot {psquote(dest)}) {psquote(a['sha256'])} {int(a['bytes'])}\n"
            text += f"& (Join-Path $SepalithRoot {psquote(runtime+'/'+b['server'])}) '-m' (Join-Path $SepalithRoot {psquote(model_path)}) '--alias' 'sepalith' '--host' '127.0.0.1' '--port' '18099' '-c' '8192' '-t' '8' '-ngl' '{gpu}' '--temp' '0'\nexit $LASTEXITCODE\n"
            (output / f'sepalith-{key}.ps1').write_text(text)
        else:
            text = SH
            for a, dest in assets:
                text += f'download {shlex.quote(a["url"])} "$SEPALITH_ROOT/{dest}" {shlex.quote(a["sha256"])} {int(a["bytes"])}\n'
            text += f'chmod +x "$SEPALITH_ROOT/{runtime}/{b["server"]}"\n'
            text += f'exec "$SEPALITH_ROOT/{runtime}/{b["server"]}" -m "$SEPALITH_ROOT/{model_path}" --alias sepalith --host 127.0.0.1 --port 18099 -c 8192 -t 8 -ngl {gpu} --temp 0\n'
            p = output / f'sepalith-{key}.sh'
            p.write_text(text); p.chmod(0o755)


    # The universal launcher embeds each target, so it needs no second script download.
    shell = """#!/usr/bin/env bash
set -euo pipefail
case "$(uname -s)" in Linux) platform=linux;; Darwin) platform=darwin;; *) echo 'Unsupported OS' >&2; exit 1;; esac
case "$(uname -m)" in x86_64|amd64) arch=x64;; aarch64|arm64) arch=arm64;; *) echo 'Unsupported CPU' >&2; exit 1;; esac
backend="${SEPALITH_BACKEND:-auto}"
if [[ "$backend" == auto ]]; then
  backend=cpu
  if [[ "$platform" == darwin ]]; then backend=metal
  elif command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -Eq 'deviceType.*PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then backend=vulkan; fi
fi
case "$platform-$arch-$backend" in
"""
    target_scripts = [output / f"sepalith-{b['platform']}-{b['arch']}-{b['backend']}.sh" for b in manifest['bundles'] if b['platform'] != 'win32']
    supported = '|'.join(p.stem.removeprefix('sepalith-') for p in target_scripts)
    if supported:
        shell = shell.rsplit('case "$platform-$arch-$backend" in\n', 1)[0] + f'case "$platform-$arch-$backend" in {supported}) ;; *) backend=cpu;; esac\ncase "$platform-$arch-$backend" in\n'
    for p in sorted(target_scripts):
        key = p.stem.removeprefix('sepalith-')
        shell += key + ')\n' + p.read_text() + '\n;;\n'
    shell += "*) echo 'No bundled runtime for this target; try SEPALITH_BACKEND=cpu' >&2; exit 1;;\nesac\n"
    universal = output / 'sepalith.sh'
    universal.write_text(shell); universal.chmod(0o755)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    generate(json.loads(args.manifest.read_text()), args.output)
