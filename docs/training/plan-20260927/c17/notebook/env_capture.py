"""Capture notebook environment for C17 latency receipts (stdout JSON)."""
import hashlib
import json
import pathlib
import platform
import subprocess

W = pathlib.Path.home() / 'sepalith-c17-latency'


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'error: {e}'


def hashes(build: str) -> dict[str, str]:
    out = {}
    for f in sorted((W / 'llama.cpp' / build / 'bin').glob('*')):
        if f.is_file() and (f.name in {'llama-server'} or f.suffix == '.so' or '.so.' in f.name):
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


env = {
    'hostname': platform.node(),
    'kernel': platform.release(),
    'os': sh('. /etc/os-release; echo "$PRETTY_NAME $BUILD_ID"'),
    'cpu': sh("lscpu | sed -n 's/^Model name: *//p'"),
    'cpus': sh('nproc'),
    'ram': sh("free -b | awk '/Mem:/{print $2}'"),
    'compiler': sh('gcc --version | head -1'),
    'cmake': sh('cmake --version | head -1'),
    'mesa_vulkan_driver': sh("vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion|deviceType'"),
    'mesa_package': sh('pacman -Q vulkan-radeon mesa 2>/dev/null'),
    'power_profile': sh('powerprofilesctl get 2>/dev/null; cat /sys/firmware/acpi/platform_profile 2>/dev/null'),
    'ac_online': sh('cat /sys/class/power_supply/AC*/online 2>/dev/null'),
    'cpu_governor': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
    'llama_cpp_head': sh(f'git -C {W}/llama.cpp/src rev-parse HEAD'),
    'llama_cpp_tag': sh(f'git -C {W}/llama.cpp/src describe --tags'),
    'llama_server_version_cpu': sh(f'{W}/llama.cpp/build-cpu/bin/llama-server --version 2>&1 | tail -2'),
    'llama_server_version_vulkan': sh(f'{W}/llama.cpp/build-vulkan/bin/llama-server --version 2>&1 | tail -4'),
    'cmake_cache_cpu': sh(f"grep -E '^GGML_(NATIVE|VULKAN|CUDA|HIP|BLAS)[:=]' {W}/llama.cpp/build-cpu/CMakeCache.txt"),
    'cmake_cache_vulkan': sh(f"grep -E '^GGML_(NATIVE|VULKAN|CUDA|HIP|BLAS)[:=]' {W}/llama.cpp/build-vulkan/CMakeCache.txt"),
    'binary_sha256_cpu': hashes('build-cpu'),
    'binary_sha256_vulkan': hashes('build-vulkan'),
    'model_sha256_verified_on_notebook': sh(f'cat {W}/models/sha256.txt'),
    'python': platform.python_version(),
    'other_workloads_note': 'Notebook not exclusive: the user\'s own root/docker workloads were left running as instructed; see load_before/load_after.',
}
print(json.dumps(env, indent=2))
