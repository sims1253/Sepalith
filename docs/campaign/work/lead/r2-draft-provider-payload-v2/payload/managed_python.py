"""Fixed managed Python and candidate dependency checks for the R2 cloud entry.

The pinned Ray image is a Python 3.11 image.  The entry therefore bootstraps
the exact CPython 3.10.19 runtime with the reviewed uv flow during the bounded
setup phase, then verifies the discovered interpreter and development headers
before creating its venv.  This module only plans and verifies those commands;
the entrypoint owns execution and deadlines.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

MANAGED_PYTHON_VERSION = (3, 10, 19)
UV_VERSION = "0.11.23"
UV_SPEC = f"uv=={UV_VERSION}"
PROBE = (
    "import json,sys,sysconfig; "
    "from pathlib import Path; "
    "print(json.dumps({"
    "'version':list(sys.version_info[:3]),"
    "'implementation':sys.implementation.name,"
    "'executable':str(Path(sys.executable).resolve()),"
    "'prefix':str(Path(sys.prefix).resolve()),"
    "'header':str((Path(sysconfig.get_path('include'))/'Python.h').resolve())"
    "}))"
)

# These are the imports used by target-runtime-v3, the pinned DeepSpec source,
# and the profile trainer.  matplotlib is intentionally absent from the
# canonical candidate file and is not part of this runtime closure.
REQUIRED_IMPORTS = (
    "torch",
    "transformers",
    "safetensors",
    "tensorboard",
    "prettytable",
    "deepspec",
)
REQUIRED_REQUIREMENTS = {
    "torch",
    "transformers",
    "numpy",
    "pyyaml",
    "tqdm",
    "triton",
    "typing_extensions",
    "sentencepiece",
    "safetensors",
    "datasets",
    "huggingface_hub",
    "tokenizers",
    "psutil",
    "packaging",
    "tensorboard",
    "prettytable",
}


def bootstrap_paths(run: str | Path) -> dict[str, Path]:
    """Return fresh, run-owned paths used by the uv bootstrap flow."""
    root = Path(run).expanduser().resolve()
    return {
        "tools": root / "bootstrap-tools",
        "python_root": root / "python",
        "venv": root / "venv",
        "uv_cache": root / "uv-cache",
    }


def bootstrap_commands(
    system_python: str | Path,
    run: str | Path,
    *,
    paths: Mapping[str, str | Path] | None = None,
) -> dict[str, list[str]]:
    """Build the pinned recovery commands without probing the host or network.

    ``uv python find`` is deliberately run with ``--resolve-links`` and
    ``--no-python-downloads``.  The preceding ``uv python install`` is the
    only command allowed to obtain CPython, and its destination is run-owned.
    """
    layout = dict(bootstrap_paths(run) if paths is None else {
        key: Path(value).expanduser().resolve() for key, value in paths.items()
    })
    required = {"tools", "python_root", "venv", "uv_cache"}
    require(required <= set(layout), "bootstrap layout is incomplete")
    uv = layout["tools"] / "bin" / "uv"
    return {
        "uv_install": [
            str(Path(system_python).resolve()),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--no-input",
            "install",
            "--target",
            str(layout["tools"]),
            UV_SPEC,
        ],
        "uv_version": [str(uv), "--version"],
        "python_install": [str(uv), "python", "install", "3.10.19"],
        "python_find": [
            str(uv),
            "python",
            "find",
            "--managed-python",
            "--no-python-downloads",
            "--no-project",
            "--resolve-links",
            "3.10.19",
        ],
        "venv": [str(uv), "venv", "--python", "<discovered-python>", str(layout["venv"])],
    }


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def bounded_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve one staged relative path without allowing traversal or links."""
    root_path = Path(root).expanduser().resolve()
    value = Path(relative)
    require(not value.is_absolute() and value.parts not in ((), (".",)), "staged path must be relative")
    require(".." not in value.parts, "staged path traversal is forbidden")
    candidate = root_path.joinpath(*value.parts)
    require(candidate.resolve().is_relative_to(root_path), "staged path escapes root")
    chain = [candidate]
    parent = candidate.parent
    while parent != root_path and root_path in parent.parents:
        chain.append(parent)
        parent = parent.parent
    chain.append(root_path)
    require(not any(item.is_symlink() for item in chain), "staged path symlink is forbidden")
    return candidate


def discover_result(output: str, root: str | Path) -> Path:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    require(len(lines) == 1, "managed Python discovery must return one path")
    path = bounded_path(root, Path(lines[0]).resolve().relative_to(Path(root).resolve())) if Path(lines[0]).is_absolute() else bounded_path(root, lines[0])
    require(path.is_file() and os.access(path, os.X_OK), "managed Python executable is missing")
    return path.resolve()


def validate_runtime_record(record: Mapping[str, Any], interpreter: str | Path, root: str | Path) -> dict[str, Any]:
    expected = list(MANAGED_PYTHON_VERSION)
    require(record.get("version") == expected, "managed Python runtime version differs")
    require(record.get("implementation") == "cpython", "managed Python implementation differs")
    interpreter_path = Path(interpreter).resolve(strict=True)
    executable = Path(str(record.get("executable", ""))).resolve(strict=True)
    require(executable == interpreter_path, "managed Python runtime executable differs")
    root_path = Path(root).resolve(strict=True)
    require(executable.is_relative_to(root_path), "managed Python executable is outside owned root")
    prefix = Path(str(record.get("prefix", ""))).resolve(strict=True)
    require(prefix.is_dir() and prefix.is_relative_to(root_path), "managed Python prefix is outside owned root")
    header = Path(str(record.get("header", ""))).resolve(strict=True)
    require(header.is_file() and header.name == "Python.h" and header.is_relative_to(root_path), "managed Python headers are missing or outside owned root")
    return {
        "version": expected,
        "implementation": "cpython",
        "executable": str(executable),
        "prefix": str(prefix),
        "header": str(header),
    }


def probe_runtime(interpreter: str | Path, root: str | Path, *, runner=subprocess.run) -> dict[str, Any]:
    """Run the fixed interpreter probe; callers may inject a CPU test runner."""
    path = Path(interpreter).resolve(strict=True)
    result = runner(
        [str(path), "-I", "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    require(result.returncode == 0, "managed Python probe failed")
    try:
        record = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise ValueError("managed Python probe returned invalid JSON") from exc
    return validate_runtime_record(record, path, root)


def requirement_names(path: str | Path) -> set[str]:
    names: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower().replace("-", "_")
        if name:
            names.add(name)
    return names


def validate_candidate_requirements(path: str | Path, expected_sha256: str, *, sha256_file) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    digest = sha256_file(path)
    require(digest == expected_sha256, "candidate requirements SHA-256 differs")
    names = requirement_names(path)
    missing = sorted(REQUIRED_REQUIREMENTS - names)
    require(not missing, "candidate requirements omit: " + ", ".join(missing))
    require("matplotlib" not in names, "matplotlib must remain outside the canonical candidate closure")
    return {
        "path": str(path),
        "sha256": digest,
        "required_packages_present": True,
        "matplotlib_present": False,
    }


def check_imports(
    interpreter: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Check the runtime import closure without loading any model or rows."""
    script = (
        "import importlib.util,json; names=" + repr(list(REQUIRED_IMPORTS)) + "; "
        "missing=[name for name in names if importlib.util.find_spec(name) is None]; "
        "print(json.dumps({'missing':missing,'imports':names}))"
    )
    result = runner(
        [str(Path(interpreter).resolve()), "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=None if environment is None else dict(environment),
    )
    require(result.returncode == 0, "managed Python import probe failed")
    record = json.loads(result.stdout)
    require(not record.get("missing"), "runtime imports missing: " + ", ".join(record["missing"]))
    return {"imports": list(REQUIRED_IMPORTS), "missing": []}
