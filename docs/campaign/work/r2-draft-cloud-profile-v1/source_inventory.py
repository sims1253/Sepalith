"""Portable source and dependency inventory for the bounded cloud profile.

The list is deliberately explicit.  It does not walk a checkout and rejects
``.git`` and the local dependency overlay so a cloud payload cannot silently
depend on workstation state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


SOURCE_PATHS = (
    # Runtime and adapter contracts.
    "docs/campaign/work/r2-draft-target-runtime-v2/target_runtime.py",
    "docs/campaign/work/r2-draft-target-runtime-v2/prepare_teacher_cache.py",
    "docs/campaign/work/r2-draft-target-runtime-v2/pretokenized_cache_bridge.py",
    "docs/campaign/work/r2-draft-target-runtime-v2/teacher_rows.py",
    "docs/campaign/work/r2-draft-target-runtime-v2/minicpm5_dspark_config.py",
    "docs/campaign/work/r2-draft-cloud-profile-v1/native_cuda_smoke.py",
    "docs/campaign/work/r2-draft-cloud-profile-v1/profile_orchestrator.py",
    "docs/campaign/work/r2-draft-cloud-profile-v1/source_inventory.py",
    "docs/campaign/work/r2-draft-adapter-hardening-v1/prepare_teacher_cache.py",
    "docs/campaign/work/r2-draft-adapter-hardening-v1/pretokenized_cache_bridge.py",
    "docs/campaign/work/r2-draft-adapter-hardening-v1/teacher_rows.py",
    # Root-owned warm-start entry and portable profile config.
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/warmstart_trainer.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/profile_config.py",
    "docs/campaign/work/r2-draft-upstream-smoke-v1/upstream_dspark_smoke.py",
    # Pinned upstream model, cache, trainer, and utility closure used here.
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/train.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/data/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/data/parser.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/data/target_cache_dataset.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/data/cuda_prefetcher.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/common.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/loss.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/markov_head.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/qwen3/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/qwen3/config.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/gemma4/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/gemma4/config.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/modeling/dspark/gemma4/modeling.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/trainer/base_trainer.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/trainer/dspark_trainer.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/trainer/ckpt_manager.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/config.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/constant/__init__.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/constant/public.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/metrics.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/optim.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/sampling.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/training_logger.py",
    "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/utils/hfai_suspend.py",
)

REQUIREMENT_PATHS = (
    "docs/campaign/work/lead/r2-cloud-control-preparation/payload/requirements.txt",
    "docs/campaign/work/lead/r2-cloud-control-preparation/payload/package-pins.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative(path: str) -> None:
    parts = Path(path).parts
    if Path(path).is_absolute() or ".git" in parts or "dependency-overlay" in parts:
        raise ValueError(f"non-portable inventory path: {path}")


def _rows(root: Path, paths: Iterable[str], kind: str) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        _validate_relative(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"inventory source is missing: {path}")
        rows.append(
            {
                "kind": kind,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def build_inventory(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    return {
        "schema": 1,
        "portable": True,
        "excluded": [".git", "dependency-overlay"],
        "files": _rows(root, SOURCE_PATHS, "source"),
        "requirements": _rows(root, REQUIREMENT_PATHS, "requirements"),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_inventory(args.root), indent=2, sort_keys=True))
