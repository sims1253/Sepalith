"""Load frozen packet templates with their campaign paths rebased.

Packet templates in ``docs/campaign`` record absolute paths into the worktree
they were prepared in. Tests rewrite every ``.../docs/campaign/<rel>`` string to
the first root that holds ``<rel>``: the snapshot (``CAMPAIGN_ROOT``), then the
local overlay for large artifacts the snapshot excludes
(``SEPALITH_CAMPAIGN_OVERLAY``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from sepalith.training.paths import CAMPAIGN_ROOT

OVERLAY = Path(os.environ.get("SEPALITH_CAMPAIGN_OVERLAY", "/mnt/e/sepalith/campaign-20260915/campaign-overlay"))
LEAD = CAMPAIGN_ROOT / "work" / "lead"
_PREFIX = re.compile(r"^/.+?/docs/campaign/(.+)$")


def resolve(relative: str) -> Path:
    for root in (CAMPAIGN_ROOT, OVERLAY):
        if (root / relative).exists():
            return root / relative
    return CAMPAIGN_ROOT / relative


def rebase(value: Any) -> Any:
    if isinstance(value, str):
        match = _PREFIX.match(value)
        return str(resolve(match.group(1))) if match else value
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item) for key, item in value.items()}
    return value


def load(path: Path) -> Any:
    return rebase(json.loads(Path(path).read_text()))


def rebased_copy(path: Path, directory: Path) -> Path:
    """Write a rebased copy of a JSON template and return its path."""
    destination = Path(directory) / Path(path).name
    destination.write_text(json.dumps(load(path), indent=2) + "\n")
    return destination
