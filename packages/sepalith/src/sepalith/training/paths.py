"""Filesystem roots the training runtime needs, with the campaign defaults.

``CHECKPOINT_ROOT`` is the prefix that trainer, archive and bulk-cache outputs
must live under (``/mnt/e/`` on the PC). ``CAMPAIGN_ROOT`` is the frozen
campaign snapshot that packet templates and receipts resolve against; it
defaults to ``docs/campaign`` in this repository.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT_ROOT = os.environ.get("SEPALITH_CHECKPOINT_ROOT", "/mnt/e/")
CAMPAIGN_ROOT = Path(os.environ.get("SEPALITH_CAMPAIGN_ROOT", REPO_ROOT / "docs" / "campaign"))


def campaign_path(relative: str) -> Path:
    """Resolve a path relative to ``docs/campaign`` of the frozen snapshot."""
    return CAMPAIGN_ROOT / relative
