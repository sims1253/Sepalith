"""Filesystem roots the training runtime needs, with the campaign defaults.

``CHECKPOINT_ROOT`` is the directory that trainer, archive and bulk-cache
outputs must live inside (``/mnt/e`` on the PC, env ``SEPALITH_CHECKPOINT_ROOT``).
``CAMPAIGN_ROOT`` is the frozen campaign snapshot that packet templates and
receipts resolve against; it defaults to ``docs/campaign`` in this repository.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT_ROOT = "/mnt/e"


def checkpoint_root(value: str) -> Path:
    """Normalize a configured checkpoint root; reject relative or filesystem roots."""
    root = Path(os.path.normpath(value)) if value else Path()
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError(f"checkpoint root must be an absolute directory below the filesystem root: {value!r}")
    return root


CHECKPOINT_ROOT = checkpoint_root(os.environ.get("SEPALITH_CHECKPOINT_ROOT", DEFAULT_CHECKPOINT_ROOT))
CAMPAIGN_ROOT = Path(os.environ.get("SEPALITH_CAMPAIGN_ROOT", REPO_ROOT / "docs" / "campaign"))


def under_checkpoint_root(path: str | os.PathLike[str], root: Path | None = None) -> bool:
    """Return whether ``path`` lies strictly inside the checkpoint root.

    The comparison is by path components after lexical normalization, so a
    sibling such as ``/mnt/e-other`` or ``/mnt/e/../x`` is not inside ``/mnt/e``.
    """
    root = CHECKPOINT_ROOT if root is None else root
    candidate = Path(os.path.normpath(os.fspath(path)))
    return candidate.is_absolute() and candidate != root and candidate.is_relative_to(root)


def campaign_path(relative: str) -> Path:
    """Resolve a path relative to ``docs/campaign`` of the frozen snapshot."""
    return CAMPAIGN_ROOT / relative
