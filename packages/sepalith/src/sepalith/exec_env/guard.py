"""Split and sealed-data guard for the execution environment (TRAIN split only).

Sealed data: every input path goes through ``sepalith.data.synth.build.guard_path``
(refuses the sealed-final prefixes named in CONTEXT.md and any path with a
``final``/``sealed`` component, symlinks resolved).

Split: ``DAT-02-global-split-v2.json`` (sha256 pinned; package-level identity
groups). A CPT document is admitted only if

1. its ``group_id`` is a ``train_group`` in the split, and
2. its package name (normalised as in C10: NFKC, lowercase, ``[0-9a-z._]``)
   occurs in the split's ``pkg:`` identity forms, and only in ``train_group``
   groups (also checked with dots removed).

Anything else (DEV, final-candidate, quarantine, excluded or unknown) is
dropped. Only aggregate counts of dropped packages are ever reported.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from sepalith.data.synth.build import guard_path

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
SPLIT = DATA / "DAT-02-global-split-v2.json"
SPLIT_SHA256 = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
CPT_ROWS = DATA / "CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl"
CPT_ROWS_SHA256 = "7b3edfd052de4be5d1180144f75497aef5e862eeed816b372c5d751a855425bf"
TRAIN = "train_group"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(guard_path(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_package(name: str) -> str:
    return re.sub(r"[^0-9a-z._]", "", unicodedata.normalize("NFKC", name).lower())


class SplitGuard:
    def __init__(self, path: Path = SPLIT, *, verify_sha: bool = True) -> None:
        if verify_sha and sha256_file(path) != SPLIT_SHA256:
            raise SystemExit(f"split file {path} differs from its pinned sha256")
        split = json.loads(Path(guard_path(path)).read_text())
        self.by_group: dict[str, str] = {g["group_id"]: g["split"] for g in split["groups"]}
        self.by_package: dict[str, set[str]] = defaultdict(set)
        for group in split["groups"]:
            for form in group["identity_forms"]:
                if form.startswith("pkg:"):
                    self.by_package[norm_package(form[4:])].add(group["split"])

    def package_splits(self, package: str) -> set[str]:
        name = norm_package(package)
        return self.by_package.get(name, set()) | self.by_package.get(name.replace(".", ""), set())

    def verdict(self, group_id: str, package: str) -> str:
        """``train`` or the reason a document is excluded."""
        if self.by_group.get(group_id) != TRAIN:
            return "group_not_train"
        splits = self.package_splits(package)
        if not splits:
            return "package_not_in_split"
        if splits != {TRAIN}:
            return "package_in_non_train_split"
        return "train"
