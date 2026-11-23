#!/usr/bin/env python3
"""Final panel v1: build DEV-format PRM03 cases from the raw parents of DAT-02 groups.

Design: docs/training/plan-20260927/final-panel-v1.md.

One public entry point, :func:`build_panel`, turns the groups of one DAT-02
split into cases. The families, geometry and filters are those of the C05
DEV250 builder, so the sealed-final v2 launcher scores the cases unchanged.

* ``dev_group`` and ``train_group`` builds are rehearsals. They read only
  those groups' parent sources.
* A ``final_candidate_group`` build reads final content. It runs only with
  an :class:`Authorization` from :func:`authorize`. That needs a user-issued
  receipt naming this file's sha256 and the recipe's sha256, plus the
  receipt's exact sha256.

Errors from the CLI print a fixed message only, never exception text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import stat
import sys
import tarfile
from types import ModuleType
from typing import Any
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYNTH = ROOT / "experiments/synthetic-data"
GUARD_DIR = HERE.parent / "final-admission-guards"
PROTOCOL_DIR = HERE.parent / "final-protocol-v2"
for _path in (ROOT / "packages/sepalith/src", SYNTH, GUARD_DIR, PROTOCOL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from admission_guard import _write_immutable_json, canonical_json, sha256_bytes, sha256_json  # noqa: E402
from sepalith.campaign_protocol import (  # noqa: E402
    build_training_row,
    render_prompt,
    utf16_length,
)
from sepalith.training.rl.campaign_token_audit import length_profiles, selected_context  # noqa: E402

Json = dict[str, Any]
RECIPE_PATH = HERE / "recipe.json"
SCHEMA = "sepalith.final-panel.v1"
FINAL_SPLIT = "final_candidate_group"
REHEARSAL_SPLITS = ("dev_group", "train_group")
FAMILIES = ("finish_block", "no_op", "pipe_rewrite", "format_propagation",
            "rename_propagation", "na_rm_propagation")
SCENARIO_FAMILIES = ("rename_propagation", "pipe_rewrite", "na_rm_propagation")
URI_TAG = {"dev_group": "dev", "train_group": "train", FINAL_SPLIT: "final"}
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
QUOTED = re.compile(r'''"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*\'''')
NA_RM_TRUE = re.compile(r"(?<![\w.])na\.rm\s*=\s*TRUE(?![\w.])")
NATIVE_PIPE_END = re.compile(r"\|>\s*$")


class PanelError(ValueError):
    """A build invariant failed. The code names the failure; it never holds source text."""


# --------------------------------------------------------------------- pins

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(*parts: object) -> int:
    return int(sha256_json([3407, *parts])[:16], 16)


def load_recipe(path: Path = RECIPE_PATH) -> Json:
    recipe = json.loads(Path(path).read_bytes())
    if recipe.get("schema") != SCHEMA + ".recipe":
        raise PanelError("recipe_schema")
    if set(recipe["quotas"]) != set(FAMILIES) or sum(recipe["quotas"].values()) != recipe["case_count_target"]:
        raise PanelError("recipe_quotas")
    return recipe


def verify_modules(recipe: Json) -> dict[str, str]:
    """Every reused module must match the recipe's pins byte for byte."""
    observed = {name: sha256_file(ROOT / name) for name in recipe["modules"]}
    if observed != recipe["modules"]:
        raise PanelError("module_pin_mismatch")
    return observed


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PanelError("module_load_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = ("docs/campaign/work/lead/r2-task-materialization-a/reviewed-source/home/m0hawk/.t3/"
           "worktrees/Sepalith/tuesday-execution-20260912/experiments/training/campaign_admission_completion.py")
REPAIR = "docs/campaign/work/finish-boundary-repair-v2/finish_boundary_repair_v2.py"
CUT = "docs/training/plan-20260927/c05/cut_dev_finish.py"


@dataclass(frozen=True)
class Modules:
    scenarios: ModuleType
    astfim: ModuleType
    cut: ModuleType
    completion: ModuleType
    repair: ModuleType

    @classmethod
    def load(cls) -> Modules:
        import build_astfim
        import scenarios

        cut = _load("cut_dev_finish", ROOT / CUT)
        cut.load_rules()
        return cls(scenarios, build_astfim, cut, _load("campaign_admission_completion", ROOT / ADAPTER),
                   _load("finish_boundary_repair_v2", ROOT / REPAIR))


def loaded_repo_modules() -> set[str]:
    """Repository files imported into this process outside the package tree and this constructor's own dirs."""
    own = (ROOT / "packages/sepalith/src", HERE, GUARD_DIR, PROTOCOL_DIR)
    found = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not isinstance(name, str) or not os.path.isabs(name):
            continue  # e.g. torch's virtual '_classes.py' / '_ops.py' modules
        path = Path(name).resolve()
        if path.is_relative_to(ROOT) and not any(path.is_relative_to(d) for d in own):
            found.add(str(path.relative_to(ROOT)))
    return found


def verify_closure(recipe: Json) -> None:
    """Every imported repository module must be one the recipe pins."""
    if not loaded_repo_modules() <= set(recipe["modules"]):
        raise PanelError("unpinned_module_loaded")


# ------------------------------------------------------------- authorization

@dataclass(frozen=True)
class Authorization:
    """Proof that the user approved one final build of this exact constructor and recipe."""

    receipt_sha256: str
    receipt: Json
    path: Path


def authorize(path: Path, expected_sha256: str, recipe_path: Path = RECIPE_PATH) -> Authorization:
    data = Path(path).read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise PanelError("authorization_hash_mismatch")
    receipt = json.loads(data)
    if (receipt.get("schema") != SCHEMA + ".build-authorization" or receipt.get("status") != "user-approved"
            or receipt.get("split") != FINAL_SPLIT
            or receipt.get("constructor_sha256") != sha256_file(Path(__file__))
            or receipt.get("recipe_sha256") != sha256_file(recipe_path)
            or receipt.get("weights_and_harness_frozen") is not True):
        raise PanelError("authorization_not_for_this_build")
    for name in ("freeze_statement_sha256", "memberships_sha256", "weights_sha256", "harness_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(name))):
            raise PanelError("authorization_missing_" + name)
    for name in ("out_dir", "attempt_marker"):
        if not isinstance(receipt.get(name), str) or not Path(receipt[name]).is_absolute():
            raise PanelError("authorization_missing_" + name)
    return Authorization(expected_sha256, receipt, Path(path))


def claim_attempt(authorization: Authorization, out: Path) -> None:
    """One final build per authorization: create the named marker exclusively before any final read."""
    if out != Path(authorization.receipt["out_dir"]):
        raise PanelError("out_dir_differs_from_authorization")
    marker = Path(authorization.receipt["attempt_marker"])
    import datetime

    payload = canonical_json({"schema": SCHEMA + ".attempt", "authorization_sha256": authorization.receipt_sha256,
                              "out_dir": str(out),
                              "claimed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}).encode() + b"\n"
    _write_immutable_json(marker, payload)  # raises if the marker exists: no second attempt


# ------------------------------------------------------------------ parents

@dataclass(frozen=True)
class Parent:
    group_id: str
    identity: str      # pkg:<name>, the DAT-02 identity form
    package: str       # corpus directory name, case preserved
    version_dir: Path  # <root>/<Package>/<version>

    @property
    def r_dir(self) -> Path:
        return self.version_dir / self.package / "R"


def normalize_package(name: str) -> str:
    return unicodedata.normalize("NFKC", name).lower()


def groups_of(registry: Json, split: str) -> list[Json]:
    """Only the requested split's entries leave this function."""
    chosen = [group for group in registry["groups"] if group["split"] == split and not group["flags"]]
    return sorted(chosen, key=lambda group: group["group_id"])


def parents_for(groups: Iterable[Json], corpus_root: Path, astfim: ModuleType) -> tuple[list[Parent], Counter[str]]:
    """Map each `pkg:` identity form to exactly one corpus directory (NFKC-lowercase equality)."""
    index: dict[str, list[str]] = defaultdict(list)
    for name in sorted(os.listdir(corpus_root)):
        index[normalize_package(name)].append(name)
    parents, reasons = [], Counter()
    for group in groups:
        found = 0
        for form in sorted(f for f in group["identity_forms"] if f.startswith("pkg:")):
            names = index.get(normalize_package(form[4:]), [])
            if len(names) != 1:
                reasons["identity_not_unique_in_corpus" if names else "identity_not_in_corpus"] += 1
                continue
            version = pick_version(corpus_root / names[0], astfim)
            if version is None or not (version / names[0] / "R").is_dir():
                reasons["no_version_or_r_dir"] += 1
                continue
            try:
                checked_tree(version, corpus_root)
            except (PanelError, OSError):
                reasons["parent_tree_refused"] += 1
                continue
            if not _has_license(version / names[0] / "DESCRIPTION"):
                reasons["license_missing"] += 1
                continue
            parents.append(Parent(group["group_id"], form, names[0], version))
            found += 1
        reasons["groups_with_parent" if found else "groups_without_parent"] += 1
    return parents, reasons


def pick_version(package_dir: Path, astfim: ModuleType) -> Path | None:
    """C05's highest-version rule (build_astfim.version_key) with a name tie-break, so readdir order never matters."""
    versions = [v for v in sorted(package_dir.iterdir()) if v.is_dir() and astfim.version_key(v.name) != (-1,)]
    return max(versions, key=lambda v: (astfim.version_key(v.name), v.name)) if versions else None


def _has_license(description: Path) -> bool:
    try:
        text = description.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(line.startswith("License:") and line[8:].strip() for line in text.splitlines())


def checked_tree(root: Path, allowed: Path) -> None:
    """Refuse a package tree that leaves the allowed root or holds any symlink."""
    resolved = root.resolve(strict=True)
    if resolved != root or not resolved.is_relative_to(allowed):
        raise PanelError("source_tree_outside_allowed_root")
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            if stat.S_ISLNK(os.lstat(Path(directory) / name).st_mode):
                raise PanelError("symlink_in_source_tree")


# --------------------------------------------------------------- candidates

@dataclass
class Candidate:
    family: str
    parent: Parent
    rel: str                       # R/<file>
    site: list[object]             # row-ID material; never source text
    source: Path                   # raw input file, or tarball for format
    member: str | None             # tarball member for format
    source_bytes: bytes
    result: Json                   # adapter-shaped: context, selection_source, target_body, operation
    extras: dict[str, tuple[Path, bytes]] = field(default_factory=dict)   # name -> (origin, exact bytes used)
    row_id: str = ""
    function_name: str | None = None   # finish: the harvested function, looked up in the full source


def mixed_or_lone_cr(text: str) -> bool:
    without_crlf = text.replace("\r\n", "")
    return "\r" in without_crlf or ("\r\n" in text and "\n" in without_crlf)


def parse_gate(parser: Any, result: Json) -> str | None:  # noqa: ANN401
    """C05's two R parse checks: the post-event snapshot and the post-edit document must parse."""
    selection = result["selection_source"]
    text = selection["document_text"]
    if parser.parse(text.encode()).root_node.has_error:
        return "history_after_snapshot_R_parse_error"
    lines, eol = _lines(text)
    start, end = selection["region_start_line"], selection["region_end_line"]
    edited = lines[:start] + list(result["target_body"]) + lines[end + 1:]
    if parser.parse(("\r\n" if eol == "crlf" else "\n").join(edited).encode()).root_node.has_error:
        return "full_post_edit_R_parse_error"
    return None


def _lines(text: str) -> tuple[list[str], str]:
    eol = "crlf" if "\r\n" in text else "lf"
    return text.replace("\r\n", "\n").split("\n"), eol


def _context(path: str, uri: str, lines: Sequence[str], eol: str, start: int, region_old: Sequence[str],
             cursor: int, history: list[Json]) -> Json:
    end = start + len(region_old) - 1
    text = ("\r\n" if eol == "crlf" else "\n").join(lines)
    digest = hashlib.sha256(text.encode()).hexdigest()
    context = {
        "schema_version": "sepalith.prompt.prm03.v1", "path": path, "prefix": [], "suffix_lines": [],
        "region_old": list(region_old),
        "cursor": {"region_line_index": 0, "code_point_column": cursor,
                   "utf16_column": utf16_length(region_old[0][:cursor])},
        "replacement_range": {"uri": uri, "document_version": 0, "content_sha256": digest,
                              "start": {"line": start, "character": 0},
                              "end": {"line": end, "character": utf16_length(lines[end])}},
        "document_eol": eol, "scope_mode": "off", "scope_lines": [], "selected_references": [],
        "diagnostics": [], "retrieval": [], "history": history,
    }
    return {"context": context, "selection_source": {
        "availability": "full_snapshot", "document_text": text, "content_sha256": digest,
        "region_start_line": start, "region_end_line": end}}


def convert_example(example: Json, before: str, uri: str, event_id: str) -> Json:
    """Rebuild the lost DAT-07 converter: apply the example's event, locate its region, build the result."""
    lines, eol = _lines(before)
    match = HUNK.search(example["event_diff"])
    if match is None:
        raise PanelError("event_hunk_missing")
    old_start, old_count = int(match[1]) - 1, int(match[2] or 1)
    new_count = int(match[4] or 1)
    body = example["event_diff"].split("@@\n", 1)[1].rsplit("\n```", 1)[0].split("\n")
    old = [line[1:] for line in body if line.startswith("-")]
    new = [line[1:] for line in body if line.startswith("+")]
    if lines[old_start:old_start + old_count] != old or len(old) != old_count or len(new) != new_count:
        raise PanelError("event_does_not_apply")
    prefix, region_old = example["prefix"], example["region_old"]
    hits = [i for i in range(len(lines)) if lines[i:i + len(region_old)] == region_old
            and lines[max(0, i - len(prefix)):i] == prefix]
    if len(hits) != 1:
        raise PanelError("region_not_unique")
    start = hits[0] + (new_count - old_count if hits[0] > old_start else 0)
    after = lines[:old_start] + new + lines[old_start + old_count:]
    if after[start:start + len(region_old)] != region_old:
        raise PanelError("region_moved_by_event")
    sep = "\r\n" if eol == "crlf" else "\n"
    before_sha = hashlib.sha256(sep.join(lines).encode()).hexdigest()
    last = old_start + old_count - 1
    history = [{
        "event_id": event_id, "kind": "observed_edit_diff", "path": example["path"],
        "workspace_revision_before": before_sha, "old_text": "\n".join(old), "new_text": "\n".join(new),
        "event_diff": example["event_diff"],
        "range_utf16": {"uri": uri, "document_version": 0, "content_sha256": before_sha,
                        "start": {"line": old_start, "character": 0},
                        "end": {"line": last, "character": utf16_length(lines[last])}},
    }]
    result = _context(example["path"], uri, after, eol, start, region_old, example["cursor_idx"], history)
    result.update(operation="replace", target_body=list(example["region_new"]))
    return result


def noop_sites(lines: Sequence[str]) -> list[tuple[int, str]]:
    """DEV250's no-op rule: an already-applied `na.rm = TRUE` or a trailing native pipe, outside comments and strings."""
    sites = []
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        code = QUOTED.sub('""', line).split("#", 1)[0]
        if NA_RM_TRUE.search(code):
            sites.append((index, "already_applied_na_rm_true"))
        elif NATIVE_PIPE_END.search(code):
            sites.append((index, "already_applied_native_pipe_operator"))
    return sites


def noop_result(rel: str, uri: str, lines: Sequence[str], eol: str, index: int) -> Json:
    """One no-op case: the site line unchanged, cursor at its end, no history."""
    result = _context(rel, uri, lines, eol, index, [lines[index]], len(lines[index]), [])
    result.update(operation="no_op", target_body=[lines[index]])
    return result


class Builder:
    """Generate every candidate of one group, family by family, with C05's per-(family, group) cap."""

    def __init__(self, modules: Modules, recipe: Json, split: str, cpt: frozenset[str], workdir: Path) -> None:
        self.m, self.recipe, self.split, self.cpt = modules, recipe, split, cpt
        self.cap = recipe["per_family_group_cap"]
        self.tag = URI_TAG[split]
        self.tarballs = Path(recipe["inputs"]["tarball_root"])
        self.reasons: Counter[str] = Counter()
        # C05's harvest re-picks a version from the package directory; pin it to the checked one.
        modules.cut.version_dir = lambda parent: Path(parent["normalized_versions"][0])
        self.finish_rows = workdir / "finish-rows.jsonl"
        self.finish_rows.touch(exist_ok=False)
        self._finish_line = 0

    def _files(self, parent: Parent) -> Iterator[tuple[Path, bytes, str]]:
        for path in sorted(p for p in parent.r_dir.iterdir() if p.is_file() and p.suffix.lower() == ".r"):
            raw = path.read_bytes()
            if not raw or len(raw) > self.m.scenarios.MAX_FILE_BYTES:
                self.reasons["file_empty_or_large"] += 1
                continue
            if sha256_bytes(raw) in self.cpt:
                self.reasons["file_in_cpt"] += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeError:
                self.reasons["file_not_utf8"] += 1
                continue
            if mixed_or_lone_cr(text):
                self.reasons["file_mixed_eol_or_lone_cr"] += 1
                continue
            yield path, raw, text

    def _uri(self, row_id: str, rel: str) -> str:
        return f"file:///sepalith/{self.tag}/{row_id}/{rel}"

    def _identify(self, cand: Candidate) -> Candidate:
        cand.row_id = "final-v1-" + sha256_json(
            [3407, cand.parent.group_id, cand.parent.package, cand.rel, cand.family, cand.site])[:24]
        return cand

    def group(self, parents: Sequence[Parent]) -> list[Candidate]:
        out: list[Candidate] = []
        for family in FAMILIES:
            kept: list[Candidate] = []
            for parent in parents:
                if len(kept) >= self.cap:
                    break
                make = {"finish_block": self.finish, "no_op": self.noop,
                        "format_propagation": self.format}.get(family, self.scenario)
                for cand in make(parent, family, self.cap - len(kept)):
                    kept.append(cand)
            out.extend(kept)
        return out

    def scenario(self, parent: Parent, family: str, room: int) -> Iterator[Candidate]:
        extract = {"rename_propagation": self.m.scenarios.extract_rename,
                   "pipe_rewrite": self.m.scenarios.extract_pipe,
                   "na_rm_propagation": self.m.scenarios.extract_na_rm}[family]
        seen = set()
        for path, raw, text in self._files(parent):
            rel = f"R/{path.name}"
            try:
                bundle = self.m.scenarios.Bundle(parent.package, rel, raw)
                examples = extract(bundle, random.Random(seed_for(parent.group_id, rel, family)), cap=4)
            except Exception:  # noqa: BLE001 - an extractor crash on one file skips that file, as in C05
                self.reasons[f"{family}:extractor_error"] += 1
                continue
            for example in examples:
                if example.get("family") != family:
                    continue
                key = (rel, tuple(example["region_new"]))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    self.m.scenarios.validate_example(example)
                except Exception:  # noqa: BLE001 - invalid examples are dropped, as in C05
                    self.reasons[f"{family}:invalid_example"] += 1
                    continue
                cand = yield from self._convert(parent, family, rel, path, None, raw, text, example)
                if cand:
                    room -= 1
                    if room == 0:
                        return

    def format(self, parent: Parent, family: str, room: int) -> Iterator[Candidate]:
        tar_path = self.tarballs / f"{parent.package}_{parent.version_dir.name}.tar.gz"
        if (not tar_path.is_file() or tar_path.is_symlink()
                or tar_path.stat().st_size > self.m.scenarios.TAR_MAX_BYTES):
            self.reasons["format:no_tarball_or_too_large"] += 1
            return
        normalized = {p.name: p for p in parent.r_dir.iterdir() if p.is_file() and p.suffix.lower() == ".r"}
        members: dict[str, bytes] = {}
        try:
            with tarfile.open(tar_path, "r:gz") as archive:
                for info in archive.getmembers():
                    parts = info.name.split("/")
                    if (len(parts) == 3 and parts[0] == parent.package and parts[1] == "R" and info.isfile()
                            and parts[2] in normalized
                            and info.size <= self.m.scenarios.MAX_FILE_BYTES and parts[2] not in members):
                        handle = archive.extractfile(info)
                        if handle is not None:
                            members[parts[2]] = handle.read()
        except (tarfile.TarError, OSError, EOFError):
            self.reasons["format:tarball_unreadable"] += 1
            return
        for name in sorted(members):
            raw, norm_path = members[name], normalized[name]
            norm = norm_path.read_bytes()
            if sha256_bytes(raw) in self.cpt or sha256_bytes(norm) in self.cpt:
                self.reasons["format:file_in_cpt"] += 1
                continue
            try:
                if len(norm) > self.m.scenarios.MAX_FILE_BYTES or mixed_or_lone_cr(norm.decode("utf-8")):
                    raise UnicodeError
            except UnicodeError:
                self.reasons["format:normalized_not_clean_utf8"] += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeError:
                self.reasons["format:not_utf8"] += 1
                continue
            if mixed_or_lone_cr(text):
                self.reasons["format:mixed_eol_or_lone_cr"] += 1
                continue
            rel = f"R/{name}"
            examples = self.m.scenarios.format_pairs_from_lines(
                parent.package, rel, raw.decode("utf-8", "replace").splitlines(),
                norm.decode("utf-8", "replace").splitlines())
            for example in examples:
                cand = yield from self._convert(parent, family, rel, tar_path, f"{parent.package}/R/{name}",
                                                raw, text, example, extras={"normalized_after": (norm_path, norm)})
                if cand:
                    room -= 1
                    if room == 0:
                        return

    def _convert(self, parent: Parent, family: str, rel: str, source: Path, member: str | None, raw: bytes,
                 text: str, example: Json, extras: dict[str, tuple[Path, bytes]] | None = None,
                 ) -> Iterator[Candidate]:
        match = HUNK.search(example["event_diff"])
        site = [example.get("note", ""), int(match[1]) if match else -1, example["cursor_idx"],
                len(example["prefix"]), sha256_json(example["prefix"])]
        cand = self._identify(Candidate(family, parent, rel, site, source, member, raw, {}, extras or {}))
        try:
            cand.result = convert_example(example, text, self._uri(cand.row_id, rel),
                                          sha256_bytes(f"{cand.row_id}:event".encode())[:32])
        except PanelError as error:
            self.reasons[f"{family}:{error}"] += 1
            return False
        failed = parse_gate(self.m.scenarios.parser, cand.result)
        if failed:
            self.reasons[f"{family}:{failed}"] += 1
            return False
        yield cand
        return True

    def noop(self, parent: Parent, family: str, room: int) -> Iterator[Candidate]:
        per_file = []
        for path, raw, text in self._files(parent):
            lines, eol = _lines(text)
            sites = noop_sites(lines)
            if sites:
                per_file.append((path, raw, lines, eol, sites))
        found = []
        for round_ in range(room):
            for path, raw, lines, eol, sites in per_file:
                if len(found) < room and round_ < len(sites):
                    found.append((path, raw, lines, eol, *sites[round_]))
        for path, raw, lines, eol, index, basis in found:
            rel = f"R/{path.name}"
            cand = self._identify(Candidate(family, parent, rel, [index, basis], path, None, raw, {}))
            cand.result = noop_result(rel, self._uri(cand.row_id, rel), lines, eol, index)
            yield cand

    def finish(self, parent: Parent, family: str, room: int) -> Iterator[Candidate]:
        cut = self.m.cut
        stats: Counter[str] = Counter()
        rows, bases = cut.harvest({"package": parent.package, "group_id": parent.group_id,
                                   "normalized_versions": [str(parent.version_dir)]}, set(self.cpt), stats)
        self.reasons.update({f"finish:{k}": v for k, v in stats.items()})
        by_base = defaultdict(list)
        for row in rows:
            by_base[row["base_id"]].append(row)
        for base in sorted(bases, key=lambda b: b["source_file"])[:room]:
            row = random.Random(seed_for(parent.group_id, base["base_id"])).choice(by_base[base["base_id"]])
            line = (json.dumps(row) + "\n").encode()
            with self.finish_rows.open("ab") as stream:
                stream.write(line)
            self._finish_line += 1
            source = Path(base["source_file"])
            if not source.resolve().is_relative_to(parent.r_dir.resolve()):
                raise PanelError("finish_source_outside_checked_parent")
            raw = source.read_bytes()
            if sha256_bytes(raw) != base["source_sha256"]:
                raise PanelError("finish_source_changed_after_harvest")
            cand = self._identify(Candidate(family, parent, base["path"],
                                            [base["base_id"], row["transform"], row["cut"]],
                                            source, None, raw, {}, function_name=base["fn"]))
            result, why = self._finish_result(row, line, cand)
            if result is None:
                self.reasons[f"finish:{why}"] += 1
                continue
            cand.result = result
            yield cand

    def _finish_result(self, raw: Json, line: bytes, cand: Candidate) -> tuple[Json | None, str]:
        """C05 finish_packet: the DAT-04B adapter, then DEV75's outer-brace correction."""
        ref = {"file": str(self.finish_rows), "line": self._finish_line, "raw_line_sha256": sha256_bytes(line),
               "source_sha256": sha256_file(self.finish_rows), "split": self.split, "row_id": cand.row_id,
               "group_id": cand.parent.group_id, "document_role": "source_derived_simulated_pre_edit",
               "source_constructor": "finish_block_v5_prefix", "target_convention": "suffix",
               "uri": self._uri(cand.row_id, raw["path"]), "document_version": 0}
        return convert_finish(self.m, raw, ref)


def convert_finish(m: Modules, raw: Json, ref: Json) -> tuple[Json | None, str]:
    """C05 finish_packet: the DAT-04B adapter, then DEV75's outer-brace correction."""
    from sepalith.campaign_protocol import utf16_to_codepoint_column

    result = m.completion.convert_completion(raw, ref)
    if result["status"] != "converted":
        return None, str(result.get("reason"))
    target = "\n".join(result["target_body"])
    if "\n" not in target or target.rsplit("\n", 1)[-1].strip(" \t") != "":
        return None, "target_not_brace_correctable"
    before, ctx = result["selection_source"]["document_text"], result["context"]

    def splice(text: str) -> str:
        return m.repair.apply_document_replacement(
            before, ctx["replacement_range"], text, region_old=ctx["region_old"],
            utf16_to_codepoint_column=utf16_to_codepoint_column)

    parse = m.scenarios.parser.parse
    if not parse(splice(target).encode()).root_node.has_error:
        return None, "uncorrected_document_already_parses"
    if parse(splice(target + "}").encode()).root_node.has_error:
        return None, "corrected_document_does_not_parse"
    result["target_body"] = (target + "}").split("\n")
    return result, ""


# ----------------------------------------------------------------- function ids

def function_ids(package: str, rel: str, text: str, spans: Iterable[tuple[int, int]], parser: Any) -> list[str]:  # noqa: ANN401
    """Exec-pool IDs (sha256(package NUL path NUL name)[:16]) of top-level functions meeting any span."""
    tree = parser.parse(text.encode())
    ids = set()
    for node in tree.root_node.children:
        if node.type != "binary_operator" or node.child_count < 3:
            continue
        lhs, op, rhs = node.children[0], node.children[1], node.children[-1]
        if lhs.type != "identifier" or op.type not in ("<-", "=") or rhs.type != "function_definition":
            continue
        first, last = node.start_point[0], node.end_point[0]
        if any(first <= end and start <= last for start, end in spans):
            ids.add(function_id(package, rel, text.encode()[lhs.start_byte:lhs.end_byte].decode()))
    return sorted(ids)


def function_id(package: str, rel: str, name: str) -> str:
    """exec_env/pool.py's function_id: sha256(package NUL R/path NUL name)[:16]."""
    return hashlib.sha256(f"{package}\0{rel}\0{name}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------ selection

class FlowNetwork:
    """C05 build_dev250.max_flow_allocation (Edmonds-Karp, sorted adjacency) as a reusable residual network.

    Pass 1 is C05's allocation unchanged. Pass 2 only raises the source edge
    of finish_block and keeps augmenting the same flow. An augmenting path
    leaves the source on an edge with residual capacity and never returns to
    it, so no other slot's total can fall.
    """

    def __init__(self, supply: Counter[tuple[str, str]], quotas: Mapping[str, int], group_cap: int) -> None:
        self.cap: dict[tuple[object, object], int] = defaultdict(int)
        self.adj: dict[object, set[object]] = defaultdict(set)
        self.flow: dict[tuple[object, object], int] = defaultdict(int)
        for s, q in quotas.items():
            self._edge("S", ("slot", s), q)
        for (s, g), n in sorted(supply.items()):
            self._edge(("slot", s), ("group", g), n)
        for g in sorted({g for _, g in supply}):
            self._edge(("group", g), "T", group_cap)
        self.order = {n: i for i, n in enumerate(sorted(self.adj, key=repr))}

    def _edge(self, u: object, v: object, c: int) -> None:
        self.cap[(u, v)] += c
        self.adj[u].add(v)
        self.adj[v].add(u)

    def raise_slot(self, slot: str, extra: int) -> None:
        self.cap[("S", ("slot", slot))] += extra

    def augment(self) -> None:
        while True:
            parent: dict[object, object] = {"S": None}
            queue: deque[object] = deque(["S"])
            while queue and "T" not in parent:
                u = queue.popleft()
                for v in sorted(self.adj[u], key=self.order.__getitem__):
                    if v not in parent and self.cap[(u, v)] - self.flow[(u, v)] > 0:
                        parent[v] = u
                        queue.append(v)
            if "T" not in parent:
                return
            path, v = [], "T"
            while parent[v] is not None:
                path.append((parent[v], v))
                v = parent[v]
            push = min(self.cap[e] - self.flow[e] for e in path)
            for u, v in path:
                self.flow[(u, v)] += push
                self.flow[(v, u)] -= push

    def allocation(self) -> dict[tuple[str, str], int]:
        return {(s[1], g[1]): n for (s, g), n in list(self.flow.items())
                if isinstance(s, tuple) and s[0] == "slot" and isinstance(g, tuple) and g[0] == "group" and n > 0}


def allocate(candidates: list[Json], quotas: Mapping[str, int], group_cap: int) -> tuple[list[Json], Json]:
    """Max-flow under the group cap; unfilled edit slots move to finish_block once; seeded candidate order.

    A no_op or finish_block shortfall is not moved: it shrinks the panel.
    """
    ordered = sorted(candidates, key=lambda c: sha256_json([3407, c["row_id"]]))
    supply = Counter((c["family"], c["group_id"]) for c in ordered)
    network = FlowNetwork(supply, quotas, group_cap)
    network.augment()
    first = Counter()
    for (slot, _), n in network.allocation().items():
        first[slot] += n
    moved = sum(quotas[f] - first[f] for f in quotas if f not in ("finish_block", "no_op"))
    if moved:
        network.raise_slot("finish_block", moved)
        network.augment()
    alloc = network.allocation()
    take, chosen = Counter(), []
    for c in ordered:
        key = (c["family"], c["group_id"])
        if take[key] < alloc.get(key, 0):
            take[key] += 1
            chosen.append(c)
    by_family = Counter(c["family"] for c in chosen)
    targets = {f: (first[f] if f not in ("finish_block", "no_op") else quotas[f]) for f in quotas}
    targets["finish_block"] += moved
    return chosen, {
        "quotas": dict(quotas), "pass1_by_family": {f: first[f] for f in FAMILIES},
        "moved_to_finish_block": moved, "targets_after_move": targets,
        "selected_by_family": {f: by_family[f] for f in FAMILIES},
        "shortfall": {f: targets[f] - by_family[f] for f in FAMILIES if by_family[f] < targets[f]},
    }


# ----------------------------------------------------------------------- build

@dataclass
class Panel:
    split: str
    cases: list[Json]
    payloads: dict[str, Json]
    staged: dict[str, tuple[Path, bytes]]   # row_id -> (relative staged path, bytes)
    extras: dict[str, tuple[Path, bytes]]   # staged secondary inputs (format normalized-after files)
    receipt: Json


def dev_prompt_hashes(recipe: Json) -> frozenset[str]:
    """Prompt hashes of DEV75 and DEV250 (dev rows); a final case must not repeat one."""
    hashes = set()
    for item in recipe["inputs"]["dev_panels"]:
        with Path(item["path"]).open("rb") as stream:
            for line in stream:
                if line.strip():
                    hashes.add(json.loads(line)["prompt_sha256"])
    return frozenset(hashes)


def cpt_hashes(recipe: Json) -> frozenset[str]:
    index = recipe["inputs"]["cpt_index"]
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        return frozenset(h for (h,) in connection.execute("select source_sha256 from documents"))
    finally:
        connection.close()


def overlap_audit(cases: Sequence[Json], memberships: Json) -> Json:
    """The v2 launcher's own audit(), aggregate counts only."""
    import protocol as v2

    specs = [v2.Case(v2.SourceCaseSpec.from_mapping({**c, "source_path": "/staged/" + c["row_id"]}),
                     frozenset(c["function_ids"])) for c in cases]
    overlap, counts = v2.audit(specs, memberships)
    return {**counts, "ceiling_pct": 20, "exceeds_ceiling": len(overlap) * 100 > 20 * counts["groups"]}


def build_panel(*, split: str, registry: Json, recipe: Json, tokenizer: Any, workdir: Path,  # noqa: ANN401
                cpt: frozenset[str], dev_prompts: frozenset[str], authorization: Authorization | None = None,
                group_ids: Sequence[str] | None = None, quotas: Mapping[str, int] | None = None,
                modules: Modules | None = None, memberships: Json | None = None) -> Panel:
    """Build one panel. Rehearsal splits may limit groups and quotas; the final split may not."""
    if split == FINAL_SPLIT:
        if authorization is None:
            raise PanelError("final_build_requires_authorization")
        if group_ids is not None or quotas is not None or not cpt or memberships is None:
            raise PanelError("final_build_uses_recipe_as_is")
        # The one guarded entry point: re-verify the receipt from disk, then claim the one-shot marker
        # before any final metadata or source is read.
        authorization = authorize(authorization.path, authorization.receipt_sha256)
        claim_attempt(authorization, Path(authorization.receipt["out_dir"]))
    elif split not in REHEARSAL_SPLITS:
        raise PanelError("unknown_split")
    modules = modules or Modules.load()
    if split == FINAL_SPLIT:
        modules.completion.ALLOWED_SPLITS = frozenset({*modules.completion.ALLOWED_SPLITS, FINAL_SPLIT})
    quotas = dict(quotas or recipe["quotas"])
    corpus = Path(recipe["inputs"]["corpus_root"])
    groups = groups_of(registry, split)
    if group_ids is not None:
        wanted = set(group_ids)
        groups = [g for g in groups if g["group_id"] in wanted]
    parents, parent_reasons = parents_for(groups, corpus, modules.astfim)
    builder = Builder(modules, recipe, split, cpt, workdir)
    by_group: dict[str, list[Parent]] = defaultdict(list)
    for parent in parents:
        by_group[parent.group_id].append(parent)

    usable, seen_prompts, seen_rows, filtered = [], set(), set(), Counter()
    raw_candidates = Counter()
    for group_id in sorted(by_group):
        for cand in builder.group(by_group[group_id]):
            raw_candidates[cand.family] += 1
            record = _finish_candidate(cand, tokenizer, split, recipe, modules)
            if isinstance(record, str):
                filtered[f"{cand.family}:{record}"] += 1
                continue
            if record["prompt_sha256"] in dev_prompts:
                filtered[f"{cand.family}:same_prompt_as_dev_panel"] += 1
                continue
            if record["prompt_sha256"] in seen_prompts:
                filtered[f"{cand.family}:duplicate_prompt"] += 1
                continue
            if record["row_id"] in seen_rows:
                filtered[f"{cand.family}:duplicate_row_id"] += 1
                continue
            seen_prompts.add(record["prompt_sha256"])
            seen_rows.add(record["row_id"])
            usable.append(record)
    chosen, allocation = allocate(usable, quotas, recipe["group_cap"])
    chosen.sort(key=lambda c: c["row_id"])

    groups_used = Counter(c["group_id"] for c in chosen)
    floor = recipe["floor"]
    selected = allocation["selected_by_family"]
    floor_failures = [] if len(chosen) >= floor["cases"] else ["cases"]
    floor_failures += [family for family, minimum in floor["families"].items() if selected[family] < minimum]
    receipt = {
        "schema": SCHEMA + ".build",
        "split": split,
        "status": "stopped_below_floor" if floor_failures else "built",
        "floor_failures": floor_failures,
        "counts": {
            "groups_in_split": len(groups), "parents": len(parents), "groups_with_parent": len(by_group),
            "parent_mapping": dict(parent_reasons), "candidates_by_family": dict(raw_candidates),
            "usable_candidates": len(usable), "cases": len(chosen), "groups_used": len(groups_used),
            "cases_per_group_hist": dict(Counter(groups_used.values())),
            "by_operation": dict(Counter(c["operation"] for c in chosen)),
        },
        "allocation": allocation,
        "exclusions": dict(sorted((builder.reasons + filtered).items())),
        "authorization_sha256": authorization.receipt_sha256 if authorization else None,
        "cpt_filter": bool(cpt),
    }
    if memberships is not None and chosen:
        receipt["overlap_audit"] = overlap_audit(chosen, memberships)
        if receipt["overlap_audit"]["exceeds_ceiling"] and receipt["status"] == "built":
            # The unchanged launcher would take its seal and then abort; stop here instead.
            receipt["status"] = "stopped_over_overlap_ceiling"
    staged = {c["row_id"]: c.pop("_staged") for c in chosen}
    extras = {k: v for c in chosen for k, v in c.pop("_extras").items()}
    payloads = {c["row_id"]: c.pop("_payload") for c in chosen}
    receipt["_provenance"] = {c["row_id"]: c.pop("_provenance") for c in chosen}
    for c in chosen:
        c.pop("prompt_sha256")
        c.pop("operation")
    return Panel(split, chosen, payloads, staged, extras, receipt)


def _finish_candidate(cand: Candidate, tokenizer: Any, split: str, recipe: Json,  # noqa: ANN401
                      modules: Modules) -> Json | str:
    """Window, validate, length-filter and describe one candidate; return a reason code on rejection."""
    try:
        context, _selection = selected_context(cand.result)
    except (ValueError, KeyError, TypeError):
        return "window_failed"
    operation, region_new = cand.result["operation"], list(cand.result["target_body"])
    try:
        row = build_training_row(context, operation=operation, region_new=region_new, tokenizer=tokenizer,
                                 row_id=cand.row_id, family=cand.family, package_id=cand.parent.identity,
                                 split="final_locked_v1" if split == FINAL_SPLIT else "dev")
    except (ValueError, KeyError, TypeError):
        return "row_invalid"
    lengths = length_profiles(row)
    if lengths["response_with_terminal_eos"] > recipe["filters"]["generation_cap"]:
        return "reference_exceeds_generation_cap"
    if lengths["sequence"] > recipe["filters"]["max_sequence"]:
        return "sequence_over_max"
    context_map = context.to_dict()
    target = {"operation": operation, "region_new": region_new}
    selection = cand.result["selection_source"]
    parser = modules.scenarios.parser
    if cand.function_name is not None:
        # A finish document is a cut prefix; take the ID from the complete original source instead.
        full = cand.source_bytes.decode("utf-8", "replace")
        everything = [(0, full.count("\n") + 1)]
        ids = [i for i in function_ids(cand.parent.package, cand.rel, full, everything, parser)
               if i == function_id(cand.parent.package, cand.rel, cand.function_name)]
    else:
        spans = [(selection["region_start_line"], selection["region_end_line"])]
        ids = function_ids(cand.parent.package, cand.rel, selection["document_text"], spans, parser)
    for event in context_map["history"]:
        r = event["range_utf16"]
        before = cand.source_bytes.decode("utf-8", "replace")
        ids = sorted(set(ids) | set(function_ids(cand.parent.package, cand.rel, before,
                                                  [(r["start"]["line"], r["end"]["line"])],
                                                  modules.scenarios.parser)))
    # Content-addressed staging: no package or file name enters a staged path, and the raw
    # tarball member and the normalized file can never share one.
    source_sha = sha256_bytes(cand.source_bytes)
    staged_rel = Path(source_sha + ".R")
    extras = {}
    provenance = {"package": cand.parent.package, "version": cand.parent.version_dir.name, "rel": cand.rel,
                  "source": str(cand.source) + (f"::{cand.member}" if cand.member else ""),
                  "source_sha256": source_sha}
    for name, (path, data) in cand.extras.items():
        extras[f"{cand.row_id}:{name}"] = (Path(sha256_bytes(data) + ".R"), data)
        provenance[name] = {"source": str(path), "sha256": sha256_bytes(data)}
    return {
        "row_id": cand.row_id,
        "split_identity": f"{cand.parent.identity}::{cand.parent.identity}::{cand.parent.group_id}::{cand.row_id}",
        "package_id": cand.parent.identity,
        "repository_id": cand.parent.identity,
        "group_id": cand.parent.group_id,
        "family": cand.family,
        "source_path": "",  # set at write time, under the staged root
        "source_sha256": source_sha,
        "context_sha256": sha256_json(context_map),
        "target_sha256": sha256_json(target),
        "function_ids": ids,
        "operation": operation,
        "prompt_sha256": hashlib.sha256(render_prompt(context).encode()).hexdigest(),
        "_payload": {"context": context_map, "target": target},
        "_staged": (staged_rel, cand.source_bytes),
        "_extras": extras,
        "_provenance": provenance,
    }


# ----------------------------------------------------------------------- write

def _write_readonly(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def write_panel(panel: Panel, out: Path, *, recipe_sha256: str, inputs: Json) -> Json:
    """Write staged sources, cases, payloads, semantic receipt and receipts into a fresh mode-700 directory.

    `build-receipt.json` holds aggregate counts and file digests only. Per-case provenance (original
    paths and hashes) goes to `provenance.json`, which stays in this private directory.
    """
    if not out.is_absolute() or os.path.lexists(out):
        raise PanelError("output_must_be_fresh_absolute_directory")
    out.mkdir(mode=0o700)
    sources = out / "final-sources"
    sources.mkdir(mode=0o700)
    written: dict[Path, bytes] = {}
    for _key, (relative, data) in [*panel.staged.items(), *panel.extras.items()]:
        if relative in written:
            if written[relative] != data:
                raise PanelError("staged_path_conflict")
            continue
        _write_readonly(sources / relative, data)
        written[relative] = data
    for case in panel.cases:
        case["source_path"] = str(sources / panel.staged[case["row_id"]][0])
    split_label = "final_locked_v1" if panel.split == FINAL_SPLIT else f"rehearsal_{panel.split}"
    cases_bytes = canonical_json({"split": split_label, "cases": panel.cases}).encode() + b"\n"
    _write_immutable_json(out / "cases.json", cases_bytes)
    _write_immutable_json(out / "payloads.json", canonical_json(panel.payloads).encode() + b"\n")
    receipt = dict(panel.receipt)
    _write_immutable_json(out / "provenance.json", canonical_json(receipt.pop("_provenance")).encode() + b"\n")
    cases_sha = sha256_bytes(cases_bytes)
    semantic = {
        "schema": "dat08.semantic-source-case-receipt.v1", "status": "source_cases_verified",
        "split": split_label, "source_lock_sha256": cases_sha, "content_access": "post_freeze_source_bytes",
        "verified_case_ids": [c["row_id"] for c in panel.cases],
        "constructor_sha256": sha256_file(Path(__file__)),
        "case_bindings": {c["row_id"]: {k: c[k] for k in ("split_identity", "source_sha256", "context_sha256",
                                                           "target_sha256")} for c in panel.cases},
        "issuance_mismatches": [],
    }
    _write_immutable_json(out / "semantic-receipt.json", canonical_json(semantic).encode() + b"\n")
    for directory, _dirs, _names in os.walk(sources, topdown=False):
        os.chmod(directory, 0o500)
    receipt.update({
        "constructor_sha256": semantic["constructor_sha256"], "recipe_sha256": recipe_sha256, "inputs": inputs,
        "outputs": {"cases_sha256": cases_sha, "case_count": len(panel.cases),
                    "payloads_sha256": sha256_file(out / "payloads.json"),
                    "semantic_receipt_sha256": sha256_file(out / "semantic-receipt.json"),
                    "provenance_sha256": sha256_file(out / "provenance.json"),
                    "source_root": str(sources)},
    })
    return receipt


def launcher_dry_run(out: Path, tokenizer: Any, *, weights_sha256: str, harness_sha256: str,  # noqa: ANN401
                     rehearsal: bool) -> Json:
    """Run the v2 launcher's own load, semantic-receipt, root, source-read and construct steps on the outputs.

    The real run takes its seal (`started.json`) before these steps, so any defect found later would
    leave a dead run. A rehearsal panel is relabelled in memory only.
    """
    import protocol as v2
    from admission_guard import AdmissionGuard

    document = json.loads((out / "cases.json").read_bytes())
    semantic = json.loads((out / "semantic-receipt.json").read_bytes())
    if rehearsal:
        document["split"] = semantic["split"] = "final_locked_v1"
    cases = v2.load_cases(document, len(document["cases"]))
    stamp = "2026-09-14T10:00:00Z"
    guard = AdmissionGuard(
        allowed_source_roots=[out / "final-sources"], train_identities=(), dev_identities=(),
        freeze_receipt={"status": "frozen", "weights_frozen": True, "harness_frozen": True,
                        "final_access_unlocked": True, "weights_frozen_at": stamp, "harness_frozen_at": stamp,
                        "weights_sha256": weights_sha256, "harness_sha256": harness_sha256},
        semantic_receipt=semantic, expected_weights_sha256=weights_sha256,
        expected_harness_sha256=harness_sha256, expected_source_lock_sha256=sha256_file(out / "cases.json"),
        observed_at=stamp)
    guard._validate_freeze_receipt()
    guard._validate_semantic_receipt([case.spec for case in cases])
    prepared, mismatches = v2.construct(cases, out / "payloads.json", guard, set(), tokenizer)
    if mismatches["cases"] or len(prepared) != len(cases):
        raise PanelError("launcher_dry_run_failed")
    return {"cases_loaded": len(cases), "constructed": len(prepared), "hash_mismatches": mismatches["cases"]}


# ------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=(*REHEARSAL_SPLITS, FINAL_SPLIT))
    parser.add_argument("--out", type=Path, required=True, help="fresh absolute directory (mode 700)")
    parser.add_argument("--tokenizer", type=Path, required=True, help="campaign tokenizer directory")
    parser.add_argument("--memberships", type=Path, help="the eight inventories (required for a final build)")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--models", type=Path, help="final: the plan's models object, checked against the authorization")
    parser.add_argument("--freeze-statement", type=Path, help="final: the user's written freeze statement")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--groups", type=int, help="rehearsal only: first N groups in seeded order")
    parser.add_argument("--scale", type=float, help="rehearsal only: scale the quotas")
    parser.add_argument("--no-cpt-filter", action="store_true",
                        help="rehearsal only: train sources are CPT documents; measure supply without the filter")
    args = parser.parse_args(argv)
    final = args.split == FINAL_SPLIT
    if final and (args.groups or args.scale or args.no_cpt_filter or not args.memberships or not args.models
                  or not args.freeze_statement or not args.authorization or not args.authorization_sha256):
        parser.error("a final build needs --authorization, --authorization-sha256, --memberships, --models and "
                     "--freeze-statement, and takes no rehearsal options")
    try:
        report = _run(args, final)
    except Exception:  # noqa: BLE001 - exception text may hold source text in a final build
        print("Final panel build stopped. No details are printed; inspect the inputs with the custodian.",
              file=sys.stderr)
        return 2
    print(json.dumps({k: report.get(k) for k in ("status", "floor_failures", "counts", "allocation",
                                                  "overlap_audit", "launcher_dry_run", "outputs")},
                     indent=1, sort_keys=True))
    return 0 if report["status"] == "built" else 2


def _run(args: argparse.Namespace, final: bool) -> Json:
    recipe = load_recipe()
    authorization = authorize(args.authorization, args.authorization_sha256) if final else None
    module_pins = verify_modules(recipe)
    memberships = None
    if args.memberships:
        data = args.memberships.read_bytes()
        if authorization is not None and sha256_bytes(data) != authorization.receipt["memberships_sha256"]:
            raise PanelError("memberships_differ_from_authorization")
        memberships = json.loads(data)
    if authorization is not None:
        import protocol as v2

        if v2.harness_hash() != authorization.receipt["harness_sha256"]:
            raise PanelError("harness_differs_from_authorization")
        if args.out.resolve() != Path(authorization.receipt["out_dir"]):
            raise PanelError("out_dir_differs_from_authorization")
        # The weights and the freeze statement are compared by value, not only by shape.
        if sha256_json(json.loads(args.models.read_bytes())) != authorization.receipt["weights_sha256"]:
            raise PanelError("models_differ_from_authorization")
        if sha256_file(args.freeze_statement) != authorization.receipt["freeze_statement_sha256"]:
            raise PanelError("freeze_statement_differs_from_authorization")
    # Every build parses exactly the pinned DAT-02 bytes: split labels decide what may be read.
    registry_bytes = Path(recipe["inputs"]["dat02"]["path"]).read_bytes()
    if sha256_bytes(registry_bytes) != recipe["inputs"]["dat02"]["sha256"]:
        raise PanelError("dat02_pin_mismatch")
    if final:
        for item in recipe["inputs"]["dev_panels"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise PanelError("dev_panel_pin_mismatch")
    registry = json.loads(registry_bytes)
    group_ids = quotas = None
    if not final:
        ordered = sorted((g["group_id"] for g in groups_of(registry, args.split)),
                         key=lambda g: sha256_json([3407, g]))
        group_ids = ordered[:args.groups] if args.groups else None
        if args.scale:
            quotas = {f: max(1, round(q * args.scale)) for f, q in recipe["quotas"].items()}
    out = args.out.resolve()
    if os.path.lexists(out) or out.parent.resolve() != out.parent:
        raise PanelError("output_must_be_fresh_with_real_parent")
    tokenizer = load_tokenizer(args.tokenizer)
    # Everything that needs no final content is checked before build_panel claims the attempt.
    for root in (recipe["inputs"]["corpus_root"], recipe["inputs"]["tarball_root"]):
        if not Path(root).is_dir():
            raise PanelError("source_root_unavailable")
    modules = Modules.load()
    verify_closure(recipe)
    cpt = frozenset() if args.no_cpt_filter else cpt_hashes(recipe)
    dev_prompts = dev_prompt_hashes(recipe)
    work = out.parent / (out.name + ".work")
    work.mkdir(mode=0o700)
    from importlib.metadata import version

    inputs = {"modules": module_pins, "dat02_sha256": recipe["inputs"]["dat02"]["sha256"],
              "distributions": {n: version(n) for n in ("tree-sitter", "tree-sitter-r", "tokenizers",
                                                         "transformers")},
              "tokenizer": str(args.tokenizer), "memberships_sha256":
              sha256_file(args.memberships) if args.memberships else None}
    try:
        panel = build_panel(split=args.split, registry=registry, recipe=recipe, tokenizer=tokenizer,
                            workdir=work, cpt=cpt, dev_prompts=dev_prompts, authorization=authorization,
                            group_ids=group_ids, quotas=quotas, memberships=memberships, modules=modules)
        if panel.receipt["status"] != "built":
            # A stopped build publishes nothing but its aggregate receipt.
            report = {k: v for k, v in panel.receipt.items() if k != "_provenance"}
            report.update(constructor_sha256=sha256_file(Path(__file__)),
                          recipe_sha256=sha256_file(RECIPE_PATH), inputs=inputs)
            out.mkdir(mode=0o700)
        else:
            report = write_panel(panel, out, recipe_sha256=sha256_file(RECIPE_PATH), inputs=inputs)
            pins = authorization.receipt if authorization else {"weights_sha256": "0" * 64,
                                                                "harness_sha256": "0" * 64}
            report["launcher_dry_run"] = launcher_dry_run(
                out, tokenizer, weights_sha256=pins["weights_sha256"], harness_sha256=pins["harness_sha256"],
                rehearsal=not final)
            if authorization is not None:
                report["authorization"] = {k: authorization.receipt[k] for k in (
                    "weights_sha256", "harness_sha256", "memberships_sha256", "freeze_statement_sha256")}
            # The plan keys A3 fixes; A4 copies them unchanged into the launcher plan.
            fragment = {
                "cases": {"path": str(out / "cases.json"), "sha256": report["outputs"]["cases_sha256"]},
                "case_count": report["outputs"]["case_count"],
                "semantic_receipt": {"path": str(out / "semantic-receipt.json"),
                                     "sha256": report["outputs"]["semantic_receipt_sha256"]},
                "payloads_path": str(out / "payloads.json"),
                "source_roots": [report["outputs"]["source_root"]],
                "memberships": {"path": str(args.memberships.resolve()), "sha256": inputs["memberships_sha256"]}
                if args.memberships else None,
            }
            _write_immutable_json(out / "plan-fragment.json", canonical_json(fragment).encode() + b"\n")
        _write_immutable_json(out / "build-receipt.json", canonical_json(report).encode() + b"\n")
        for path in out.glob("*.json"):
            path.chmod(0o400)
    except BaseException:
        if final:
            remove_tree(out)  # unpublished final outputs are deleted unread; the attempt marker stays
        raise
    finally:
        if final:
            # The intermediate finish rows hold final text; nothing outside the panel directory keeps it.
            remove_tree(work)
    return report


def load_tokenizer(path: Path) -> Any:  # noqa: ANN401
    """The campaign tokenizer, after checking its two files against the pinned hashes."""
    from sepalith.campaign_protocol import TOKENIZER_CONFIG_SHA256, TOKENIZER_JSON_SHA256
    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer

    for name, digest in (("tokenizer.json", TOKENIZER_JSON_SHA256), ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256)):
        if sha256_file(Path(path) / name) != digest:
            raise PanelError("tokenizer_differs_from_campaign_pin")
    return load_pinned_reference_tokenizer(path)


def remove_tree(path: Path) -> None:
    if not os.path.lexists(path):
        return
    for directory, _dirs, _names in os.walk(path):
        os.chmod(directory, 0o700)
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.modules.setdefault("final_panel", sys.modules[__name__])
    raise SystemExit(main())
