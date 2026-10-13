#!/usr/bin/env python3
"""Build the DAT-02 metadata-only global source split.

This script deliberately reads only identity and boundary *keys*.  It never
reads a target value, prints a row, or emits training records.  The resulting
manifest is a source-group registry and remains non-training data.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
DATASET = Path("/mnt/h/sepalith/datasets")
INVENTORY = PLAN / "docs/campaign/work/data/DAT-01-source-inventory.json"
HASH_MANIFEST = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-01-source-hashes.json")
SPLIT_VERSION = "DAT-02-global-v2"
DEFAULT_OUTPUT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")
HEX_RE = re.compile(r"^[0-9a-f]{32,128}$", re.I)


def norm_name(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^a-z0-9._/-]+", "", text)


def repo_token(value: object) -> str:
    """Return a conservative owner/repository token for URL-like values."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    known_hosts = {"github.com", "gitlab.com", "bitbucket.org"}
    # Support the common git transport spelling without treating an explicit
    # unknown host as GitHub by default.
    ssh_match = re.fullmatch(r"(?:[^@/:]+@)?([^/:]+):(.+)", text)
    if ssh_match and "://" not in text:
        ssh_host = ssh_match.group(1).lower()
        ssh_path = ssh_match.group(2).strip("/").lower()
        if ssh_host in known_hosts:
            bits = [x for x in ssh_path.split("/") if x]
            if len(bits) >= 2:
                if bits[1].endswith(".git"):
                    bits[1] = bits[1][:-4]
                return f"repo:{ssh_host}/{bits[0]}/{bits[1]}"
        return ""
    if "://" not in text:
        raw_bits = [x for x in text.strip("/").split("/") if x]
    else:
        raw_bits = []
    # urlsplit treats a bare owner/repository as a hostname plus one path
    # segment. Handle that spelling explicitly before URL parsing.
    if "://" not in text and len(raw_bits) == 2 and raw_bits[0].lower() not in known_hosts:
        host = ""
        path = "/".join(raw_bits).lower()
    else:
        parsed = urlsplit(text if "://" in text else "https://" + text)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/").lower()
    if path.endswith(".git"):
        path = path[:-4]
    bits = [x for x in path.split("/") if x]
    if host and len(bits) >= 2 and host in {"github.com", "gitlab.com", "bitbucket.org"}:
        return f"repo:{host}/{bits[0]}/{bits[1]}"
    if not host and len(bits) == 2 and all(bits):
        # A bare owner/repository is the common GitHub spelling. Keep the
        # host so it matches an explicit GitHub URL and cannot merge with a
        # repository on another hosting service.
        return f"repo:github.com/{bits[0]}/{bits[1]}"
    return ""


def package_token(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    text = norm_name(raw)
    if not text or text in {"na", "none", "null", "unknown", "n/a"}:
        return ""
    first_component = text.split("/", 1)[0] if "/" in text else ""
    if (
        "://" in raw
        or first_component in {"github.com", "gitlab.com", "bitbucket.org"}
        or ("/" in text and text.count("/") == 1 and "." in first_component)
    ):
        return repo_token(raw)
    if "/" in text and len(text.split("/")) == 2:
        # package_or_repo frequently contains owner/repository.
        return repo_token(text)
    return f"pkg:{text}"


def hash_parent(value: object) -> str:
    if value is None:
        return ""
    text = norm_name(value)
    return f"parent:{text}" if HEX_RE.fullmatch(text) else ""


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                yield line_no, None
                continue
            # Do not access target/text/prompt/full_prompt values in this pass.
            yield line_no, row


def nonempty(row: dict, keys: list[str]) -> bool:
    return all(key in row and row[key] not in (None, "", []) for key in keys)


def present(row: dict, keys: list[str]) -> bool:
    """Check boundary-key presence while allowing an empty target body."""
    return all(key in row and row[key] is not None for key in keys)


class SplitBuilder:
    def __init__(self, file_hashes: dict[str, dict]) -> None:
        self.dsu = DSU()
        self.file_hashes = file_hashes
        self.token_flags: dict[str, set[str]] = collections.defaultdict(set)
        self.observations: list[dict] = []
        self.candidate_rows: list[dict] = []
        self.source_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.source_files: dict[str, set[str]] = collections.defaultdict(set)
        self.bad_json: collections.Counter = collections.Counter()
        self.alias_edges: collections.Counter = collections.Counter()
        self.token_display: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.candidate_named_tokens: set[str] = set()

    def tokens_for(self, row: dict, source: str, line_no: int) -> tuple[list[str], list[str]]:
        tokens: list[str] = []
        parents: list[str] = []
        # A package name is intentionally unioned with its explicit repository
        # URL. This conservative edge prevents package/repository alias leaks.
        for key in ("repo", "repo_url", "source_url", "upstream", "parent_link"):
            tok = repo_token(row.get(key))
            if tok:
                tokens.append(tok)
                self.token_display[tok][str(row.get(key))[:160]] += 1
        for key in ("package", "package_or_repo"):
            tok = package_token(row.get(key))
            if tok:
                tokens.append(tok)
                self.token_display[tok][str(row.get(key))[:160]] += 1
        for key in ("sha_full", "content_hash", "file_sha", "corpus_key"):
            tok = hash_parent(row.get(key))
            if tok:
                tokens.append(tok)
                parents.append(tok)
        # A deterministic source/edit identity blocks source-site crossings.
        path = norm_name(row.get("path"))
        if path and parents:
            for parent in parents:
                tokens.append(f"edit:{parent}|{path}")
        if not tokens:
            tokens = [f"unresolved:{source}:{line_no}"]
        for token in tokens:
            self.dsu.add(token)
        head = tokens[0]
        for token in tokens[1:]:
            self.dsu.union(head, token)
            self.alias_edges["metadata_or_parent_union"] += 1
        return sorted(set(tokens)), sorted(set(parents))

    def observe(self, path: Path, source: str, phase: str, row: dict | None, line_no: int, *,
                family: str = "", final_capable: bool = False, boundary: bool = False,
                flags: tuple[str, ...] = (), candidate: bool = False) -> None:
        if row is None:
            self.bad_json[source] += 1
            return
        tokens, parents = self.tokens_for(row, source, line_no)
        for token in tokens:
            self.token_flags[token].update(flags)
        self.source_counts[source][phase] += 1
        self.source_files[source].add(str(path))
        if candidate:
            self.candidate_named_tokens.update(
                token for token in tokens if token.startswith(("pkg:", "repo:"))
            )
            self.candidate_rows.append({
                "source": source,
                "family": family or str(row.get("family") or "unknown"),
                "file": str(path),
                "line": line_no,
                "tokens": tokens,
                "parents": parents,
                "boundary": bool(boundary),
                "final_capable": bool(final_capable),
                "phase": phase,
            })

    def flag_file(self, path: Path, source: str, phase: str, flags: tuple[str, ...], *,
                  candidate: bool = False, final_capable: bool = False) -> None:
        for line_no, row in jsonl(path):
            self.observe(path, source, phase, row, line_no, flags=flags,
                         candidate=candidate, final_capable=final_capable)

    def candidate_file(self, path: Path, source: str, phase: str, *, family: str = "",
                       final_capable: bool = False, boundary_fn=None) -> None:
        for line_no, row in jsonl(path):
            boundary = bool(boundary_fn(row)) if row is not None and boundary_fn else False
            self.observe(path, source, phase, row, line_no, family=family,
                         final_capable=final_capable, boundary=boundary, candidate=True)

    def root_info(self) -> tuple[dict[str, dict], dict[str, str]]:
        members: dict[str, list[str]] = collections.defaultdict(list)
        for token in self.dsu.parent:
            members[self.dsu.find(token)].append(token)
        group_id_by_root: dict[str, str] = {}
        groups: dict[str, dict] = {}
        for root, tokens in members.items():
            canonical = "|".join(sorted(tokens))
            gid = "g-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
            group_id_by_root[root] = gid
            flags = set()
            for token in tokens:
                flags.update(self.token_flags.get(token, ()))
            groups[gid] = {
                "group_id": gid,
                "canonical_token_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "token_count": len(tokens),
                "token_type_counts": dict(collections.Counter(t.split(":", 1)[0] for t in tokens)),
                "named_package_count": sum(t.startswith("pkg:") for t in tokens),
                "named_repository_count": sum(t.startswith("repo:") for t in tokens),
                "flags": sorted(flags),
                "identity_forms": sorted(
                    x for x in tokens if x in self.token_display
                )[:12],
                "source_counts": collections.Counter(),
                "families": collections.Counter(),
                "files": set(),
                "candidate_rows": 0,
                "boundary_rows": 0,
                "final_capable_rows": 0,
                "parent_tokens": set(),
            }
        return groups, group_id_by_root


def assign_group_split(flags: set[str] | list[str], candidate_rows: int, bucket: int | None) -> str:
    """Apply the precedence policy to one canonical identity group."""
    flags = set(flags)
    # Historical evaluation and TU3 held-out identities override the
    # historical sft_v7 marker when a group has multiple source identities.
    if "historical_eval" in flags:
        return "quarantine_historical_eval"
    if "tu3" in flags:
        return "quarantine_tu3"
    if "sft_v7" in flags:
        return "train_group" if candidate_rows else "quarantine_sft_v7"
    if candidate_rows == 0:
        return "excluded_source_only"
    if bucket is None:
        return "quarantine_unresolved_identity"
    if bucket < 10:
        return "final_candidate_group"
    if bucket < 20:
        return "dev_group"
    return "train_group"


def alias_fixtures() -> dict[str, bool]:
    """Exercise canonicalization and precedence on metadata-only fixtures."""
    bare = repo_token("Org/Repo")
    github = repo_token("https://github.com/ORG/REPO.git/")
    github_host_qualified = repo_token("github.com/ORG/REPO")
    github_ssh = repo_token("git@github.com:ORG/REPO.git")
    github_trailing = repo_token("https://github.com/org/repo/")
    upstream = repo_token("https://github.com/org/upstream.git")
    gitlab = repo_token("https://gitlab.com/org/repo.git")
    unknown_host = repo_token("https://example.org/owner/repo")
    assert bare == github == github_host_qualified == github_ssh == github_trailing == "repo:github.com/org/repo"
    assert gitlab == "repo:gitlab.com/org/repo"
    assert gitlab != github
    assert unknown_host == ""

    fork_builder = SplitBuilder({})
    fork_tokens, _ = fork_builder.tokens_for(
        {"repo": "https://github.com/org/fork.git", "upstream": "https://github.com/org/upstream.git"},
        "fixture_fork", 1
    )
    assert upstream in fork_tokens
    assert fork_builder.dsu.find("repo:github.com/org/fork") == fork_builder.dsu.find(upstream)

    fixture_builder = SplitBuilder({})
    left, left_parents = fixture_builder.tokens_for(
        {"repo": "Org/Repo", "path": "R/a.R", "content_hash": "a" * 64},
        "fixture_left", 1
    )
    right, right_parents = fixture_builder.tokens_for(
        {"source_url": "https://github.com/ORG/REPO.git", "path": "R/a.R", "content_hash": "a" * 64},
        "fixture_right", 1
    )
    assert set(left) & set(right)
    assert left_parents == right_parents == ["parent:" + "a" * 64]
    assert fixture_builder.dsu.find(left[0]) == fixture_builder.dsu.find(right[0])

    assert assign_group_split({"sft_v7"}, 1, 5) == "train_group"
    assert assign_group_split({"sft_v7"}, 0, 5) == "quarantine_sft_v7"
    assert assign_group_split({"sft_v7", "historical_eval"}, 1, 5) == "quarantine_historical_eval"
    assert assign_group_split({"sft_v7", "tu3"}, 1, 5) == "quarantine_tu3"
    return {
        "bare_owner_repo_equals_github_url": True,
        "host_qualified_github_and_ssh_alias_supported": True,
        "trailing_git_and_case_normalized": True,
        "github_upstream_alias_supported": True,
        "gitlab_host_remains_independent": True,
        "unknown_explicit_host_not_guessed": True,
        "source_parent_crossing_unifies": True,
        "historical_eval_overrides_sft_v7": True,
        "tu3_overrides_sft_v7": True,
        "sft_v7_candidate_routes_train": True,
        "sft_v7_source_only_quarantines": True,
    }


def p(rel: str) -> Path:
    return DATASET / rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    fixture_results = alias_fixtures()
    inventory = json.loads(INVENTORY.read_text())
    hash_data = json.loads(HASH_MANIFEST.read_text())
    file_hashes = {x["path"]: x for x in hash_data["files"]}
    builder = SplitBuilder(file_hashes)

    # Historical/exclusion identities are unioned first so a candidate that
    # shares a package, fork alias, or exact parent cannot escape its group.
    sft7 = p("sft_v7/train.jsonl")
    sft3eval = p("sft_v3/eval.jsonl")
    print("reading historical exclusion identities", flush=True)
    builder.flag_file(sft7, "sft_v7", "historical_train", ("sft_v7",))
    builder.flag_file(sft3eval, "sft_v3_eval", "historical_eval", ("historical_eval", "sft_v3_eval"))
    builder.flag_file(p("edit_pairs_v1/eval.jsonl"), "edit_pairs_eval", "historical_eval", ("historical_eval",))
    builder.flag_file(p("nextcoder_r_v1/eval.jsonl"), "b11_eval", "historical_eval", ("historical_eval", "b11_eval"))
    builder.flag_file(p("nextcoder_r2_cross_v1/eval.jsonl"), "tu3_eval", "historical_eval", ("historical_eval", "tu3"))

    # Primary structured source candidates. Scenario batteries are represented
    # for global identity but remain final-ineligible historical batteries.
    print("reading edit/scenario/case candidates", flush=True)
    edit = p("edit_pairs_v1/examples.jsonl")
    builder.candidate_file(edit, "edit_pairs_train", "train", family="edit_pairs",
                           final_capable=True,
                           boundary_fn=lambda r: nonempty(r, ["repo", "sha_full", "path", "prefix", "region_old", "region_new", "suffix", "cursor_idx"]))
    scenario_det = [
        "rename_propagation", "pipe_rewrite", "na_rm_propagation", "format_propagation",
        "doc_sync", "no_op", "comment_insert",
    ]
    scenario_teacher = [
        "comment_drafting", "roxygen_drafting", "mid_roxygen", "comment_to_code_real",
        "comment_to_code_synthetic", "comment_to_code_gemini",
    ]
    for fam in scenario_det + scenario_teacher:
        path = p(f"scenarios_v1/{fam}.jsonl")
        builder.candidate_file(path, f"scenario_{fam}", "historical_scenario",
                               family=fam, final_capable=False,
                               boundary_fn=lambda r: present(r, ["package", "path", "prefix", "region_old", "region_new", "suffix", "cursor_idx"]))

    # All non-base, non-ASTFIM case records retain structured source-parent
    # metadata. They remain conditional until DAT-03 validators run.
    case_dir = DATASET / "cases_v1"
    for path in sorted(case_dir.glob("*.jsonl")):
        name = path.name
        if (name.endswith(".done.jsonl") or name.endswith("_bases.jsonl")
                or name == "base_samples_spark.jsonl"
                or name.startswith("astfim_partial")):
            continue
        family = name.removesuffix(".jsonl")
        builder.candidate_file(path, f"case_{family}", "case_candidate", family=family,
                               final_capable=True,
                               boundary_fn=lambda r: present(r, ["package", "path", "prefix", "region_old", "region_new", "suffix", "cursor_idx"]) and any(nonempty(r, [k]) for k in ["content_hash", "corpus_key", "base_sample_id"]))

    # B11 train rows are part of the global identity map but need authored
    # joins and stay replay-only. Authored/seeds are identity evidence.
    builder.flag_file(p("nextcoder_r_v1/authored.jsonl"), "b11_authored", "b11_authored", ("b11",))
    builder.flag_file(p("nextcoder_r_v1/seeds.jsonl"), "b11_seeds", "b11_seed", ("b11",))
    builder.flag_file(p("nextcoder_r_v1/train.jsonl"), "b11_train", "b11_train", ("b11",))
    builder.flag_file(p("nextcoder_r2_cross_v1/authored.jsonl"), "tu3_authored", "tu3_authored", ("tu3",))
    builder.flag_file(p("nextcoder_r2_cross_v1/seeds.jsonl"), "tu3_seeds", "tu3_seed", ("tu3",))
    builder.flag_file(p("nextcoder_r2_cross_v1/train.jsonl"), "tu3_train", "tu3_train", ("tu3",))

    # Real trace and environment identities are retained as historical
    # groups. They cannot become SFT rows because this pass never adds a
    # target boundary for them.
    builder.flag_file(p("spec_traces/traces.jsonl"), "spec_traces", "historical_trace", ("historical_trace",))
    builder.flag_file(p("loc1_s0_r/set.jsonl"), "loc1_set", "environment", ("environment",))
    builder.flag_file(p("loc1_s0_r/corpus.jsonl"), "loc1_corpus", "environment", ("environment",))
    for name in ("trajectories.jsonl", "trajectories_v2.jsonl"):
        builder.flag_file(p(f"sim_trajectories_v1/{name}"), "sim_trajectories", "historical_trajectory", ("historical_trajectory",))

    groups, group_id_by_root = builder.root_info()
    # Attach candidate metadata only after all unions are complete.
    parent_splits: dict[str, set[str]] = collections.defaultdict(set)
    source_group_sets: dict[str, set[str]] = collections.defaultdict(set)
    for row in builder.candidate_rows:
        root = builder.dsu.find(row["tokens"][0])
        gid = group_id_by_root[root]
        g = groups[gid]
        g["candidate_rows"] += 1
        g["boundary_rows"] += int(row["boundary"])
        g["final_capable_rows"] += int(row["final_capable"] and row["boundary"])
        g["source_counts"][row["source"]] += 1
        g["families"][row["family"]] += 1
        g["files"].add(row["file"])
        g["parent_tokens"].update(row["parents"])
        source_group_sets[row["source"]].add(gid)

    # root_info has already gathered flags from every token in each group.
    # Keep this pass linear in group count; do not rescan all DSU tokens for
    # every group.
    for gid, g in groups.items():
        g["flags"] = sorted(g["flags"])
        g["source_counts"] = dict(g["source_counts"])
        g["families"] = dict(g["families"])
        g["files"] = sorted(g["files"])
        g["parent_tokens"] = sorted(g["parent_tokens"])

    # Assignment is over canonical groups, never rows. The bucket rule is
    # stable and yields 80/10/10 train/dev/final before exclusions. Historical
    # eval/TU3 held-out identities override sft_v7. A sft_v7 group with a
    # supported candidate row remains train-only and receives an explicit
    # final/dev-ineligible marker; source-only sft_v7 identities quarantine.
    for gid, g in groups.items():
        if not g["token_type_counts"].get("unresolved", 0):
            bucket = int(hashlib.sha256(f"{SPLIT_VERSION}|{gid}".encode()).hexdigest()[:8], 16) % 100
        else:
            bucket = None
        if "sft_v7" in g["flags"] and g["candidate_rows"]:
            g["flags"] = sorted(set(g["flags"]) | {"final/dev_ineligible:sft_v7"})
        g["split"] = assign_group_split(g["flags"], g["candidate_rows"], bucket)

    # Final-eligible rows are explicit structured edit/case rows only. A
    # historical scenario group can be assigned to a group split for leakage
    # control but contributes no final row until regenerated.
    split_rows: collections.Counter = collections.Counter()
    split_sources: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    final_groups: set[str] = set()
    final_eligible_rows = 0
    for row in builder.candidate_rows:
        root = builder.dsu.find(row["tokens"][0]); gid = group_id_by_root[root]; split = groups[gid]["split"]
        split_rows[split] += 1
        split_sources[split][row["source"]] += 1
        if row["final_capable"] and row["boundary"] and split == "final_candidate_group":
            final_groups.add(gid); final_eligible_rows += 1

    # Every source parent/edit token may occur in one group only. Since split
    # is attached to group IDs, this check catches accidental alias crossings.
    group_parent_splits: dict[str, set[str]] = collections.defaultdict(set)
    for gid, g in groups.items():
        for parent in g["parent_tokens"]:
            group_parent_splits[parent].add(g["split"])
    parent_crossings = {p: sorted(s) for p, s in group_parent_splits.items() if len(s) > 1}

    # Keep only a bounded summary of groups in-repo; the complete row-free
    # registry is written to data-work as the non-training artifact.
    group_out = []
    for gid in sorted(groups):
        g = groups[gid]
        group_out.append({k: g[k] for k in ["group_id", "canonical_token_sha256", "token_count", "token_type_counts", "named_package_count", "named_repository_count", "flags", "identity_forms", "parent_tokens", "source_counts", "families", "files", "candidate_rows", "boundary_rows", "final_capable_rows", "split"]})

    split_id_material = {
        "algorithm": f"sha256({SPLIT_VERSION}|canonical-token-union)",
        "groups": group_out,
        "input_inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        "input_hash_manifest_sha256": hashlib.sha256(HASH_MANIFEST.read_bytes()).hexdigest(),
    }
    split_id = SPLIT_VERSION + "-" + hashlib.sha256(json.dumps(split_id_material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    output = {
        "task": "DAT-02",
        "status": "partial_with_deterministic_split",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_id": split_id,
        "purpose": "non-training global package/repository/source split metadata",
        "inputs": {
            "inventory": str(INVENTORY),
            "inventory_sha256": split_id_material["input_inventory_sha256"],
            "hash_manifest": str(HASH_MANIFEST),
            "hash_manifest_sha256": split_id_material["input_hash_manifest_sha256"],
            "hashed_input_files": hash_data["file_count"],
            "raw_inputs_read_only": True,
            "final_records_opened": False,
        },
        "identity_algorithm": {
            "package_normalization": "NFKC, lowercase, remove whitespace/non-name punctuation",
            "repository_normalization": "canonical host/owner/repository from repo/repo_url/source_url/upstream; bare owner/repository defaults to github.com; strip .git and lowercase",
            "union_edges": ["same normalized package name", "explicit repository/source URL", "upstream URL", "full parent/content/file hash", "parent-hash + path source edit"],
            "fork_policy": "explicit upstream/repository metadata unions a fork with its upstream; absent evidence is not guessed",
            "unresolved_policy": "rows without a package/repository or parent token get a unique unresolved group and cannot enter final",
            "trajectory_policy": "trajectory files are grouped as complete historical trajectory identities and excluded from candidate splits",
            "assignment": "sha256(DAT-02-global-v2|group_id) mod 100: <10 final, 10-19 dev, otherwise train; historical_eval then tu3 override sft_v7; candidate-bearing sft_v7 is train-only with final/dev_ineligible:sft_v7",
            "alias_fixtures": fixture_results,
        },
        "render_semantics": {
            "no_op": "Raw legacy empty no_op fields are a sentinel. Semantic region_new must equal region_old; serialized model body is [NO_EDIT] followed by the existing UPDATED terminator. Never interpret the empty sentinel as deletion.",
            "deletion": "An empty region_new is deletion only when the authoritative source/scorer identifies a non-empty replaced range.",
            "tokenizer_guard": "PRM-04-special-token-mode evidence supports split_special_tokens=True with no auto-added specials matching native parse:false for 8/8 fixtures, including literal </s>. DAT-03 must validate full fixtures and the explicit BOS/EOS policy. DAT-02 does not inspect target values and applies no blanket literal-EOS exclusion.",
        },
        "counts": {
            "source_records": len(inventory["sources"]),
            "identity_groups_total": len(groups),
            "candidate_identity_groups": sum(g["candidate_rows"] > 0 for g in groups.values()),
            "candidate_rows_observed": len(builder.candidate_rows),
            "candidate_rows_with_boundary": sum(int(x["boundary"]) for x in builder.candidate_rows),
            "candidate_rows_final_capable_before_exclusions": sum(int(x["final_capable"] and x["boundary"]) for x in builder.candidate_rows),
            "groups_by_split": dict(collections.Counter(g["split"] for g in groups.values())),
            "candidate_rows_by_split": dict(split_rows),
            "candidate_rows_by_split_source": {k: dict(v) for k, v in split_sources.items()},
            "actual_clean_final_candidate_packages": len(final_groups),
            "actual_clean_final_candidate_rows": final_eligible_rows,
            "named_package_identity_tokens_total": sum(t.startswith("pkg:") for t in builder.dsu.parent),
            "named_repository_identity_tokens_total": sum(t.startswith("repo:") for t in builder.dsu.parent),
            "named_package_identity_groups_total": sum(g["named_package_count"] > 0 for g in groups.values()),
            "named_repository_identity_groups_total": sum(g["named_repository_count"] > 0 for g in groups.values()),
            "candidate_named_package_identity_tokens": sum(t.startswith("pkg:") for t in builder.candidate_named_tokens),
            "candidate_named_repository_identity_tokens": sum(t.startswith("repo:") for t in builder.candidate_named_tokens),
            "actual_clean_final_candidate_named_packages": sum(g["named_package_count"] for gid, g in groups.items() if gid in final_groups),
            "actual_clean_final_candidate_named_repositories": sum(g["named_repository_count"] for gid, g in groups.items() if gid in final_groups),
            "actual_clean_final_families": dict(
                collections.Counter(
                    family
                    for gid in final_groups
                    for family, count in groups[gid]["families"].items()
                    for _ in range(count)
                )
            ),
            "sft_v7_groups_excluded": sum("sft_v7" in g["flags"] for g in groups.values()),
            "sft_v7_candidate_groups_train_only": sum("sft_v7" in g["flags"] and g["candidate_rows"] > 0 and g["split"] == "train_group" for g in groups.values()),
            "sft_v7_source_only_groups_quarantined": sum("sft_v7" in g["flags"] and g["candidate_rows"] == 0 and g["split"] == "quarantine_sft_v7" for g in groups.values()),
            "historical_eval_groups_quarantined": sum("historical_eval" in g["flags"] for g in groups.values()),
            "unresolved_groups_quarantined": sum(g["split"] == "quarantine_unresolved_identity" for g in groups.values()),
            "parent_split_crossings": len(parent_crossings),
            "bad_json_lines_by_source": dict(builder.bad_json),
        },
        "source_roles": {
            "final_candidate_sources": ["edit_pairs_train", "case_* rows with explicit source parent and boundary"],
            "group_only_historical_sources": ["scenario_*", "spec_traces", "loc1_*", "sim_trajectories"],
            "replay_only_sources": ["b11_train", "tu3_train"],
            "quarantine_sources": ["sft_v7 source-only identities", "sft_v3_eval", "*_eval", "TU3 contradiction pack", "materialized SFT/RL derivatives"],
        },
        "checks": {
            "split_group_disjoint": all(g["split"] for g in groups.values()),
            "source_parent_no_cross_split": not parent_crossings,
            "historical_sft_v7_excluded_from_final": all(g["split"] != "final_candidate_group" or "sft_v7" not in g["flags"] for g in groups.values()),
            "historical_eval_excluded_from_final": all(g["split"] != "final_candidate_group" or "historical_eval" not in g["flags"] for g in groups.values()),
            "tu3_excluded": all(g["split"] != "final_candidate_group" or "tu3" not in g["flags"] for g in groups.values()),
            "sft_v7_candidate_train_only_or_held_out_override": all(
                not ("sft_v7" in g["flags"] and g["candidate_rows"] > 0)
                or "historical_eval" in g["flags"]
                or "tu3" in g["flags"]
                or g["split"] == "train_group"
                for g in groups.values()
            ),
            "sft_v7_candidate_groups_flagged_final_dev_ineligible": all(
                not ("sft_v7" in g["flags"] and g["candidate_rows"] > 0)
                or "final/dev_ineligible:sft_v7" in g["flags"]
                for g in groups.values()
            ),
            "final_records_opened": False,
            "raw_inputs_changed": False,
        },
        "unresolved": [
            "Package/repository aliases are unioned only where package, URL, upstream, or parent-hash evidence exists; undocumented forks remain separate/quarantined.",
            "Clean counts are identity/boundary counts, not DAT-03 validator-passing counts.",
            "Historical scenario batteries are group-mapped for leakage control but are final-ineligible until regenerated from raw parents.",
            "B11 train rows require authored/seeds join; TU3 remains held out for contradictory labels.",
            "No row targets or proposed final answers were inspected or emitted.",
        ] + ([
            f"{sum(builder.bad_json.values())} malformed JSON lines were omitted from candidate rows: "
            + ", ".join(f"{source}={count}" for source, count in sorted(builder.bad_json.items()))
        ] if builder.bad_json else []),
        "groups": group_out,
        "reproduction": {
            "script": str(PLAN / "docs/campaign/work/data/DAT-02-build-split-v2.py"),
            "command": f"python3 {PLAN / 'docs/campaign/work/data/DAT-02-build-split-v2.py'} --output {DEFAULT_OUTPUT}",
            "cpu_threads": 2,
            "large_output_label": "non-training metadata only",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print("split_id", split_id, flush=True)
    print("groups", len(groups), "candidate_rows", len(builder.candidate_rows), "final_packages", len(final_groups), "final_rows", final_eligible_rows, flush=True)
    print("splits", json.dumps(output["counts"]["groups_by_split"], sort_keys=True), flush=True)
    print("source_rows", json.dumps(output["counts"]["candidate_rows_by_split_source"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
