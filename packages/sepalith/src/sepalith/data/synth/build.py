"""Build SYNTH data: seeds -> normalised sources -> cases -> verified, deduplicated,
decontaminated, distribution-shaped drop-in TRAIN rows plus sidecar, ledgers
and manifest.

    # 1. normalise only (writes sources/), 2. CPT check on sources/, 3. full build
    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.data.synth.build \\
        --seeds docs/training/plan-20260927/synth/seeds-v2 --out .../SYNTH-v2 --normalise-only
    PYTHONPATH=... python -m sepalith.data.synth.cpt_check --sources .../SYNTH-v2/sources --out .../SYNTH-v2/cpt-hits.json
    PYTHONPATH=... python -m sepalith.data.synth.build --seeds ... --out .../SYNTH-v2 --cpt-hits .../SYNTH-v2/cpt-hits.json

Sealed data: ``guard_path`` resolves every path (symlinks included) and refuses
anything under the sealed-final locations named in CONTEXT.md or containing a
``final``/``sealed`` path component. The DEV check hashes the DEV panels and
DEV-split construction sources in ``DEV_SOURCES``; only aggregate counts are
reported. v2 does not read the DAT-02 split file (v1 read it for package names).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

from sepalith.campaign_protocol import TokenEncoder
from sepalith.data.synth import GENERATOR_ID, GENERATOR_VERSION
from sepalith.data.synth import cpt_check, dedup, finish_block, format_propagation, normalize, profile, roxygen
from sepalith.data.synth.cases import Case
from sepalith.data.synth.structure import describe
from sepalith.training.rl.air_canonical import AirFormatter

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586")
TRAIN_ROWS = DATA / "DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl"
TRAIN_ROWS_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
DEV_SOURCES = (
    "/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl",
    str(DATA / "C05-dev250-v1r3/dev250-v1.jsonl"),
    str(DATA / "C05-dev250-v1r3-phase1/*.jsonl"),
    str(DATA / "C05-dev250-v1r3-phase1/sources/*"),
    str(DATA / "C05-dev-finish-authored-pilot/*.jsonl"),
    str(DATA / "C05-dev-finish-authored-pilot2/*.jsonl"),
    str(DATA / "C05-dev-finish-authored-v1/*.jsonl"),
    str(DATA / "C05-dev-finish-cut-v1/*.jsonl"),
    str(DATA / "DAT-07-dev-source-snapshots/*"),
    str(DATA / "DAT-07-dev-panel-candidates.json"),
    str(DATA / "DAT-07-dev-completion-candidates-v1.jsonl"),
)
# Sealed-final locations named in docs/training/plan-20260927/CONTEXT.md.
FORBIDDEN_PREFIXES = (
    str(DATA / "DAT-07-final-"),
    os.path.expanduser("~/.local/state/sepalith/campaign-20260915/sealed-final"),
)
FORBIDDEN_COMPONENT_RE = re.compile(r"final|sealed", re.IGNORECASE)
BEHAVIOUR_HARNESS = Path(__file__).with_name("r_behaviour.R")

# Decontamination thresholds (token fingerprints over the edit span; k = 12).
DEV_CONTAINMENT_MAX = 0.3
DEV_RUN_MAX = 24
TRAIN_CONTAINMENT_MAX = 0.5
TRAIN_RUN_MAX = 40
CPT_CONTAINMENT_MAX = 0.5
CPT_RUN_MAX = 40
NEAR_DUP_THRESHOLD = 0.8

# Selection shaping (finish_block) and family mix.
PER_UNIT_MAX = 3  # v3: up to 3 cuts per function, at most PER_UNIT_BUCKET_MAX in one length bucket
PER_UNIT_BUCKET_MAX = 2
TEST_FILE_MAX_SHARE = 0.01
FULL_BODY_MAX_SHARE = 0.148  # v3: TRAIN's measured full-body share (confirmed by both reviewers in round 2)
ROXYGEN_MAX_SHARE = 0.10
ROXYGEN_TAG_ORDER_MAX_SHARE = 0.2
FB_BUCKETS = ("2", "3-4", "5-8", "9-12", "13-20", "21-30", "31-45")
PREFIX_BUCKET_NAMES = ("<=10", "11-20", "21-40", "41-60", ">60")
DEV_MASKED_LONG_MAX = 0.8  # drop masked-DEV containment >= 0.8 when the target has >= 9 lines
COHORT_SHIFT_MAX = 0.03  # max change of any finish_block length-bucket share in TRAIN + SYNTH
SYNTH_BUCKET_RATIO_MAX = 2.0  # SYNTH alone: no bucket above twice its TRAIN share


def guard_path(path: str | os.PathLike[str]) -> str:
    """Refuse sealed/final data by string and by resolved path (symlinks followed)."""
    text = os.path.expanduser(str(path))
    resolved = os.path.realpath(text)
    for candidate in (text, resolved):
        if any(candidate.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise SystemExit(f"refusing to open a sealed path: {path}")
        if any(FORBIDDEN_COMPONENT_RE.search(part) for part in Path(candidate).parts):
            raise SystemExit(f"refusing to open a path with a final/sealed component: {path}")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_tokenizer() -> TokenEncoder:
    from transformers import AutoTokenizer

    from sepalith.campaign_protocol import TOKENIZER_JSON_SHA256
    if sha256_file(PARENT / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise SystemExit("parent tokenizer differs from the pinned tokenizer")
    return AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True, use_fast=True, trust_remote_code=False)


def make_scorer(workers: int) -> tuple[Any, AirFormatter, Any]:
    from sepalith.training.rl.campaign_r_parse_probe import DEFAULT_HARNESS, RParseOnlyProbe, sha_file
    from sepalith.training.rl.reward_v21 import RewardV21Config, RewardV21Scorer, tokenizer_decoder
    probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS))
    formatter = AirFormatter(("air",))
    scorer = RewardV21Scorer(probe, formatter, RewardV21Config(), workers=workers)
    return scorer, formatter, tokenizer_decoder(PARENT / "tokenizer.json")


def git_head() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() or None


# ---------------------------------------------------------------------------
# Stage 1: seeds -> normalised sources, behavioural checks
# ---------------------------------------------------------------------------
def seed_files(seeds: Path) -> list[Path]:
    """Seed sources; ``<batch>/checks/`` holds behavioural checks, not seeds."""
    return sorted(p for p in seeds.rglob("*.R") if p.is_file() and p.relative_to(seeds).parts[1] != "checks")


def batch_roots(roots: Sequence[Path]) -> dict[str, Path]:
    """Batch directory -> seed root; batch names must be unique across roots."""
    out: dict[str, Path] = {}
    for root in roots:
        for batch in sorted(p.name for p in root.iterdir() if p.is_dir()):
            if batch in out:
                raise SystemExit(f"batch {batch} appears in two seed roots")
            out[batch] = root
    return out


def normalise(roots: Sequence[Path], out: Path, formatter: AirFormatter, blank_policy: str
              ) -> tuple[list[dict], list[tuple[Path, str]]]:
    batch_roots(roots)
    files = [(root, path) for root in roots for path in seed_files(root)]
    structures = describe([path for _root, path in files])
    ledger = []
    normalised = []
    for root, path in files:
        relative = str(path.relative_to(root))
        text, info = normalize.normalise_file(path, relative, structures[str(path)], formatter,
                                              blank_policy=blank_policy)
        ledger.append(info)
        if text is None:
            continue
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        normalised.append((target, relative))
    return ledger, normalised


def package_and_path(relative: str) -> tuple[str, str]:
    """``b01-x/<pkg>/R/a.R`` -> (``synth:<pkg>``, ``R/a.R``)."""
    parts = relative.split("/")
    return f"synth:{parts[1]}", "/".join(parts[2:])


def behaviour_checks(roots: Sequence[Path], normalised: list[tuple[Path, str]]) -> dict[str, dict]:
    """Run ``<batch>/checks/<rest>`` against each normalised source that has one."""
    out: dict[str, dict] = {}
    by_batch = batch_roots(roots)
    for source, relative in normalised:
        batch, rest = relative.split("/", 1)
        seeds = by_batch[batch]
        check = seeds / batch / "checks" / rest
        if not check.is_file():
            out[relative] = {"status": "no_check", "entered": []}
            continue
        try:
            proc = subprocess.run(["nice", "-n", "10", "Rscript", "--vanilla", str(BEHAVIOUR_HARNESS),
                                   str(source), str(check)], capture_output=True, text=True, timeout=180,
                                  check=False)
            lines = dict(line.split("\t", 1) for line in proc.stdout.strip().split("\n") if "\t" in line)
            status = lines.get("status", "fail\tno output")
            entered = [name for name in lines.get("entered", "").split(",") if name]
        except subprocess.TimeoutExpired:
            status, entered = "fail\ttimeout", []
        # pass/fail comes from the plain run; ``entered`` from the traced run (trace() on every
        # closure the seed defines), i.e. units whose body executed while the check ran.
        out[relative] = {"status": "pass" if status == "ok" else "fail", "detail": status[:200],
                         "check": str(check.relative_to(seeds)), "entered": entered}
    return out


# ---------------------------------------------------------------------------
# Stage 2: cases
# ---------------------------------------------------------------------------
def generate_cases(normalised: list[tuple[Path, str]], formatter: AirFormatter, families: set[str]) -> list[Case]:
    structures = describe([p for p, _ in normalised])
    # Function names defined anywhere in each package (for the invisible-helper rule).
    package_functions: dict[str, set[str]] = defaultdict(set)
    for path, relative in normalised:
        if structures[str(path)].ok:
            package_functions[package_and_path(relative)[0]].update(
                u.name for u in structures[str(path)].units if u.kind == "assign_function" and u.name)
    cases: list[Case] = []
    for path, relative in normalised:
        structure = structures[str(path)]
        if not structure.ok:
            continue
        lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
        package_id, rel_path = package_and_path(relative)
        produced: list[Case] = []
        if "finish_block" in families:
            produced += finish_block.generate(lines, structure, package_id, rel_path,
                                              frozenset(package_functions[package_id]))
        if "roxygen_drafting" in families:
            produced += roxygen.generate(lines, structure, package_id, rel_path)
        if "format_propagation" in families:
            produced += format_propagation.generate(lines, structure, package_id, rel_path, formatter)
        for case in produced:
            case.meta["seed"] = relative
            case.meta["batch"] = relative.split("/")[0]
            case.meta["unit_key"] = f"{relative}:{case.meta.get('unit_lines', [0])[0]}"
        cases += produced
    return cases


# ---------------------------------------------------------------------------
# Stage 4: decontamination references
# ---------------------------------------------------------------------------
def train_indexes() -> tuple[dedup.ReferenceIndex, dedup.TokenIndex, dedup.TokenIndex]:
    if sha256_file(TRAIN_ROWS) != TRAIN_ROWS_SHA256:
        raise SystemExit("TRAIN rows hash mismatch")
    lines4 = dedup.ReferenceIndex("train")
    exact = dedup.TokenIndex("train")
    masked = dedup.TokenIndex("train-masked", masked=True)
    with TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            text = row["prompt_text"] + "\n" + row["target_body_text"]
            lines4.grams |= dedup.line_4grams(text)
            exact.add(text)
            masked.add(text)
    exact.freeze()
    masked.freeze()
    return lines4, exact, masked


def dev_indexes() -> tuple[dedup.ReferenceIndex, dedup.TokenIndex, dedup.TokenIndex, dict]:
    lines4 = dedup.ReferenceIndex("dev")
    exact = dedup.TokenIndex("dev")
    masked = dedup.TokenIndex("dev-masked", masked=True)
    files: list[str] = []
    for pattern in DEV_SOURCES:
        files += sorted(glob.glob(guard_path(pattern)))
    inputs = {}
    for path in files:
        guard_path(path)
        inputs[path] = sha256_file(Path(path))
        for text in dedup.texts_of_file(path):
            lines4.documents += 1
            lines4.grams |= dedup.line_4grams(text)
            exact.add(text)
            masked.add(text)
    exact.freeze()
    masked.freeze()
    return lines4, exact, masked, {"files": len(files), "documents_hashed": lines4.documents, "sha256": inputs}


def load_cpt(path: Path | None, sources: Path) -> tuple[set[int], dedup.TokenIndex, dict | None, bool]:
    """CPT hits, bound to the current source tree, the pinned corpus and tokenizer."""
    index = dedup.TokenIndex("cpt")
    if path is None:
        return set(), index, None, False
    data = json.loads(guard_path(path) and path.read_text())
    expected = cpt_check.binding(sources)
    if data.get("schema") != cpt_check.SCHEMA or data.get("binding") != expected:
        raise SystemExit(f"CPT hit file binding differs: {data.get('binding')} != {expected}")
    index.add_hashes(np.array(data["hit_token_grams"], dtype=np.uint64))
    index.freeze()
    info = {k: v for k, v in data.items() if not k.startswith("hit_") or k.endswith("_count")}
    return set(data["hit_line_grams"]), index, info, True


def dedup_key(case: Case) -> set[int]:
    lead = case.doc[max(case.window_start, case.region_start - 3):case.region_start]
    return dedup.shingles("\n".join([*lead, *case.region_old, "=>", *case.target]))


# ---------------------------------------------------------------------------
# Stage 5: distribution shaping
# ---------------------------------------------------------------------------
def train_cells() -> dict[str, dict[tuple[bool, bool], float]]:
    """TRAIN finish_block: per length bucket, the share of (has blank, has comment)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for _family, _context, target in profile.train_items({"finish_block"}):
        n_lines = profile.target_length(_context["region_old"], target)
        if target and not target[0].strip():
            target = target[1:]  # the leading blank line is a TRAIN-only artefact
        counts[profile.bucket(n_lines)][(any(not x.strip() for x in target),
                                             any(profile.is_comment(x) for x in target))] += 1
    return {b: {cell: n / sum(c.values()) for cell, n in c.items()} for b, c in counts.items()}


def train_bucket_counts() -> Counter:
    """TRAIN finish_block target-length buckets under ``profile.target_length``."""
    return Counter(profile.bucket(profile.target_length(c["region_old"], t))
                   for _f, c, t in profile.train_items({"finish_block"}))


def case_length(case: Case) -> int:
    return profile.target_length(case.region_old, case.target)


def case_prefix_bucket(case: Case) -> str:
    return profile.prefix_bucket(case.region_start - case.window_start)


def train_prefix_shares() -> dict[str, float]:
    counts = Counter(profile.prefix_bucket(p) for p, _t in profile.train_finish_geometry())
    total = sum(counts.values())
    return {name: counts[name] / total for name in PREFIX_BUCKET_NAMES}


def cell_order(candidates: list, cells: dict[tuple[bool, bool], float]) -> list:
    """Interleave a bucket's candidates (cut groups) so any prefix follows TRAIN's
    (blank, comment) cell mix."""
    by_cell: dict[tuple[bool, bool], list] = defaultdict(list)
    for it in candidates:
        target = it[0][0].target
        by_cell[(any(not x.strip() for x in target), any(profile.is_comment(x) for x in target))].append(it)
    taken: Counter = Counter()
    ordered = []
    while any(by_cell.values()):
        # the cell furthest below its TRAIN share (among non-empty cells) goes next
        n = len(ordered) + 1
        cell = max((c for c in by_cell if by_cell[c]), key=lambda c: cells.get(c, 0.0) * n - taken[c])
        ordered.append(by_cell[cell].pop(0))
        taken[cell] += 1
    return ordered


def shape_finish(items: list, rejected: Counter) -> tuple[list, dict]:
    """Per-unit cap, then a greedy cohort-level length rule: SYNTH is added only
    while the TRAIN + SYNTH finish_block length distribution stays within
    COHORT_SHIFT_MAX of TRAIN's in every bucket (both directions); inside each
    bucket candidates follow TRAIN's (blank line, comment) mix; finally the
    full-body and test-file caps. ``items`` are (case, result)."""
    from sepalith.data.synth.common import rng_for
    report: dict = {}
    for item in items:
        item[0].meta["length_bucket"] = profile.bucket(case_length(item[0]))
        item[0].meta["prefix_bucket"] = case_prefix_bucket(item[0])
    # Cut groups: the packaging alternatives of one cut (same target).
    cut_groups: dict[str, list] = defaultdict(list)
    for item in sorted(items, key=lambda it: (it[0].meta.get("alt_rank", 0), it[0].row_id())):
        cut_groups[item[0].meta.get("cut_key") or item[0].row_id()].append(item)
    by_unit: dict[str, list] = defaultdict(list)
    for key in sorted(cut_groups, key=lambda k: cut_groups[k][0][0].row_id()):
        by_unit[cut_groups[key][0][0].meta["unit_key"]].append(cut_groups[key])
    pool = []
    for unit_items in by_unit.values():
        seen_buckets: Counter = Counter()
        kept = []
        # distinct length buckets first, then a second cut in an already used bucket
        for allowance in (1, PER_UNIT_BUCKET_MAX):
            for item in unit_items:
                b = item[0][0].meta["length_bucket"]
                if item not in kept and len(kept) < PER_UNIT_MAX and seen_buckets[b] < allowance:
                    kept.append(item)
                    seen_buckets[b] += 1
        rejected[("finish_block", "per_unit_cap")] += len(unit_items) - len(kept)
        pool += kept
    outside = [g for g in pool if g[0][0].meta["length_bucket"] not in FB_BUCKETS]
    rejected[("finish_block", "length_bucket_outside_train_range")] += len(outside)
    pool = [g for g in pool if g[0][0].meta["length_bucket"] in FB_BUCKETS]
    train = train_bucket_counts()
    train_total = sum(train[b] for b in FB_BUCKETS)
    train_share = {b: train[b] / train_total for b in FB_BUCKETS}
    cells = train_cells()
    queues = {}
    for b in FB_BUCKETS:
        candidates = [g for g in pool if g[0][0].meta["length_bucket"] == b]
        rng_for("shape", b).shuffle(candidates)
        queues[b] = cell_order(candidates, cells.get(b, {}))
    added: Counter = Counter()
    added_prefix: Counter = Counter()
    prefix_share = train_prefix_shares()

    def prefix_ok(item: tuple) -> bool:
        synth_total = sum(added.values()) + 1
        pb = item[0].meta["prefix_bucket"]
        return synth_total < 50 or (added_prefix[pb] + 1) / synth_total <= SYNTH_BUCKET_RATIO_MAX * prefix_share[pb]

    def within(extra: str) -> bool:
        total = train_total + sum(added.values()) + 1
        if any(abs((train[b] + added[b] + (b == extra)) / total - train_share[b]) > COHORT_SHIFT_MAX
               for b in FB_BUCKETS):
            return False
        synth_total = sum(added.values()) + 1
        # the ratio cap binds once SYNTH has a meaningful size (first 50 picks are exempt)
        return synth_total < 50 or (added[extra] + 1) / synth_total <= SYNTH_BUCKET_RATIO_MAX * train_share[extra]

    chosen = []
    progress = True
    while progress:
        progress = False
        for b in reversed(FB_BUCKETS):  # longest buckets first
            if not queues[b] or not within(b):
                continue
            group = next((g for g in queues[b] if any(prefix_ok(it) for it in g)), None)
            if group is None:
                continue
            # the alternative whose prompt-length bucket has the most headroom vs TRAIN
            pick = min((it for it in group if prefix_ok(it)),
                       key=lambda it: ((added_prefix[it[0].meta["prefix_bucket"]] + 1)
                                       / prefix_share[it[0].meta["prefix_bucket"]], it[0].meta.get("alt_rank", 0)))
            queues[b].remove(group)
            chosen.append(pick)
            added[b] += 1
            added_prefix[pick[0].meta["prefix_bucket"]] += 1
            progress = True
    for b in FB_BUCKETS:
        rejected[("finish_block", "cohort_length_rule")] += len(queues[b])
    # Apply the caps until nothing changes, so all hold on the final set: test-file and
    # full-body shares, and the SYNTH-alone bucket ratio (drops can shift bucket shares).
    changed = True
    while changed:
        changed = False
        for b in FB_BUCKETS:
            in_b = [it for it in chosen if it[0].meta["length_bucket"] == b]
            others = len(chosen) - len(in_b)
            share = SYNTH_BUCKET_RATIO_MAX * train_share[b]
            limit = int(share * others / (1 - share)) if share < 1 else len(in_b)
            if len(in_b) > limit:
                drop = {id(it) for it in in_b[limit:]}  # latest-added go first
                chosen = [it for it in chosen if id(it) not in drop]
                rejected[("finish_block", "synth_bucket_ratio_cap")] += len(drop)
                changed = True
        for pb in PREFIX_BUCKET_NAMES:
            in_pb = [it for it in chosen if it[0].meta["prefix_bucket"] == pb]
            others = len(chosen) - len(in_pb)
            share = SYNTH_BUCKET_RATIO_MAX * prefix_share[pb]
            limit = int(share * others / (1 - share)) if share < 1 else len(in_pb)
            if len(in_pb) > limit:
                drop = {id(it) for it in in_pb[limit:]}
                chosen = [it for it in chosen if id(it) not in drop]
                rejected[("finish_block", "synth_prefix_ratio_cap")] += len(drop)
                changed = True
        for flag, cap, reason in (("test_file", TEST_FILE_MAX_SHARE, "test_file_cap"),
                                  ("full_body", FULL_BODY_MAX_SHARE, "full_body_cap")):
            flagged = sorted((it for it in chosen if it[0].meta.get(flag)), key=lambda it: it[0].row_id())
            others = len(chosen) - len(flagged)
            limit = int(cap * others / (1 - cap))  # flagged / (others + flagged) <= cap after dropping
            if len(flagged) > limit:
                drop = {id(it) for it in flagged[limit:]}
                chosen = [it for it in chosen if id(it) not in drop]
                rejected[("finish_block", reason)] += len(drop)
                changed = True
    final = Counter(it[0].meta["length_bucket"] for it in chosen)
    final_prefix = Counter(it[0].meta["prefix_bucket"] for it in chosen)
    combined_total = train_total + sum(final.values())
    report.update(prefix_rule=f"SYNTH alone at most {SYNTH_BUCKET_RATIO_MAX:g}x TRAIN's share in every prompt "
                              "prefix-line bucket",
                  prefix_train_share={k: round(v, 4) for k, v in prefix_share.items()},
                  prefix_synth_share={k: round(final_prefix[k] / max(1, len(chosen)), 4) for k in PREFIX_BUCKET_NAMES},
                  length_definition="profile.target_length (leading blank and restated signature line removed), "
                                    "same for TRAIN and SYNTH")
    report.update(
        rule=(f"TRAIN+SYNTH finish_block length shares within {COHORT_SHIFT_MAX:.0%} of TRAIN in every bucket, and "
              f"SYNTH alone at most {SYNTH_BUCKET_RATIO_MAX:g}x TRAIN's share in every bucket"),
        train_counts={b: train[b] for b in FB_BUCKETS},
        available={b: sum(1 for g in pool if g[0][0].meta["length_bucket"] == b) for b in FB_BUCKETS},
        selected=dict(final),
        synth_share={b: round(final[b] / max(1, sum(final.values())), 4) for b in FB_BUCKETS},
        train_share={b: round(train_share[b], 4) for b in FB_BUCKETS},
        combined_share={b: round((train[b] + final[b]) / combined_total, 4) for b in FB_BUCKETS})
    return chosen, report


ROXYGEN_MATCH_TAGS = ("export", "keywords", "noRd", "rdname", "return", "description", "title", "family",
                      "inheritParams", "details", "importFrom")


def shape_roxygen(items: list, finish_total: int, rejected: Counter) -> list:
    """Cap roxygen at ROXYGEN_MAX_SHARE of the set and choose the blocks greedily so
    the selected tag frequencies approach TRAIN's (and no tag order dominates)."""
    limit = int(ROXYGEN_MAX_SHARE / (1 - ROXYGEN_MAX_SHARE) * finish_total)
    train_freq = profile.profile(profile.train_items({"roxygen_drafting"}))["roxygen_drafting"]["tag_frequency"]
    target = {t: train_freq.get(t, 0.0) for t in ROXYGEN_MATCH_TAGS}
    pool = sorted(items, key=lambda it: it[0].row_id())
    tags_of = {id(it): {m.group(1) for m in map(profile.TAG_RE.match, (x.strip() for x in it[0].target)) if m}
               for it in pool}
    chosen: list = []
    counts: Counter = Counter()
    orders: Counter = Counter()
    order_cap = max(1, int(ROXYGEN_TAG_ORDER_MAX_SHARE * min(limit, len(pool))))

    def distance(extra_tags: set[str], n: int) -> float:
        return sum(abs((counts[t] + (t in extra_tags)) / n - target[t]) for t in ROXYGEN_MATCH_TAGS)

    while pool and len(chosen) < limit:
        options = [it for it in pool if orders[it[0].meta.get("tag_order", "")] < order_cap]
        if not options:
            break
        best = min(options, key=lambda it: (distance(tags_of[id(it)], len(chosen) + 1), it[0].row_id()))
        pool.remove(best)
        chosen.append(best)
        counts.update(tags_of[id(best)])
        orders[best[0].meta.get("tag_order", "")] += 1
    rejected[("roxygen_drafting", "tag_frequency_selection_or_cap")] += len(pool)
    return chosen


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def sidecar_record(case: Case, context: dict, row: dict) -> dict:
    return {
        "context": context,
        "context_has_target_or_reward_keys": False,
        "family": case.family,
        "offline_static_source": True,
        "package_id": case.package_id,
        "prompt_sha256": hashlib.sha256(row["prompt_text"].encode("utf-8")).hexdigest(),
        "row_id": row["id"],
        "selection_geometry": {
            "availability": "synthetic_document",
            "policy_id": "synth-window",
            "document_sha256": context["replacement_range"]["content_sha256"],
            "spans": [{"kind": "prefix", "start_line": case.window_start, "end_line": case.region_start - 1},
                      {"kind": "region", "start_line": case.region_start, "end_line": case.region_end},
                      {"kind": "suffix", "start_line": case.region_end,
                       "end_line": (len(case.doc) if case.window_end is None else case.window_end) - 1}],
        },
        "source_identity": {
            "generator": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "group_id": "g-" + case.package_id.replace(":", "-"),
            "package_id": case.package_id,
            "seed": case.meta["seed"],
            "rule": case.meta["rule"],
            "registry_provenance_decision": "synthetic_original",
        },
        "split": "train",
    }


def case_record(case: Case, context: dict, row: dict, checks: dict) -> dict:
    return {
        "row_id": row["id"], "family": case.family, "package_id": case.package_id, "split": "train",
        "context": context, "target_operation": "replace", "region_new": list(case.target),
        "meta": case.meta, "verification": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, action="append", required=True,
                        help="seed root; repeat for several (batch names must be unique)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--families", default="finish_block,roxygen_drafting")
    parser.add_argument("--blank-policy", default="authored", choices=("authored", "v1"))
    parser.add_argument("--cpt-hits", type=Path, default=None,
                        help="cpt_check.py output bound to this build's sources; required for an admitted build")
    parser.add_argument("--normalise-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 workers (RULES operating rule 8)")
    families = set(args.families.split(","))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scorer, formatter, decoder = make_scorer(args.workers)

    def log(message: str) -> None:
        print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)

    log("normalise seeds")
    norm_ledger, normalised = normalise(args.seeds, out, formatter, args.blank_policy)
    if args.normalise_only:
        write_jsonl(out / "normalisation-ledger.jsonl", norm_ledger)
        log(f"{len(normalised)} sources written; stop (--normalise-only)")
        return 0
    cpt_lines, cpt_tokens, cpt_info, admitted = load_cpt(args.cpt_hits, out / "sources")
    log("behavioural checks")
    behaviour = behaviour_checks(args.seeds, normalised)
    log(f"{len(normalised)} sources; generate cases")
    cases = generate_cases(normalised, formatter, families)
    rejected = Counter((c.family, c.meta["rejected"]) for c in cases if c.meta.get("rejected"))
    candidates = [c for c in cases if not c.meta.get("rejected")]
    # Behavioural evidence per case. "entered" = the seed file's check passed and the unit's
    # function body executed during the traced run; it does not mean the check asserted the
    # behaviour of the completed code. Call units (setMethod, test_that, R6Class, ...) are
    # not functions bound by name and cannot be traced.
    for case in candidates:
        record = behaviour.get(case.meta["seed"], {"status": "no_check", "entered": []})
        unit = case.meta.get("unit", "")
        traceable = case.meta.get("unit_kind") == "assign_function"
        case.meta["behaviour"] = ("fail" if record["status"] == "fail" else
                                  "no_check" if record["status"] == "no_check" else
                                  "unit_not_traceable" if not traceable else
                                  "entered" if unit in record["entered"] else "file_checked_not_entered")
    failing = [c for c in candidates if c.meta["behaviour"] == "fail"]
    for case in failing:
        rejected[(case.family, "behaviour_check_failed")] += 1
    candidates = [c for c in candidates if c.meta["behaviour"] != "fail"]
    seen = set()
    unique = []
    for case in candidates:
        if case.row_id() in seen:
            rejected[(case.family, "exact_duplicate")] += 1
            continue
        seen.add(case.row_id())
        unique.append(case)
    log(f"{len(cases)} generated, {len(unique)} candidates; verify")

    from sepalith.data.synth.verify import verify
    tokenizer = load_tokenizer()
    results = verify(unique, tokenizer, scorer, formatter, decoder, workers=args.workers)
    verification_ledger = [{"row_id": c.row_id(), "family": c.family, "rule": c.meta["rule"], **checks}
                           for c, (_r, _x, checks) in zip(unique, results, strict=True)]
    failures: Counter = Counter()
    for c, (_r, _x, checks) in zip(unique, results, strict=True):
        if not checks["passed"]:
            for key in ("render", "length", "parse_after", "parse_before", "history_parses", "air_idempotent",
                        "reward_v21"):
                if not checks.get(key):
                    failures[(c.family, key)] += 1
            for problem in checks.get("family_problems", []):
                failures[(c.family, problem)] += 1
    verified = [(c, r) for c, r in zip(unique, results, strict=True) if r[2]["passed"]]
    log(f"verified {len(verified)}/{len(unique)}; decontaminate")

    train4, train_exact, train_masked = train_indexes()
    dev4, dev_exact, dev_masked, dev_info = dev_indexes()
    decon: Counter = Counter()
    masked_report: dict[str, Counter] = {"train": Counter(), "dev": Counter()}
    clean = []
    for case, result in verified:
        prompt = list(case.doc[case.window_start:case.region_start]) + list(case.region_old)
        window = "\n".join(case.window_after())
        grams = dedup.line_4grams(window)
        span = dedup.span_hashes(prompt, case.target)
        span_masked = dedup.span_hashes(prompt, case.target, masked=True)
        dev_c, dev_run = dedup.overlap(dev_exact.hits(span))
        train_c, train_run = dedup.overlap(train_exact.hits(span))
        cpt_c, cpt_run = dedup.overlap(cpt_tokens.hits(span))
        masked = {"train": dedup.overlap(train_masked.hits(span_masked))[0],
                  "dev": dedup.overlap(dev_masked.hits(span_masked))[0]}
        reasons = []
        if dev4.gram_hits(grams):
            reasons.append("dev_line_4gram")
        if dev_c >= DEV_CONTAINMENT_MAX or dev_run >= DEV_RUN_MAX:
            reasons.append("dev_token_overlap")
        if masked["dev"] >= DEV_MASKED_LONG_MAX and case_length(case) >= 9:
            reasons.append("dev_masked_structure_long_target")
        if train4.gram_hits(dedup.line_4grams("\n".join(case.target))):
            reasons.append("train_target_line_4gram")
        if train_c >= TRAIN_CONTAINMENT_MAX or train_run >= TRAIN_RUN_MAX:
            reasons.append("train_token_overlap")
        if grams & cpt_lines:
            reasons.append("cpt_line_4gram")
        if cpt_c >= CPT_CONTAINMENT_MAX or cpt_run >= CPT_RUN_MAX:
            reasons.append("cpt_token_overlap")
        case.meta["decontamination"] = {
            "reasons": reasons, "dev_containment": round(dev_c, 3), "dev_run": dev_run,
            "train_containment": round(train_c, 3), "train_run": train_run,
            "cpt_containment": round(cpt_c, 3), "cpt_run": cpt_run,
            "masked_train_containment": round(masked["train"], 3), "masked_dev_containment": round(masked["dev"], 3)}
        for ref in ("train", "dev"):
            value = masked[ref]
            masked_report[ref]["cases"] += 1
            masked_report[ref][">=0.5"] += value >= 0.5
            masked_report[ref][">=0.8"] += value >= 0.8
            masked_report[ref]["==1.0"] += value >= 1.0
        for reason in reasons:
            decon[(case.family, reason)] += 1
        if not reasons:
            clean.append((case, result))
    log(f"{len(clean)} clean; near-duplicate filter")
    # Packaging alternatives of one finish_block cut share their target: deduplicate one
    # representative per cut and keep or drop the cut's alternatives together.
    groups: dict[str, list] = defaultdict(list)
    for item in sorted(clean, key=lambda it: it[0].row_id()):
        groups[item[0].meta.get("cut_key") or item[0].row_id()].append(item)
    group_keys = sorted(groups, key=lambda k: min(it[0].row_id() for it in groups[k]))
    reps = [min(groups[k], key=lambda it: (it[0].meta.get("alt_rank", 0), it[0].row_id())) for k in group_keys]
    dup_of = dedup.near_duplicates([dedup_key(c) for c, _ in reps], NEAR_DUP_THRESHOLD)
    kept = []
    for key, dup in zip(group_keys, dup_of, strict=True):
        if dup is None:
            kept += groups[key]
        else:
            rejected[(groups[key][0][0].family, "near_duplicate")] += 1
    finish, shaping = shape_finish([it for it in kept if it[0].family == "finish_block"], rejected)
    final = finish + shape_roxygen([it for it in kept if it[0].family == "roxygen_drafting"], len(finish), rejected)
    final.sort(key=lambda item: (item[0].family, item[0].row_id()))
    log(f"final {len(final)}; write outputs")

    rows = [result[0] for _, result in final]
    sidecar = [sidecar_record(case, result[1], result[0]) for case, result in final]
    records = [case_record(case, result[1], result[0], result[2]) for case, result in final]
    paths = {
        "train-token-rows.jsonl": rows,
        "context-sidecar.jsonl": sidecar,
        "cases.jsonl": records,
        "verification-ledger.jsonl": verification_ledger,
        "normalisation-ledger.jsonl": norm_ledger,
        "behaviour-ledger.jsonl": [{"seed": k, **v} for k, v in sorted(behaviour.items())],
    }
    for name, payload in paths.items():
        write_jsonl(out / name, payload)

    stats: dict = {"families": Counter(c.family for c, _ in final)}
    for key in ("rule", "variant", "determinacy", "behaviour", "length_bucket", "batch"):
        table: dict[str, Counter] = defaultdict(Counter)
        for case, _ in final:
            if key in case.meta:
                table[case.family][str(case.meta[key])] += 1
        stats[key] = {f: dict(sorted(v.items())) for f, v in table.items()}
    stats["full_body"] = sum(bool(c.meta.get("full_body")) for c, _ in final if c.family == "finish_block")
    stats["signature_region"] = sum(bool(c.meta.get("signature_region")) for c, _ in final)
    stats["test_file"] = sum(bool(c.meta.get("test_file")) for c, _ in final if c.family == "finish_block")
    tokens = defaultdict(list)
    for case, result in final:
        tokens[case.family].append((result[0]["prompt_token_count"], result[0]["target_token_count"]))
    stats["tokens"] = {f: {"prompt_mean": round(sum(p for p, _ in v) / len(v), 1), "prompt_max": max(p for p, _ in v),
                           "target_mean": round(sum(t for _, t in v) / len(v), 1), "target_max": max(t for _, t in v)}
                       for f, v in tokens.items()}
    units = {c.meta["unit_key"] for c, _ in final}
    stats["diversity"] = {"cases": len(final), "distinct_units": len(units),
                          "cases_per_unit": round(len(final) / max(1, len(units)), 3),
                          "seed_files": len({c.meta["seed"] for c, _ in final}),
                          "packages": len({c.package_id for c, _ in final})}
    behaviour_summary = Counter(v["status"] for v in behaviour.values())
    summary = {
        "generated": len(cases),
        "families_built": sorted(families),
        "blank_policy": args.blank_policy,
        "generator_rejections": {f"{f}:{r}": n for (f, r), n in sorted(rejected.items())},
        "protocol_verification": {
            "meaning": "mechanical checks only: render, length, R parse, air, reward v2.1 exact, family checks",
            "candidates": len(unique), "passed": len(verified),
            "mechanical_pass_rate": round(len(verified) / max(1, len(unique)), 4),
            "failures": {f"{f}:{k}": n for (f, k), n in sorted(failures.items())},
        },
        "determinacy": stats["determinacy"],
        "behaviour": {"seed_files": dict(behaviour_summary), "final_cases": stats["behaviour"],
                      "meaning": ("entered = the seed file's base-R check passed and the unit's function body "
                                  "executed during the traced run (trace() on every closure; indirect calls and "
                                  "S3 dispatch count). Entered is not 'target behaviour asserted'.")},
        "decontamination": {
            "removed": {f"{f}:{r}": n for (f, r), n in sorted(decon.items())},
            "masked_structural_containment": {k: dict(v) for k, v in masked_report.items()},
            "dev_reference": {"files": dev_info["files"], "documents_hashed": dev_info["documents_hashed"]},
            "train_reference": {"rows": TRAIN_ROWS_SHA256},
            "cpt_reference": cpt_info,
            "fingerprint": dict(dedup.FINGERPRINT),
            "thresholds": {"dev": [DEV_CONTAINMENT_MAX, DEV_RUN_MAX], "train": [TRAIN_CONTAINMENT_MAX, TRAIN_RUN_MAX],
                           "cpt": [CPT_CONTAINMENT_MAX, CPT_RUN_MAX], "near_duplicate_jaccard": NEAR_DUP_THRESHOLD,
                           "line_4gram": "any shared 4-gram of non-trivial lines (DEV window, TRAIN target, CPT window)"},
            "split_file_read": False,
        },
        "shaping": shaping,
        "final": len(final),
        "admitted": admitted,
        "stats": stats,
    }
    (out / "build-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=dict) + "\n")
    (out / "dev-inputs.json").write_text(json.dumps(dev_info["sha256"], indent=2, sort_keys=True) + "\n")
    names = [*paths, "build-summary.json", "dev-inputs.json"]
    if args.cpt_hits and args.cpt_hits.resolve().parent == out.resolve():
        names.append(args.cpt_hits.name)
    artifacts = {name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size} for name in names}
    manifest = {
        "schema": "sepalith.synth.manifest.v2",
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION,
                      "code": "packages/sepalith/src/sepalith/data/synth/", "git_head": git_head()},
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "renderer_id": "zeta2-prm03-v1",
        "tokenizer_json_sha256": rows[0]["tokenizer_json_sha256"] if rows else None,
        "rows": len(rows),
        "families": dict(stats["families"]),
        "mechanical_pass_rate": summary["protocol_verification"]["mechanical_pass_rate"],
        "admitted": admitted,
        "artifacts": artifacts,
        "sources": {"files": len(normalised), "sha256_of_tree": cpt_check.tree_sha256(out / "sources")},
        "seed_tree": [str(root) for root in args.seeds],
        "location": str(out),
        "split": "train",
        "hf_uploaded": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"generated": len(cases), "final": len(final), "families": dict(stats["families"]),
                      "mechanical_pass_rate": manifest["mechanical_pass_rate"], "admitted": admitted}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
