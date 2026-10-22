"""Build SYNTH-v1: seeds -> normalised sources -> cases -> verified, deduplicated,
decontaminated drop-in TRAIN rows plus sidecar, ledgers and manifest.

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.data.synth.build \
        --seeds docs/training/plan-20260927/synth/seeds \
        --out /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1

The DEV check hashes the DEV panels and DEV-split source files listed in
``DEV_SOURCES`` and reports aggregate counts only. No path containing
``final`` is ever opened (``guard_path``).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
import datetime
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from sepalith.data.synth import GENERATOR_ID, GENERATOR_VERSION
from sepalith.data.synth import dedup, finish_block, format_propagation, normalize, roxygen
from sepalith.campaign_protocol import TokenEncoder
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
SPLIT = DATA / "DAT-02-global-split-v2.json"
FAMILY_SHARE_CAPS = {"roxygen_drafting": 0.15, "format_propagation": 0.20}
NEAR_DUP_THRESHOLD = 0.8
TRAIN_CONTAINMENT_MAX = 0.6
DEV_CONTAINMENT_MAX = 0.5
MIN_SHINGLES_FOR_CONTAINMENT = 15


def guard_path(path: str) -> str:
    if "final" in path.lower() or "sealed" in path.lower():
        raise SystemExit(f"refusing to open a sealed/final path: {path}")
    return path


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


# ---------------------------------------------------------------------------
# Stage 1: seeds -> normalised sources
# ---------------------------------------------------------------------------
def seed_files(seeds: Path) -> list[Path]:
    return sorted(p for p in seeds.rglob("*.R") if p.is_file())


def normalise(seeds: Path, out: Path, formatter: AirFormatter) -> tuple[list[dict], list[tuple[Path, str]]]:
    files = seed_files(seeds)
    structures = describe(files)
    ledger = []
    normalised = []
    for path in files:
        relative = str(path.relative_to(seeds))
        text, info = normalize.normalise_file(path, relative, structures[str(path)], formatter)
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


# ---------------------------------------------------------------------------
# Stage 2: cases
# ---------------------------------------------------------------------------
def generate_cases(normalised: list[tuple[Path, str]], formatter: AirFormatter) -> list[Case]:
    structures = describe([p for p, _ in normalised])
    cases: list[Case] = []
    for path, relative in normalised:
        structure = structures[str(path)]
        if not structure.ok:
            continue
        lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
        package_id, rel_path = package_and_path(relative)
        produced = (finish_block.generate(lines, structure, package_id, rel_path)
                    + roxygen.generate(lines, structure, package_id, rel_path)
                    + format_propagation.generate(lines, structure, package_id, rel_path, formatter))
        for case in produced:
            case.meta["seed"] = relative
            case.meta["batch"] = relative.split("/")[0]
            case.meta["style"] = normalize.style_for(relative)
        cases += produced
    return cases


# ---------------------------------------------------------------------------
# Stage 4: dedup and decontamination
# ---------------------------------------------------------------------------
def train_index() -> dedup.ReferenceIndex:
    if sha256_file(TRAIN_ROWS) != TRAIN_ROWS_SHA256:
        raise SystemExit("TRAIN rows hash mismatch")
    index = dedup.ReferenceIndex("train")
    with TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index.add(row["prompt_text"] + "\n" + row["target_body_text"])
    index.freeze()
    return index


def dev_index() -> tuple[dedup.ReferenceIndex, dict]:
    index = dedup.ReferenceIndex("dev")
    files: list[str] = []
    for pattern in DEV_SOURCES:
        files += sorted(glob.glob(guard_path(pattern)))
    inputs = {}
    for path in files:
        guard_path(path)
        inputs[path] = sha256_file(Path(path))
        for text in dedup.texts_of_file(path):
            index.add(text)
    index.freeze()
    return index, {"files": len(files), "documents_hashed": index.documents, "sha256": inputs}


def split_package_names() -> dict[str, set[str]]:
    """Normalised package names per DAT-02 split (identity metadata only)."""
    data = json.loads(SPLIT.read_text())
    out: dict[str, set[str]] = defaultdict(set)
    for group in data["groups"]:
        for form in group.get("identity_forms", []):
            if form.startswith("pkg:"):
                out[group["split"]].add(form.split(":", 1)[1])
    return out


def dedup_key(case: Case) -> set[int]:
    lead = case.doc[max(case.window_start, case.region_start - 3):case.region_start]
    return dedup.shingles("\n".join([*lead, *case.region_old, "=>", *case.target]))


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
            "policy_id": "synth-v1-window",
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


def git_head() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() or None


def length_bucket(n: int) -> str:
    for bound in (1, 2, 4, 8, 12, 20, 30, 45):
        if n <= bound:
            return f"<={bound}"
    return ">45"


def train_target_lengths() -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    with TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["target_operation"] == "replace":
                out[row["family"]][length_bucket(row["target_body_text"].count("\n") + 1)] += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpt-hits", type=Path, default=None,
                        help="JSON from cpt_check.py: line 4-gram hashes found in the CPT corpus")
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 workers (RULES operating rule 8)")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scorer, formatter, decoder = make_scorer(args.workers)
    log = lambda message: print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)  # noqa: E731

    log("normalise seeds")
    norm_ledger, normalised = normalise(args.seeds, out, formatter)
    if args.limit_files:
        normalised = normalised[:args.limit_files]
    log(f"{len(normalised)} sources; generate cases")
    cases = generate_cases(normalised, formatter)
    rejected = Counter((c.family, c.meta["rejected"]) for c in cases if c.meta.get("rejected"))
    candidates = [c for c in cases if not c.meta.get("rejected")]
    # exact duplicates (same id = same content)
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
    failures = Counter()
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

    # decontamination
    train = train_index()
    dev, dev_info = dev_index()
    cpt_grams: set[int] = set()
    cpt_info = None
    if args.cpt_hits:
        cpt_data = json.loads(args.cpt_hits.read_text())
        cpt_grams = set(cpt_data["hit_grams"])
        cpt_info = {k: v for k, v in cpt_data.items() if k != "hit_grams"}
    names = split_package_names()
    decon = Counter()
    clean = []
    for case, result in verified:
        window = "\n".join(case.window_after())
        grams = dedup.line_4grams(window)
        target_shingles = dedup.shingles("\n".join(case.target))
        reasons = []
        if dev.gram_hits(grams):
            reasons.append("dev_line_4gram")
        if len(target_shingles) >= MIN_SHINGLES_FOR_CONTAINMENT and dev.containment(target_shingles) >= DEV_CONTAINMENT_MAX:
            reasons.append("dev_target_containment")
        if train.gram_hits(dedup.line_4grams("\n".join(case.target))):
            reasons.append("train_target_line_4gram")
        if (len(target_shingles) >= MIN_SHINGLES_FOR_CONTAINMENT
                and train.containment(target_shingles) >= TRAIN_CONTAINMENT_MAX):
            reasons.append("train_target_containment")
        if grams & cpt_grams:
            reasons.append("cpt_line_4gram")
        package = case.package_id.split(":", 1)[1].lower()
        if any(package in names[s] for s in names if s != "train_group"):
            reasons.append("package_name_in_dev_or_final_split")
        case.meta["decontamination"] = reasons
        for reason in reasons:
            decon[(case.family, reason)] += 1
        if not reasons:
            clean.append((case, result))
    log(f"{len(clean)} clean; near-duplicate filter")
    clean.sort(key=lambda item: item[0].row_id())
    keys = [dedup_key(c) for c, _ in clean]
    dup_of = dedup.near_duplicates(keys, NEAR_DUP_THRESHOLD)
    kept = []
    for (case, result), dup in zip(clean, dup_of, strict=True):
        if dup is None:
            kept.append((case, result))
        else:
            rejected[(case.family, "near_duplicate")] += 1
    # family share caps
    by_family: dict[str, list] = defaultdict(list)
    for item in kept:
        by_family[item[0].family].append(item)
    finish = len(by_family["finish_block"])
    capped = list(by_family["finish_block"])
    for family, items in by_family.items():
        if family == "finish_block":
            continue
        share = FAMILY_SHARE_CAPS.get(family, 1.0)
        # family <= share of the final total, with finish_block as the anchor
        limit = int(share / (1 - sum(FAMILY_SHARE_CAPS.values())) * finish)
        if len(items) > limit:
            rejected[(family, "family_share_cap")] += len(items) - limit
            items = sorted(items, key=lambda item: item[0].row_id())[:limit]
        capped += items
    capped.sort(key=lambda item: (item[0].family, item[0].row_id()))
    log(f"final {len(capped)}; write outputs")

    rows = [result[0] for _, result in capped]
    sidecar = [sidecar_record(case, result[1], result[0]) for case, result in capped]
    records = [case_record(case, result[1], result[0], result[2]) for case, result in capped]
    paths = {
        "train-token-rows.jsonl": rows,
        "context-sidecar.jsonl": sidecar,
        "cases.jsonl": records,
        "verification-ledger.jsonl": verification_ledger,
        "normalisation-ledger.jsonl": norm_ledger,
    }
    for name, payload in paths.items():
        write_jsonl(out / name, payload)

    # statistics
    stats: dict = {"families": Counter(c.family for c, _ in capped)}
    lengths: dict[str, Counter] = defaultdict(Counter)
    for case, _ in capped:
        lengths[case.family][length_bucket(len(case.target))] += 1
    stats["target_lines"] = {f: dict(sorted(v.items())) for f, v in lengths.items()}
    stats["train_target_lines"] = {f: dict(sorted(v.items())) for f, v in train_target_lengths().items()}
    for key in ("rule", "variant", "bucket", "batch", "style"):
        table: dict[str, Counter] = defaultdict(Counter)
        for case, _ in capped:
            if key in case.meta:
                table[case.family][str(case.meta[key])] += 1
        stats[key] = {f: dict(sorted(v.items())) for f, v in table.items()}
    tokens = defaultdict(list)
    for case, result in capped:
        tokens[case.family].append((result[0]["prompt_token_count"], result[0]["target_token_count"]))
    stats["tokens"] = {f: {"prompt_mean": round(sum(p for p, _ in v) / len(v), 1),
                           "prompt_max": max(p for p, _ in v),
                           "target_mean": round(sum(t for _, t in v) / len(v), 1),
                           "target_max": max(t for _, t in v)} for f, v in tokens.items()}
    stats["packages"] = len({c.package_id for c, _ in capped})
    stats["seed_files"] = len({c.meta["seed"] for c, _ in capped})
    summary = {
        "generated": len(cases),
        "generator_rejections": {f"{f}:{r}": n for (f, r), n in sorted(rejected.items())},
        "verification": {
            "candidates": len(unique),
            "passed": len(verified),
            "pass_rate": round(len(verified) / max(1, len(unique)), 4),
            "failures": {f"{f}:{k}": n for (f, k), n in sorted(failures.items())},
        },
        "decontamination": {
            "removed": {f"{f}:{r}": n for (f, r), n in sorted(decon.items())},
            "dev_reference": {"files": dev_info["files"], "documents_hashed": dev_info["documents_hashed"]},
            "train_reference": {"rows": train.documents, "sha256": TRAIN_ROWS_SHA256},
            "cpt_reference": cpt_info,
            "thresholds": {"line_4gram": "any shared 4-gram of non-trivial lines",
                           "dev_target_containment": DEV_CONTAINMENT_MAX,
                           "train_target_containment": TRAIN_CONTAINMENT_MAX,
                           "min_shingles_for_containment": MIN_SHINGLES_FOR_CONTAINMENT,
                           "near_duplicate_jaccard": NEAR_DUP_THRESHOLD},
        },
        "final": len(capped),
        "stats": stats,
    }
    (out / "build-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=dict) + "\n")
    (out / "dev-inputs.json").write_text(json.dumps(dev_info["sha256"], indent=2, sort_keys=True) + "\n")
    names = [*paths, "build-summary.json", "dev-inputs.json"]
    if args.cpt_hits and args.cpt_hits.resolve().parent == out.resolve():
        names.append(args.cpt_hits.name)
    artifacts = {name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size} for name in names}
    sources = sorted((out / "sources").rglob("*.R"))
    source_digest = hashlib.sha256()
    for path in sources:
        source_digest.update(str(path.relative_to(out)).encode() + b"\0" + path.read_bytes())
    manifest = {
        "schema": "sepalith.synth.manifest.v1",
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION,
                      "code": "packages/sepalith/src/sepalith/data/synth/",
                      "git_head": git_head()},
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "renderer_id": "zeta2-prm03-v1",
        "tokenizer_json_sha256": rows[0]["tokenizer_json_sha256"] if rows else None,
        "rows": len(rows),
        "families": dict(stats["families"]),
        "verification_pass_rate": summary["verification"]["pass_rate"],
        "artifacts": artifacts,
        "sources": {"files": len(sources), "sha256_of_tree": source_digest.hexdigest()},
        "seed_tree": str(args.seeds),
        "location": str(out),
        "split": "train",
        "hf_uploaded": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("generated", "final")} | {"families": dict(stats["families"]),
                      "pass_rate": summary["verification"]["pass_rate"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
