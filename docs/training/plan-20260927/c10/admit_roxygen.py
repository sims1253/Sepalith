#!/usr/bin/env python3
"""C10: admission review of the 3,903 recommended roxygen rows.

Subcommands:

``check``
    Rebuilds the recommended-ID ledger of the DAT-10 context admission review
    (its JSON is byte-identical to the lost original, so the sha256 pinned in
    the review receipt is verified), then checks every recommended row:
    token geometry under the trainer's row validator, licence against the
    per-package provenance ledger (standing rule in SYSTEMS.md section 7),
    identity-group split against ``DAT-02-global-split-v2.json`` (TRAIN groups
    only), and duplicates against the 15,006 TRAIN rows and within the set.
    It then draws the seed-3407 semantic sample (60 rows, stratified by
    sequence length), decodes it with the parent tokenizer, runs the R parse
    and roxygen tag checks (``check_roxygen.R``) and the ``air format`` checks,
    and writes ``mechanical.jsonl``, ``checks.json`` and a human review dump
    ``review.md`` into ``--work-dir``. The work dir holds row text and stays
    outside the repository.

``finalize``
    Joins the checks with the hand classifications in
    ``semantic-classifications.json``, applies the C06-style acceptance bands,
    and writes the admitted and held ID ledgers, the arm-B cohort row file
    (15,006 TRAIN rows followed by the admitted rows, sorted by id) and the
    receipt.

Inputs are read-only. Only TRAIN-split rows are read: the roxygen queue, the
15,006 TRAIN rows and the split metadata. No DEV payload, final-set file or
upstream source pool is opened.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "packages/sepalith/src"))
from sepalith.training.sft.campaign_edit_data import validate_row  # noqa: E402

DW = Path("/mnt/e/sepalith/campaign-20260915/data-work")
QUEUE = DW / "DAT10-novel-v1/roxygen-supported-context-v1"
PROFILES = QUEUE / "selected-context-profiles.jsonl"
TOKEN_ROWS = QUEUE / "selected-context-token-rows.jsonl"
FULL_FILE_ROWS = DW / "DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-full-file-token-rows.jsonl"
UNION_LEDGER = DW / "DAT10-roxy8597-materialization-v1/dedup-ledger.jsonl"
AUTH_DEDUP = DW / "DAT10-roxy8597-root-fixes-v1/authoritative-dedup-ledger.jsonl"
TRAIN_ROWS = DW / "DAT10-finish-source-repair-v3/train-token-rows.jsonl"
SPLIT = DW / "DAT-02-global-split-v2.json"
LICENCES = Path("/mnt/nas/home/sepalith/provenance")
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586")
PINS = {
    "profiles": (PROFILES, "e263a86658726bbd92129d1e322c6cac5c3ffe7ec98accf820ea25d8f2be05d2"),
    "token_rows": (TOKEN_ROWS, "f31377348fcefe948bf5fd3d31e8289a9f79528289ebf9281fa09ec8d6f753c1"),
    "full_file_token_rows": (FULL_FILE_ROWS, "752131a0ade6c1bd30248b9663046e027cd8ac9bdb38170a2b680f9dbae6d897"),
    "union_ledger": (UNION_LEDGER, "1429f776b83b2b45005fcc3592aeebe756058a7ae911fb6cc929012c2d52764f"),
    "authoritative_dedup_ledger": (AUTH_DEDUP, "ee60274a93dd7aa13d26b7b103a8a1408516a9d00dc68bc49eeb94a2c2bec337"),
    "train_rows": (TRAIN_ROWS, "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"),
    "split": (SPLIT, "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"),
    "tokenizer": (PARENT / "tokenizer.json", "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"),
}
# sha256 of full/recommended-context-ids.json in receipts/DAT-10-roxy-context-admission-review.json.
RECOMMENDED_IDS_SHA256 = "f88607c31ad60bb4b13eb50946ab2db6bfd7daf0f300bd9d9809a2c06b631e62"
RECOMMENDED = 3903
CANDIDATES = 10017
MAX_SEQUENCE = 16384  # the SFT v2 recipe's max_sequence_tokens (C04); 16K full-weight smoke peaked at 24.2 GB
SEED = 3407
N_SAMPLE = 60
STRATA = (("le512", 0, 512), ("513-1024", 513, 1024), ("1025-4096", 1025, 4096), ("gt4096", 4097, 10**9))
STRATUM_FLOOR = 5
CLASSES = ("correct", "cosmetic", "defective")

# Standing licensing rule (SYSTEMS.md section 7): "permissive only (MIT/Apache/
# BSD/GPL-family/CC-BY/CC0/CC-BY-SA-marked); NC/ND/unspecified excluded AND
# LEDGERED." A CRAN License field is a list of alternatives separated by "|";
# a row passes when any alternative is in an allowed family. "+ file LICENSE"
# (the CRAN copyright-holder file) does not change the family.
ALLOWED = (
    ("GPL-family", re.compile(r"^(A|L)?GPL\b|^GNU (Affero |Lesser |Library )?General Public License", re.I)),
    ("MIT", re.compile(r"^MIT\b", re.I)),
    ("BSD", re.compile(r"^(BSD|FreeBSD\b)", re.I)),  # "FreeBSD" is the BSD 2-clause licence
    ("Apache", re.compile(r"^Apache", re.I)),
    ("CC0", re.compile(r"^CC0\b", re.I)),
    ("CC-BY", re.compile(r"^(CC[ -]BY|Creative Commons Attribution)(?![- ]?N[CD])(?!.*(NonCommercial|NoDerivatives))", re.I)),
)
FORBIDDEN = re.compile(r"\bN[CD]\b|NonCommercial|NoDerivatives", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def verify_pins() -> dict:
    out = {}
    for name, (path, expected) in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"pin mismatch for {name}: {path} {actual}")
        out[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return out


# --------------------------------------------------------------------------- licence


def licence_family(licence: str | None) -> tuple[bool, str]:
    if licence is None:
        return False, "no_ledger_entry"
    if FORBIDDEN.search(licence):
        return False, "nc_or_nd_terms"
    for alternative in licence.split("|"):
        alt = re.sub(r"\+\s*file\s+LICEN[CS]E", "", alternative).strip(" +,")
        for name, pattern in ALLOWED:
            if pattern.search(alt):
                return True, name
    return False, "family_not_in_standing_rule"


def package_licence(package: str, version: str) -> dict:
    path = LICENCES / f"{package}.json"
    if not path.is_file():
        return {"ledger": None, "allowed": False, "reason": "no_ledger_entry"}
    record = json.loads(path.read_text())
    allowed, family = licence_family(record.get("license"))
    result = {"ledger": str(path), "ledger_sha256": sha256_file(path), "license": record.get("license"),
              "ledger_version": record.get("version"), "allowed": allowed,
              "family" if allowed else "reason": family}
    if record.get("package") != package or record.get("version") != version:
        result.update(allowed=False, reason="ledger_package_or_version_differs")
    return result


# --------------------------------------------------------------------------- split


def norm_package(name: str) -> str:
    return re.sub(r"[^0-9a-z._]", "", unicodedata.normalize("NFKC", name).lower())


def load_split() -> tuple[dict[str, str], dict[str, set[str]]]:
    split = json.loads(SPLIT.read_text())
    by_group = {g["group_id"]: g["split"] for g in split["groups"]}
    by_package: dict[str, set[str]] = collections.defaultdict(set)
    for g in split["groups"]:
        for form in g["identity_forms"]:
            if form.startswith("pkg:"):
                by_package[form[4:]].add(g["split"])
    return by_group, by_package


# --------------------------------------------------------------------------- prompt parts


def prompt_parts(prompt: str) -> dict[str, object]:
    lines = prompt.split("\n")
    first = next(i for i, line in enumerate(lines) if line.startswith("<filename>"))
    refs = lines.index("<filename>selected_references")
    suffix_at = lines.index("<[fim-suffix]>")
    current = lines.index("<<<<<<< CURRENT")
    sep = lines.index("=======", current)
    return {"path": lines[first][len("<filename>"):], "prefix": lines[first + 1:refs],
            "suffix": lines[suffix_at + 1:current], "region_old": lines[current + 1:sep]}


def squash(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- check


def load_recommended() -> tuple[list[str], list[str], dict[str, dict]]:
    profiles: dict[str, dict] = {}
    with PROFILES.open() as stream:
        for line in stream:
            row = json.loads(line)
            profiles[row["row_id"]] = row
    accepted = []
    with UNION_LEDGER.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["union_source"] == "original_context_recommendation":
                accepted.append(row["row_id"])
    accepted.sort()
    held = sorted(set(profiles) - set(accepted))
    if len(profiles) != CANDIDATES or len(accepted) != RECOMMENDED or not set(accepted) <= set(profiles):
        raise SystemExit("recommended scope differs")
    return accepted, held, profiles


def rebuild_recommended_ledger(accepted: list[str], held: list[str], out: Path) -> str:
    # Same writer and fields as review_context_admission.py (r2-roxy-context-admission-review-v1).
    write_json(out, {
        "schema": "sepalith.dat10.roxy_context_recommendation_ids.v1",
        "scope": "full_10017",
        "status": "recommendation_only_not_training_admission",
        "accepted_recommendation_count": len(accepted),
        "held_count": len(held),
        "accepted_recommendation_ids": accepted,
        "held_ids": held,
    })
    digest = sha256_file(out)
    if digest != RECOMMENDED_IDS_SHA256:
        raise SystemExit(f"rebuilt recommended-context-ids.json differs from the review receipt: {digest}")
    return digest


def load_tokenizer() -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True)


def encode(tokenizer: "PreTrainedTokenizerBase", text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True))


def stratum(tokens: int) -> str:
    return next(name for name, low, high in STRATA if low <= tokens <= high)


def allocate(sizes: dict[str, int]) -> dict[str, int]:
    alloc = {k: min(STRATUM_FLOOR, v) for k, v in sizes.items()}
    rest = N_SAMPLE - sum(alloc.values())
    pool = {k: v - alloc[k] for k, v in sizes.items()}
    total = sum(pool.values())
    shares = {k: rest * v / total for k, v in pool.items()}
    extra = {k: int(s) for k, s in shares.items()}
    for k in sorted(shares, key=lambda k: (-(shares[k] - extra[k]), k))[: rest - sum(extra.values())]:
        extra[k] += 1
    return {k: alloc[k] + extra[k] for k in sizes}


def run_r_checks(work: Path, sample: list[dict]) -> dict[str, dict]:
    files = work / "sample"
    if files.exists():
        shutil.rmtree(files)
    files.mkdir(parents=True)
    manifest = ["key\tpath"]
    for item in sample:
        path = files / f"{item['row_id']}.R"
        path.write_text(item["document"], encoding="utf-8")
        manifest.append(f"{item['row_id']}\t{path}")
    (work / "r-manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    subprocess.run(["nice", "-n", "10", "Rscript", "--vanilla", str(HERE / "check_roxygen.R"),
                    str(work / "r-manifest.tsv"), str(work / "r-results.tsv")], check=True)
    results = {}
    lines = (work / "r-results.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        values = dict(zip(header, line.split("\t"), strict=True))
        results[values["key"]] = values
    return results


def run_air(work: Path, item: dict) -> dict:
    target = work / "air" / f"{item['row_id']}.R"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(item["document"], encoding="utf-8")
    first = subprocess.run(["nice", "-n", "10", "air", "format", str(target)], capture_output=True, text=True)
    once = target.read_text(encoding="utf-8")
    second = subprocess.run(["nice", "-n", "10", "air", "format", str(target)], capture_output=True, text=True)
    twice = target.read_text(encoding="utf-8")
    body = item["target_body"]
    return {"air_ok": first.returncode == 0 and second.returncode == 0, "air_idempotent": once == twice,
            "air_error": (first.stderr or second.stderr)[:200],
            "air_keeps_target_lines": all(line in once.split("\n") for line in body.split("\n") if line.strip())}


def roxygen_tags(body: str) -> dict:
    params: list[str] = []
    tags = collections.Counter()
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("#'"):
            continue
        match = re.match(r"#'\s*@(\w+)\s*(.*)", stripped)
        if match:
            tags[match.group(1)] += 1
            if match.group(1) == "param":
                names = match.group(2).split()[0] if match.group(2).split() else ""
                params.extend(n.strip() for n in names.split(",") if n.strip())
    return {"params": params, "tags": dict(sorted(tags.items())),
            "non_roxygen_lines": sum(1 for line in body.split("\n") if line.strip() and not line.strip().startswith("#'"))}


SPLIT_BLOCK_LINE = re.compile(r"^\s*(#+'|')")
PLAIN_COMMENT = re.compile(r"^\s*#(?!#*')")


def census_split_blocks(tokenizer: "PreTrainedTokenizerBase", wanted: set[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    seen = 0
    with FULL_FILE_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            token_row = row.get("token_row", row)
            rid = row.get("row_id") or token_row.get("id")
            if rid not in wanted:
                continue
            seen += 1
            prompt = tokenizer.decode(token_row["input_ids"][1:token_row["target_start"]], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
            prefix = prompt_parts(prompt)["prefix"]
            above = list(prefix)
            if above and not above[-1].strip():
                above.pop()
            nearest = above[-2:]
            if not nearest:
                continue
            first, second = nearest[-1], nearest[0] if len(nearest) == 2 else ""
            if SPLIT_BLOCK_LINE.match(first) or (PLAIN_COMMENT.match(first) and SPLIT_BLOCK_LINE.match(second)):
                hits[rid] = nearest
    if seen != len(wanted):
        raise SystemExit("full-file rows do not cover the census scope")
    return hits


def check(args: argparse.Namespace) -> None:
    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    pins = verify_pins()
    accepted, held, profiles = load_recommended()
    rebuilt = rebuild_recommended_ledger(accepted, held, work / "recommended-context-ids.json")
    wanted = set(accepted)
    auth = {}
    with AUTH_DEDUP.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in wanted:
                auth[row["row_id"]] = row
    tokenizer = load_tokenizer()

    # TRAIN rows: payload, prompt, target and documented-function keys.
    train_payload, train_prompt, train_target, train_fn = {}, {}, collections.defaultdict(list), {}
    train_ids = set()
    with TRAIN_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            train_ids.add(row["id"])
            ids = row["input_ids"]
            start = row["target_start"]
            train_payload[sha_ids(ids)] = row["id"]
            train_prompt[sha_ids(ids[:start])] = (row["id"], sha_ids(ids[start:]))
            body = row["target_body_text"]
            train_target[sha_text(squash(body))].append(row["id"])
            parts = prompt_parts(row["prompt_text"])
            train_fn[sha_text(squash(body) + "\x00" + squash("\n".join(parts["suffix"])))] = row["id"]

    by_group, by_package = load_split()
    rows: dict[str, dict] = {}
    with TOKEN_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] in wanted:
                rows[row["row_id"]] = row
    if set(rows) != wanted:
        raise SystemExit("token rows do not cover the recommended ids")

    records: dict[str, dict] = {}
    for rid in accepted:
        row, profile = rows[rid], profiles[rid]
        token_row = row["token_row"]
        reasons: list[str] = []
        rec: dict = {"row_id": rid, "package_id": row["package_id"], "group_id": row["group_id"]}
        try:
            validate_row(token_row, 10**6)
            if token_row["id"] != rid or token_row["family"] != "roxygen_drafting" or token_row["target_operation"] != "replace":
                raise ValueError("identity/family/operation differs")
            if profile.get("target_rewritten") is not False or row.get("semantic_support_claim") is not False:
                raise ValueError("target rewritten or support claim")
        except ValueError as error:
            reasons.append("geometry:" + str(error))
        ids = token_row["input_ids"]
        start = token_row["target_start"]
        rec.update(sequence_tokens=len(ids), prompt_tokens=start - 1,
                   supervised_target_tokens=len(ids) - start, target_body_tokens=token_row["target_body_token_count"])
        prompt_text = tokenizer.decode(ids[1:start], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        body_text = tokenizer.decode(token_row["target_body_tokens"], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        terminal_text = tokenizer.decode(token_row["target_terminal_tokens"], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        rec["prompt_roundtrip_exact"] = encode(tokenizer, prompt_text) == ids[1:start]
        rec["target_roundtrip_exact"] = encode(tokenizer, body_text + terminal_text) == ids[start:-1]
        rec["prompt_text_sha256"] = sha_text(prompt_text)
        rec["target_body_text_sha256"] = sha_text(body_text)
        if not rec["prompt_roundtrip_exact"] or not rec["target_roundtrip_exact"]:
            reasons.append("text_roundtrip_differs")
        parts = prompt_parts(prompt_text)
        # Licence.
        source = Path(row["source_path"]).parts
        at = source.index("normalized")
        package, version = source[at + 1], source[at + 2]
        lic = package_licence(package, version)
        if package != row["package_id"]:
            lic.update(allowed=False, reason="source_path_package_differs")
        rec["licence"] = lic
        if not lic["allowed"]:
            reasons.append("licence:" + lic["reason"])
        # Split.
        split = by_group.get(row["group_id"])
        package_splits = by_package.get(norm_package(package), set()) | by_package.get(norm_package(package).replace(".", ""), set())
        rec["split"] = {"group_split": split, "package_splits": sorted(package_splits)}
        if split != "train_group":
            reasons.append(f"split:group_{split}")
        if package_splits - {"train_group"}:
            reasons.append("split:package_in_" + "+".join(sorted(package_splits - {"train_group"})))
        if not package_splits:
            reasons.append("split:package_not_in_split_manifest")
        # Duplicates against TRAIN.
        payload, prompt_key, target_key = sha_ids(ids), sha_ids(ids[:start]), sha_ids(ids[start:])
        fn_key = sha_text(squash(body_text) + "\x00" + squash("\n".join(parts["suffix"])))
        rec["keys"] = {"payload": payload, "prompt": prompt_key, "target": target_key, "documented_function": fn_key,
                       "target_text": sha_text(squash(body_text))}
        if rid in train_ids:
            reasons.append("dedup:id_in_train")
        if payload in train_payload:
            reasons.append("dedup:exact_payload_in_train:" + train_payload[payload])
        elif prompt_key in train_prompt:
            other, other_target = train_prompt[prompt_key]
            reasons.append(("dedup:exact_payload_in_train:" if other_target == target_key else "conflict:same_prompt_other_target_in_train:") + other)
        elif fn_key in train_fn:
            reasons.append("dedup:same_documented_function_in_train:" + train_fn[fn_key])
        rec["train_rows_with_same_target_text"] = len(train_target.get(rec["keys"]["target_text"], []))
        # The DAT-10-roxy8597 ledger already found the union-internal exact duplicates; the
        # within-set pass below keeps the member that ledger kept.
        rec["authoritative_dedup_status"] = auth.get(rid, {}).get("dedup_status")
        if rec["authoritative_dedup_status"] not in ("new_candidate", "union_internal_exact_duplicate"):
            reasons.append("dedup:authoritative_ledger_" + str(rec["authoritative_dedup_status"]))
        rec["over_max_sequence"] = len(ids) > MAX_SEQUENCE
        if rec["over_max_sequence"]:
            reasons.append(f"queue:sequence_over_{MAX_SEQUENCE}")
        rec["reasons"] = reasons
        rec["_text"] = {"prompt": prompt_text, "body": body_text, "terminal": terminal_text, "parts": parts}
        records[rid] = rec

    # Within-set duplicates: keep one member of each key group (the one the authoritative
    # ledger kept, else the lowest row id) and hold the rest.
    for key in ("payload", "documented_function"):
        groups = collections.defaultdict(list)
        for rid, rec in records.items():
            groups[rec["keys"][key]].append(rid)
        for members in groups.values():
            members = sorted(members, key=lambda r: (records[r]["authoritative_dedup_status"] != "new_candidate", r))
            for rid in members[1:]:
                records[rid]["reasons"].append(f"dedup:within_set_{key}:" + members[0])
    for rid, rec in records.items():
        if rec["authoritative_dedup_status"] == "union_internal_exact_duplicate" and not any(
                r.startswith("dedup:within_set_payload") for r in rec["reasons"]):
            raise SystemExit(f"{rid}: authoritative duplicate not reproduced")
    prompts = collections.defaultdict(set)
    for rid, rec in records.items():
        prompts[rec["keys"]["prompt"]].add((rid, rec["keys"]["target"]))
    for members in prompts.values():
        if len({target for _, target in members}) > 1:
            for rid, _ in members:
                records[rid]["reasons"].append("conflict:same_prompt_other_target_within_set")
    within_target = collections.Counter(rec["keys"]["target_text"] for rec in records.values())
    for rec in records.values():
        rec["within_set_rows_with_same_target_text"] = within_target[rec["keys"]["target_text"]]
        rec["mechanical_pass"] = not rec["reasons"]

    # Semantic sample.
    passing = sorted(rid for rid, rec in records.items() if rec["mechanical_pass"])
    by_stratum = collections.defaultdict(list)
    for rid in passing:
        by_stratum[stratum(records[rid]["sequence_tokens"])].append(rid)
    sizes = {name: len(by_stratum[name]) for name, _, _ in STRATA}
    allocation = allocate(sizes)
    rng = random.Random(SEED)
    sample_ids = []
    for name, _, _ in STRATA:
        sample_ids.extend(sorted(rng.sample(by_stratum[name], allocation[name])))
    sample = []
    for rid in sample_ids:
        text = records[rid]["_text"]
        parts = text["parts"]
        document = "\n".join([*parts["prefix"], text["body"], *parts["suffix"]]) + "\n"
        sample.append({"row_id": rid, "stratum": stratum(records[rid]["sequence_tokens"]), "document": document,
                       "target_body": text["body"], "region_old": parts["region_old"], "path": parts["path"]})
    r_results = run_r_checks(work, sample)
    sample_checks = []
    for item in sample:
        rid = item["row_id"]
        tags = roxygen_tags(item["target_body"])
        r = r_results[rid]
        formals = [x for x in r["formals"].split(",") if x] if r["formals"] != "NA" else None
        params = [p.replace("\\dots", "...").replace("\\ldots", "...") for p in tags["params"]]
        check_row = {
            "row_id": rid, "stratum": item["stratum"], "package_id": records[rid]["package_id"],
            "sequence_tokens": records[rid]["sequence_tokens"], "target_body_tokens": records[rid]["target_body_tokens"],
            "region_old_lines": len([x for x in item["region_old"] if x != ""]),
            "document_parses": r["doc_parse_ok"] == "1", "target_plus_suffix_parses": r["tail_parse_ok"] == "1",
            "parse_error": r["error"], "documented_object": r["object"], "documented_object_kind": r["kind"],
            "formals": formals, "roxygen": tags,
            "params_not_in_formals": None if formals is None else sorted(set(params) - set(formals)),
            "formals_not_documented": None if formals is None else sorted(set(formals) - set(params)),
            **run_air(work, item),
        }
        sample_checks.append(check_row)

    # Review dump (row text; stays in the work dir).
    lines = ["# C10 semantic spot-check review dump", ""]
    for item, chk in zip(sample, sample_checks, strict=True):
        lines += [f"## {item['row_id']} ({chk['stratum']}, {chk['package_id']}, {item['path']})", "",
                  f"checks: parse={chk['document_parses']}/{chk['target_plus_suffix_parses']} object={chk['documented_object']} "
                  f"kind={chk['documented_object_kind']} formals={chk['formals']} params_not_in_formals={chk['params_not_in_formals']} "
                  f"undocumented_formals={chk['formals_not_documented']} air_ok={chk['air_ok']} idempotent={chk['air_idempotent']} "
                  f"tags={chk['roxygen']['tags']} region_old_lines={chk['region_old_lines']}", "",
                  "### target", "```r", item["target_body"], "```", "",
                  "### code after the target (first 60 lines)", "```r", *item["document"].split("\n")[
                      len(records[item["row_id"]]["_text"]["parts"]["prefix"]) + len(item["target_body"].split("\n")):][:60], "```", ""]
    (work / "review.md").write_text("\n".join(lines), encoding="utf-8")

    # Post-sample census of the defect the spot-check found (1c0be420...): the target is
    # the tail of a roxygen block whose head sits directly above it in the full file but
    # outside the selected context. Rule: in the full-file context, skipping at most one
    # blank line, one of the two nearest non-blank lines above the target region is a
    # roxygen-style line (#', ##' or a bare ' continuation).
    census = census_split_blocks(tokenizer, set(passing))
    for rid, above in census.items():
        records[rid]["reasons"].append("census:split_roxygen_block_head_outside_selected_context")
        records[rid]["census_lines_above_sha256"] = sha_text("\n".join(above))
    for rec in records.values():
        rec.pop("_text")
    write_jsonl(work / "mechanical.jsonl", [records[rid] for rid in accepted])
    write_jsonl(work / "sample-checks.jsonl", sample_checks)
    reason_counts = collections.Counter(r.split(":")[0] + ":" + r.split(":")[1] for rec in records.values() for r in rec["reasons"])
    rows_by_first = collections.Counter(rec["reasons"][0].split(":")[0] + ":" + rec["reasons"][0].split(":")[1]
                                        for rec in records.values() if rec["reasons"])
    lengths = [rec["sequence_tokens"] for rec in records.values() if rec["mechanical_pass"]]
    write_json(work / "checks.json", {
        "schema": "sepalith.c10.roxygen-admission-checks.v1",
        "pins": pins,
        "recommended_ids_rebuilt_sha256": rebuilt,
        "recommended": len(accepted),
        "mechanical_pass": len(passing),
        "mechanical_held": len(accepted) - len(passing),
        "reason_occurrences": dict(sorted(reason_counts.items())),
        "held_rows_by_first_reason": dict(sorted(rows_by_first.items())),
        "licence_families_of_passing_rows": dict(sorted(collections.Counter(records[r]["licence"].get("family") for r in passing).items())),
        "text_roundtrip_exact_rows": sum(1 for r in records.values() if r["prompt_roundtrip_exact"] and r["target_roundtrip_exact"]),
        "passing_sequence_tokens": {"max": max(lengths), "over_4096": sum(n > 4096 for n in lengths),
                                    "over_8192": sum(n > 8192 for n in lengths), "over_16384": sum(n > MAX_SEQUENCE for n in lengths)},
        "same_target_text_as_a_train_row": sum(1 for r in passing if records[r]["train_rows_with_same_target_text"]),
        "sample": {"seed": SEED, "strata_sizes": sizes, "allocation": allocation, "ids": sample_ids},
        "post_sample_census": {
            "defect": "target is the tail of a roxygen block whose head is directly above it in the full file but outside the selected context",
            "rule": "full-file context; skip at most one blank line above the target; the next line up is roxygen-style (^\\s*(#+'|')), or it is a plain # comment directly below a roxygen-style line",
            "scope_rows": len(passing), "hits": len(census), "ids": sorted(census),
        },
    })
    print(json.dumps({"mechanical_pass": len(passing), "census_hits": sorted(census), "held_by_first_reason": rows_by_first,
                      "allocation": allocation}, indent=1, default=str))


# --------------------------------------------------------------------------- finalize


def finalize(args: argparse.Namespace) -> None:
    work: Path = args.work_dir
    out: Path = args.output_dir
    checks = json.loads((work / "checks.json").read_text())
    for name, pin in checks["pins"].items():
        if sha256_file(Path(pin["path"])) != pin["sha256"]:
            raise SystemExit(f"pin changed since check: {name}")
    records = {r["row_id"]: r for r in map(json.loads, (work / "mechanical.jsonl").read_text().splitlines())}
    sample_checks = {r["row_id"]: r for r in map(json.loads, (work / "sample-checks.jsonl").read_text().splitlines())}
    classes = json.loads(args.classifications.read_text())
    labels = classes["rows"]
    if set(labels) != set(checks["sample"]["ids"]) or any(v["class"] not in CLASSES for v in labels.values()):
        raise SystemExit("classifications do not cover the sample exactly")
    counts = collections.Counter(v["class"] for v in labels.values())
    defective = sorted(k for k, v in labels.items() if v["class"] == "defective")
    n = len(labels)
    rate = len(defective) / n
    interval = wilson(len(defective), n)
    if rate <= 0.05:
        outcome = "continue"
    elif rate <= 0.10:
        outcome = "continue_list_defect_kinds"
    else:
        outcome = "stop-for-user"
    for rid in defective:
        records[rid]["reasons"].append("semantic:defective_in_spot_check")
        records[rid]["mechanical_pass"] = False
    admitted = sorted(rid for rid, rec in records.items() if not rec["reasons"])
    held = {rid: rec["reasons"] for rid, rec in sorted(records.items()) if rec["reasons"]}
    if any(records[rid]["over_max_sequence"] for rid in admitted):
        raise SystemExit("an admitted row exceeds the recipe context")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "arm-b-cohort-rows.jsonl").exists():
        raise SystemExit("fresh output required")

    # Cohort rows: the 15,006 TRAIN rows verbatim, then the admitted rows in id order.
    tokenizer = load_tokenizer()
    wanted = set(admitted)
    extra = []
    with TOKEN_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["row_id"] not in wanted:
                continue
            token_row = dict(row["token_row"])
            ids, start = token_row["input_ids"], token_row["target_start"]
            prompt_text = tokenizer.decode(ids[1:start], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            body = tokenizer.decode(token_row["target_body_tokens"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            terminal = tokenizer.decode(token_row["target_terminal_tokens"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if sha_text(prompt_text) != records[row["row_id"]]["prompt_text_sha256"]:
                raise SystemExit("decoded prompt differs from check")
            # TRAIN-row convention: target_body_text excludes the newline that separates the body
            # from the terminal line; target_text is body + "\n" + terminal.
            if not body.endswith("\n"):
                raise SystemExit(f"{row['row_id']}: body lacks the terminal separator")
            token_row.update(prompt_text=prompt_text, target_body_text=body[:-1], target_text=body + terminal)
            validate_row(token_row, MAX_SEQUENCE)
            extra.append(token_row)
    extra.sort(key=lambda r: r["id"])
    if [r["id"] for r in extra] != admitted:
        raise SystemExit("admitted rows not found")
    cohort = out / "arm-b-cohort-rows.jsonl"
    tmp = cohort.with_name(cohort.name + ".tmp")
    # TRAIN bytes verbatim, then the admitted rows.
    with tmp.open("wb") as sink:
        with TRAIN_ROWS.open("rb") as source:
            shutil.copyfileobj(source, sink)
        for row in extra:
            sink.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    tmp.replace(cohort)
    cohort_sha = sha256_file(cohort)
    ids_seen = set()
    with cohort.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in ids_seen:
                raise SystemExit("duplicate id in cohort")
            ids_seen.add(row["id"])
            validate_row(row, MAX_SEQUENCE)
    write_json(out / "admitted-ids.json", {"schema": "sepalith.c10.roxygen-admitted-ids.v1", "count": len(admitted), "ids": admitted})
    write_json(out / "held-ids.json", {"schema": "sepalith.c10.roxygen-held-ids.v1", "count": len(held), "reasons": held})
    for name in ("mechanical.jsonl", "sample-checks.jsonl", "checks.json", "recommended-context-ids.json"):
        shutil.copyfile(work / name, out / name)
    shutil.copyfile(args.classifications, out / "semantic-classifications.json")

    reason_rows = collections.Counter()
    for reasons in held.values():
        first = reasons[0].split(":")
        reason_rows[first[0] + ":" + first[1]] += 1
    tokens = collections.Counter()
    for row in extra:
        tokens["input"] += len(row["input_ids"])
        tokens["supervised"] += len(row["input_ids"]) - row["target_start"]
        tokens["body"] += row["target_body_token_count"]
    lengths = [len(r["input_ids"]) for r in extra]
    receipt = {
        "schema": "sepalith.c10.roxygen-admission-receipt.v1",
        "card": "C10",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "scope": {"recommended": checks["recommended"], "source": "DAT-10 roxygen context admission review (3,903 recommend_context_admission_pending_root_semantic_review)",
                  "recommended_ids_ledger_rebuilt_sha256": checks["recommended_ids_rebuilt_sha256"],
                  "recommended_ids_ledger_receipt_sha256": RECOMMENDED_IDS_SHA256},
        "pins": checks["pins"],
        "checks": {
            "geometry": "trainer validate_row (BOS 0, EOS 1, PRM03 boundary, complete body + terminal + EOS), family roxygen_drafting, operation replace, target not rewritten",
            "licence": "per-package provenance ledger /mnt/nas/home/sepalith/provenance/<pkg>.json; package and version must equal the row's source path; standing rule SYSTEMS.md section 7 (MIT/Apache/BSD/GPL-family/CC-BY/CC0, NC/ND/unspecified excluded); any allowed alternative of an 'A | B' licence suffices",
            "split": "row group_id is train_group in DAT-02-global-split-v2.json and the normalized package name appears in no other split",
            "dedup": "against the 15,006 TRAIN rows: exact token payload, same prompt (same target = duplicate, other target = conflict), same documented function (whitespace-normalized target text plus the code after it); within the set: exact payload and same documented function keep the lowest id; same prompt with different targets holds all; DAT-10-roxy8597 authoritative ledger status new_candidate",
            "length": f"no truncation; rows up to the recipe's {MAX_SEQUENCE}-token context are admitted and bucketed by actual length",
        },
        "results": {
            "admitted": len(admitted),
            "held": len(held),
            "held_by_first_reason": dict(sorted(reason_rows.items())),
            "reason_occurrences": checks["reason_occurrences"],
            "licence_families_admitted": dict(sorted(collections.Counter(records[r]["licence"]["family"] for r in admitted).items())),
            "same_target_text_as_a_train_row_admitted": sum(1 for r in admitted if records[r]["train_rows_with_same_target_text"]),
        },
        "post_sample_census": checks["post_sample_census"],
        "semantic_spot_check": {
            "method": "C06 method: seed 3407, decode with the parent tokenizer, R parse of the applied document, roxygen @param versus formals, air format idempotence, hand classification",
            "strata": checks["sample"]["strata_sizes"], "allocation": checks["sample"]["allocation"],
            "n": n, "counts": {c: counts.get(c, 0) for c in CLASSES},
            "defective": [{"row_id": k, "reason": labels[k]["reason"]} for k in defective],
            "cosmetic": [{"row_id": k, "reason": v["reason"]} for k, v in sorted(labels.items()) if v["class"] == "cosmetic"],
            "defect_rate": round(rate, 4), "defect_rate_wilson95": interval,
            "automatic": {
                "document_parses": sum(1 for r in sample_checks.values() if r["document_parses"]),
                "air_idempotent": sum(1 for r in sample_checks.values() if r["air_idempotent"]),
                "params_not_in_formals_rows": sum(1 for r in sample_checks.values() if r["params_not_in_formals"]),
            },
            "rule": "C06 bands: <=5% continue; 5-10% continue and list defect kinds; >10% stop-for-user",
            "outcome": outcome,
        },
        "cohort": {
            "rows": {"path": str(cohort), "sha256": cohort_sha, "bytes": cohort.stat().st_size, "count": len(ids_seen)},
            "composition": {"train_rows": 15006, "admitted_roxygen_rows": len(admitted)},
            "admitted_tokens": dict(tokens),
            "admitted_sequence_tokens": {"max": max(lengths), "over_2048": sum(n > 2048 for n in lengths),
                                         "over_4096": sum(n > 4096 for n in lengths), "over_8192": sum(n > 8192 for n in lengths),
                                         "over_16384": sum(n > 16384 for n in lengths)},
            "target_truncation": False,
        },
        "outputs": {name: {"path": str(out / name), "sha256": sha256_file(out / name)}
                    for name in ("admitted-ids.json", "held-ids.json", "mechanical.jsonl", "sample-checks.jsonl",
                                 "semantic-classifications.json", "recommended-context-ids.json")},
        "guards": {"dev_payloads_opened": False, "final_set_opened": False, "upstream_pools_opened": False, "cuda": False,
                   "training_launched": False},
    }
    write_json(out / "receipt.json", receipt)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt["results"], indent=1))
    print(json.dumps(receipt["semantic_spot_check"]["counts"]), outcome, cohort_sha)


# --------------------------------------------------------------------------- rebase-cohort


def rebase_cohort(args: argparse.Namespace) -> None:
    """Rebuild the arm-B cohort on a corrected TRAIN row file (same 15,006 ids).

    The admitted roxygen rows are copied byte for byte from the pinned earlier cohort;
    the dedup checks against TRAIN are repeated on the new TRAIN rows.
    """
    out: Path = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output required")
    base, train = args.base_cohort, args.train_rows
    if sha256_file(base) != args.base_cohort_sha256 or sha256_file(train) != args.train_rows_sha256:
        raise SystemExit("input hash differs")
    admitted = json.loads((args.admission_dir / "admitted-ids.json").read_text())["ids"]
    records = {r["row_id"]: r for r in map(json.loads, (args.admission_dir / "mechanical.jsonl").read_text().splitlines())}
    old_train_ids, tail = [], []
    with base.open("rb") as stream:
        for index, line in enumerate(stream):
            if index < 15006:
                old_train_ids.append(json.loads(line)["id"])
            else:
                tail.append(line)
    if [json.loads(line)["id"] for line in tail] != admitted:
        raise SystemExit("base cohort tail is not the admitted rows")
    new_ids, payload, prompt, fn = [], {}, {}, {}
    changed = 0
    old_by_id = {}
    with TRAIN_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            old_by_id[row["id"]] = sha_ids(row["input_ids"])
    with train.open() as stream:
        for line in stream:
            row = json.loads(line)
            validate_row(row, MAX_SEQUENCE)
            new_ids.append(row["id"])
            ids, start = row["input_ids"], row["target_start"]
            payload[sha_ids(ids)] = row["id"]
            prompt[sha_ids(ids[:start])] = row["id"]
            parts = prompt_parts(row["prompt_text"])
            fn[sha_text(squash(row["target_body_text"]) + "\x00" + squash("\n".join(parts["suffix"])))] = row["id"]
            changed += old_by_id.get(row["id"]) != sha_ids(ids)
    if new_ids != old_train_ids:
        raise SystemExit("TRAIN row ids or order differ from the base cohort")
    collisions = [rid for rid in admitted if records[rid]["keys"]["payload"] in payload
                  or records[rid]["keys"]["prompt"] in prompt or records[rid]["keys"]["documented_function"] in fn]
    if collisions:
        raise SystemExit(f"admitted rows collide with the new TRAIN rows: {collisions[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    cohort = out / "arm-b-cohort-rows.jsonl"
    tmp = cohort.with_name(cohort.name + ".tmp")
    with tmp.open("wb") as sink:
        with train.open("rb") as source:
            shutil.copyfileobj(source, sink)
        sink.writelines(tail)
    tmp.replace(cohort)
    seen = set()
    with cohort.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in seen:
                raise SystemExit("duplicate id in cohort")
            seen.add(row["id"])
            validate_row(row, MAX_SEQUENCE)
    receipt = {
        "schema": "sepalith.c10.arm-b-cohort-rebase.v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "base_cohort": {"path": str(base), "sha256": args.base_cohort_sha256},
        "train_rows": {"path": str(train), "sha256": args.train_rows_sha256, "rows": len(new_ids),
                       "rows_changed_vs_v3": changed},
        "admitted_roxygen_rows": len(admitted), "admitted_rows_copied_byte_for_byte": True,
        "dedup_vs_new_train": {"payload_prompt_or_documented_function_collisions": 0},
        "cohort": {"path": str(cohort), "sha256": sha256_file(cohort), "bytes": cohort.stat().st_size, "rows": len(seen)},
    }
    write_json(out / "cohort-receipt.json", receipt)
    print(json.dumps(receipt, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--work-dir", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--work-dir", type=Path, required=True)
    f.add_argument("--output-dir", type=Path, required=True)
    f.add_argument("--classifications", type=Path, default=HERE / "semantic-classifications.json")
    f.add_argument("--receipt", type=Path, default=HERE.parent / "receipts/C10-roxygen-admission.json")
    r = sub.add_parser("rebase-cohort")
    r.add_argument("--base-cohort", type=Path, required=True)
    r.add_argument("--base-cohort-sha256", required=True)
    r.add_argument("--train-rows", type=Path, required=True)
    r.add_argument("--train-rows-sha256", required=True)
    r.add_argument("--admission-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"check": check, "finalize": finalize, "rebase-cohort": rebase_cohort}[args.command](args)


if __name__ == "__main__":
    main()
