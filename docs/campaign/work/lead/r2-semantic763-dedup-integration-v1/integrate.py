#!/usr/bin/env python3
"""Cross-corpus semantic-candidate deduplication and exact accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

AUTHORITATIVE_SHA = "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"
EXISTING_CONTEXT_SHA = "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a"
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
SEMANTIC_IDS = 763
PINS = {
    "training_sidecar": "94e77b61f9c708626d0700cf6992ffdf364885fd1797171d5fde0767f89fa9a7",
    "prediction_inputs": "233c39c62517b6468c3335bdc0bc2e6352aed8c0359da4f9f585874ebecf1cb1",
    "profiles16": "b94742822c68393ef0fa71c6470c991e91f4d613b3d75795b7dc136e5307daf2",
    "profiles32": "0a3032b97d0d46a42da175981f7d6d6622b3c8de46bd0fd3cd96f5140b4444a9",
    "holds16": "37df1c16ddfa0d7145c007e3637f416a1b5591ce20a1680a9b648816097e332a",
    "holds32": "82424614fc26b1bcc5fdc003cb187c20ec999354960fc8fbb02741d20a4bfb4d",
    "preparation_holds": "00204856ec9daccfbea06fbf35afe9d450c77e4c43c8d8e851994d01806e864c",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path, expected_sha: str | None = None):
    h = hashlib.sha256(); count = 0
    with Path(path).open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            h.update(raw); count += 1
            try:
                yield number, json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid JSONL:{path.name}:{number}") from exc
    if expected_sha is not None:
        require(h.hexdigest() == expected_sha, f"input hash differs:{path}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); fd2 = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd2); os.close(fd2)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def target_key(row):
    return digest_text(row["target_body_text"])


def pair_key(row):
    return digest_text(row["prompt_text"] + "\0" + row["target_text"])


def normalized_source_paths(identity):
    values = set()
    def add(value):
        if not isinstance(value, str) or not value: return
        text = value.replace("\\", "/")
        marker = "/R/"
        values.add(text[text.index(marker) + 1:] if marker in text else text)
    if isinstance(identity, dict):
        add(identity.get("source_path")); add(identity.get("path"))
        ref = identity.get("source_ref", {}); prov = identity.get("source_provenance", {})
        if isinstance(ref, dict): add(ref.get("source_path")); add(ref.get("path"))
        if isinstance(prov, dict):
            add(prov.get("source_snapshot_path")); add(prov.get("path"))
            parent = prov.get("parent_identity", {})
            if isinstance(parent, dict): add(parent.get("path"))
    return values


def source_hashes(identity):
    keys = ("source_sha256", "source_snapshot_sha256", "before_snapshot_sha256", "after_snapshot_sha256", "post_edit_snapshot_sha256")
    out = set()
    def take(value):
        if isinstance(value, dict):
            for key in keys:
                got = value.get(key)
                if isinstance(got, str) and len(got) == 64: out.add(got)
    if isinstance(identity, dict):
        take(identity); take(identity.get("source_ref")); take(identity.get("source_provenance"))
    return out


def validate_tokenrow(row, tokenizer):
    required = {"id", "family", "split", "package_id", "prompt_text", "target_text", "target_body_text", "input_ids", "target_body_tokens", "target_terminal_tokens", "prompt_token_count", "target_token_count", "target_start", "bos_token_id", "eos_token_id", "tokenizer_json_sha256"}
    require(required <= set(row), "candidate token row fields missing")
    require(row["family"] == "roxygen_drafting" and row["split"] == "train" and row["bos_token_id"] == 0 and row["eos_token_id"] == 1 and row["tokenizer_json_sha256"] == TOKENIZER_SHA, "candidate token identity differs")
    ids = row["input_ids"]
    require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 130560 for x in ids), "candidate token IDs invalid")
    require(ids[0] == 0 and ids[-1] == 1 and row["target_start"] == 1 + row["prompt_token_count"], "candidate token boundary differs")
    require(ids[row["target_start"]:] == row["target_body_tokens"] + row["target_terminal_tokens"] + [1] and len(ids) == 2 + row["prompt_token_count"] + row["target_token_count"], "candidate target token splice differs")
    require(tokenizer.encode(row["prompt_text"], add_special_tokens=False).ids == ids[1:row["target_start"]], "candidate prompt re-encode differs")
    require(tokenizer.encode(row["target_text"], add_special_tokens=False).ids == row["target_body_tokens"] + row["target_terminal_tokens"], "candidate terminal protocol re-encode differs")
    require(tokenizer.encode(row["prompt_text"] + row["target_text"], add_special_tokens=False).ids == ids[1:-1], "candidate complete sequence re-encode differs")


def load_existing(rows_path, context_path):
    rows = {}; by_pair = defaultdict(set); by_prompt = defaultdict(set)
    for _, row in read_jsonl(rows_path, AUTHORITATIVE_SHA):
        row_id = row.get("id"); require(isinstance(row_id, str) and row_id not in rows, "existing row ID duplicate")
        record = {"id": row_id, "pair": pair_key(row), "prompt": digest_text(row["prompt_text"]), "target": target_key(row), "package_id": row.get("package_id"), "source_hashes": set(), "source_paths": set()}
        rows[row_id] = record; by_pair[record["pair"]].add(row_id); by_prompt[record["prompt"]].add((record["target"], row_id))
    require(len(rows) == 15006, "authoritative row denominator differs")
    seen = set()
    for _, sidecar in read_jsonl(context_path, EXISTING_CONTEXT_SHA):
        row_id = sidecar.get("row_id"); require(row_id in rows and row_id not in seen, "existing context ID closure differs"); seen.add(row_id)
        identity = sidecar.get("source_identity", {}); rows[row_id]["source_hashes"] = source_hashes(identity); rows[row_id]["source_paths"] = normalized_source_paths(identity)
    require(seen == set(rows), "existing context denominator differs")
    by_source_target = defaultdict(set); by_path_target = defaultdict(set)
    for row in rows.values():
        for source_hash in row["source_hashes"]: by_source_target[(source_hash, row["target"])].add(row["id"])
        for source_path in row["source_paths"]: by_path_target[(row["package_id"], source_path, row["target"])].add(row["id"])
    return rows, by_pair, by_prompt, by_source_target, by_path_target


def load_sidecars(training_sidecar, prediction_inputs, profiles16, profiles32):
    sidecars = {}; predictions = {}; profile = {16384: {}, 32768: {}}
    for _, row in read_jsonl(training_sidecar, PINS["training_sidecar"]):
        require(row["row_id"] not in sidecars, "semantic sidecar duplicate ID"); sidecars[row["row_id"]] = row
    for _, row in read_jsonl(prediction_inputs, PINS["prediction_inputs"]):
        require(row["row_id"] not in predictions, "prediction input duplicate ID")
        require(not any(any(word in key.lower() for word in ("target", "gold", "reward", "completion")) for key in row), "prediction input exposes target-shaped field")
        predictions[row["row_id"]] = row
    for context, path, expected in ((16384, profiles16, PINS["profiles16"]), (32768, profiles32, PINS["profiles32"])):
        for _, row in read_jsonl(path, expected):
            require(row["row_id"] not in profile[context], "profile duplicate ID"); profile[context][row["row_id"]] = row
    require(len(sidecars) == len(predictions) == 762, "semantic reconstructed denominator differs")
    return sidecars, predictions, profile


def hold_map(*paths):
    result = {}
    for path, expected in paths:
        for _, row in read_jsonl(path, expected):
            row_id = row["row_id"]; reason = row.get("reason") or row.get("status") or "unknown_hold"
            require(row_id not in result, "hold ID duplicated"); result[row_id] = reason
    return result


def integrate(args):
    started = time.monotonic()
    from tokenizers import Tokenizer
    require(sha256(args.tokenizer) == TOKENIZER_SHA, "tokenizer pin differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.encode_special_tokens = True
    existing, by_pair, by_prompt, by_source_target, by_path_target = load_existing(args.existing_rows, args.existing_context)
    sidecars, predictions, profiles = load_sidecars(args.training_sidecar, args.prediction_inputs, args.profiles16, args.profiles32)
    candidates = {16384: {}, 32768: {}}
    for context, path, expected in ((16384, args.tokenrows16, "51c01ae15a8bcaf70f1e9400a617a9c65d2cbe4bd60d46decbef5783a0a17907"), (32768, args.tokenrows32, "ae3513d127804bd7ffb6912980a3cb31fdaa3ec2f0290f715f682dbfaae39b80")):
        for _, row in read_jsonl(path, expected):
            require(row["id"] not in candidates[context], "candidate duplicate ID"); validate_tokenrow(row, tokenizer); candidates[context][row["id"]] = row
    holds16 = hold_map((args.holds16, PINS["holds16"]), (args.preparation_holds, PINS["preparation_holds"])); holds32 = hold_map((args.holds32, PINS["holds32"]), (args.preparation_holds, PINS["preparation_holds"]))
    all_ids = set(candidates[32768]) | set(holds32); require(len(all_ids) == SEMANTIC_IDS and not (set(candidates[32768]) & set(holds32)), "32K semantic accounting differs")
    require(set(candidates[16384]) <= set(candidates[32768]), "16K candidate is absent at 32K")
    output_parent = args.output.parent; output_parent.mkdir(parents=True, exist_ok=True); require(not args.output.exists(), "fresh output required")
    temp = Path(tempfile.mkdtemp(prefix="." + args.output.name + ".", dir=output_parent))
    ledgers = []; selected = []; provenance = []; status_counts = Counter(); duplicates = Counter(); target_lengths = []; prompt_lengths = []; mode_counts = Counter(); helper_counts = Counter()
    try:
        for row_id in sorted(all_ids):
            if row_id not in candidates[32768]:
                status = "hold"; reason = holds32[row_id]; ledgers.append({"row_id": row_id, "status": status, "reason": reason, "available_16k": row_id in candidates[16384], "available_32k": False}); status_counts[reason] += 1; continue
            row = candidates[16384].get(row_id) or candidates[32768][row_id]
            context = 16384 if row_id in candidates[16384] else 32768
            require(row_id in sidecars and row_id in predictions and row_id in profiles[context], "candidate sidecar/profile closure differs")
            sidecar = sidecars[row_id]; prediction = predictions[row_id]; profile = profiles[context][row_id]
            require(profile["prompt_sha256"] == digest_text(row["prompt_text"]) and profile["sequence_tokens"] == len(row["input_ids"]) and profile["target_tokens"] == row["target_token_count"], "candidate profile differs")
            expected_target_sha = digest_text(row["target_body_text"] + ("" if row["target_body_text"].endswith("\n") else "\n"))
            require(sidecar["target_sha256"] == profile["target_sha256"] == expected_target_sha, "candidate complete target differs")
            require(sidecar["identity"]["row_id"] == row_id and sidecar["identity"]["package_id"] == row["package_id"] and sidecar["identity"]["split"] == "train_group" and all(sidecar["identity"]["checks"].values()), "candidate TRAIN/provenance checks differ")
            preedit = prediction["preedit_text"]; require(digest_text(preedit) == prediction["preedit_sha256"] == profile["preedit_sha256"], "candidate pre-edit identity differs")
            leak = bool(row["target_body_text"] and (row["target_body_text"] in row["prompt_text"] or row["target_body_text"] in preedit))
            reason = None; matches = []
            if leak: reason = "hold_exact_target_leak_in_prediction_context"
            elif row_id in existing:
                reason = "duplicate_existing_id" if existing[row_id]["pair"] == pair_key(row) else "contradiction_existing_id"
                matches = [row_id]
            elif pair_key(row) in by_pair:
                reason = "duplicate_existing_prompt_target"; matches = sorted(by_pair[pair_key(row)])
            else:
                prompt_matches = by_prompt.get(digest_text(row["prompt_text"]), set()); conflicting = sorted(old_id for old_target, old_id in prompt_matches if old_target != target_key(row))
                if conflicting: reason = "contradiction_existing_prompt_different_target"; matches = conflicting
            identity = sidecar["identity"]; candidate_source_hashes = source_hashes(identity) | {identity["source_sha256"]}; candidate_paths = normalized_source_paths(identity)
            if reason is None:
                source_matches = set()
                for value in candidate_source_hashes: source_matches |= by_source_target.get((value, target_key(row)), set())
                if source_matches: reason = "duplicate_existing_source_target"; matches = sorted(source_matches)
            if reason is None:
                path_matches = set()
                for value in candidate_paths: path_matches |= by_path_target.get((row["package_id"], value, target_key(row)), set())
                if path_matches: reason = "duplicate_existing_source_path_target"; matches = sorted(path_matches)
            if reason is not None:
                status_counts[reason] += 1; duplicates[reason] += 1
                ledgers.append({"row_id": row_id, "status": "excluded", "reason": reason, "matches": matches, "available_16k": row_id in candidates[16384], "available_32k": True}); continue
            tier = "new_candidate_16k" if context == 16384 else "new_candidate_32k_only"
            status_counts[tier] += 1; target_lengths.append(row["target_token_count"]); prompt_lengths.append(row["prompt_token_count"]); mode_counts[profile["mode"]] += 1
            helper_counts["with_helpers" if profile["helper_count"] else "without_helpers"] += 1
            selected.append(row)
            provenance.append({"row_id": row_id, "context_size": context, "mode": profile["mode"], "prompt_sha256": profile["prompt_sha256"], "target_sha256": profile["target_sha256"], "preedit_sha256": profile["preedit_sha256"], "postedit_source_sha256": profile["postedit_source_sha256"], "source_identity": identity, "required_helper_spans": sidecar["required_helper_spans"], "external_import_dependencies": sidecar["external_import_dependencies"], "complete_target_tokens": row["target_token_count"], "target_truncated": False, "selection_target_or_gold_used": False})
            ledgers.append({"row_id": row_id, "status": "candidate", "reason": tier, "available_16k": row_id in candidates[16384], "available_32k": True, "context_size": context})
        require(len(ledgers) == SEMANTIC_IDS and len({x["row_id"] for x in ledgers}) == SEMANTIC_IDS, "final semantic accounting differs")
        require(len(selected) + sum(1 for x in ledgers if x["status"] != "candidate") == SEMANTIC_IDS, "selected/exclusion conservation differs")
        internal_pair = defaultdict(list); internal_prompt = defaultdict(set); internal_source_target = defaultdict(list); internal_path_target = defaultdict(list)
        for row, evidence in zip(selected, provenance, strict=True):
            require(row["id"] == evidence["row_id"], "candidate/provenance order differs")
            target = target_key(row); identity = evidence["source_identity"]
            internal_pair[pair_key(row)].append(row["id"]); internal_prompt[digest_text(row["prompt_text"])].add(target)
            internal_source_target[(identity["source_sha256"], target)].append(row["id"])
            for source_path in normalized_source_paths(identity): internal_path_target[(row["package_id"], source_path, target)].append(row["id"])
        internal = {"prompt_target_duplicate_rows": sum(len(v) - 1 for v in internal_pair.values()), "prompt_contradiction_groups": sum(len(v) > 1 for v in internal_prompt.values()), "source_target_duplicate_rows": sum(len(v) - 1 for v in internal_source_target.values()), "source_path_target_duplicate_rows": sum(len(v) - 1 for v in internal_path_target.values())}
        require(not any(internal.values()), "selected candidate internal duplicate/contradiction remains")
        def write_jsonl(name, rows):
            path = temp / name
            with path.open("x") as handle:
                for value in rows: handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            return {"path": name, "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}
        outputs = {"candidate-tokenrows.jsonl": write_jsonl("candidate-tokenrows.jsonl", selected), "candidate-provenance.jsonl": write_jsonl("candidate-provenance.jsonl", provenance), "exclusion-ledger.jsonl": write_jsonl("exclusion-ledger.jsonl", ledgers)}
        def summary(values):
            if not values: return {"count": 0, "min": None, "max": None, "mean": None}
            return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}
        manifest = {"schema": "sepalith.dat10.semantic763.dedup_integration.v1", "status": "complete_review_only_root_admission_required", "training_admission": False, "semantic_denominator": SEMANTIC_IDS, "existing_denominator": len(existing), "status_counts": dict(sorted(status_counts.items())), "candidate_rows": len(selected), "candidate_ids_sha256": digest_text("\n".join(x["id"] for x in selected) + ("\n" if selected else "")), "candidate_internal_dedup": internal, "context_policy": {"prediction_rule": "16K first; 32K only when 16K unavailable; selection never reads target", "generation_reserve": 1024, "context_counts": {"16384": sum(x["context_size"] == 16384 for x in provenance), "32768": sum(x["context_size"] == 32768 for x in provenance)}}, "mode_counts": dict(sorted(mode_counts.items())), "helper_counts": dict(sorted(helper_counts.items())), "target_token_lengths": summary(target_lengths), "prompt_token_lengths": summary(prompt_lengths), "outputs": outputs, "inputs": {"authoritative_rows_sha256": AUTHORITATIVE_SHA, "existing_context_sha256": EXISTING_CONTEXT_SHA, "semantic_32k_tokenrows_sha256": "ae3513d127804bd7ffb6912980a3cb31fdaa3ec2f0290f715f682dbfaae39b80", "tokenizer_sha256": TOKENIZER_SHA}, "checks": {"exact_763_accounting": True, "exact_15006_existing_accounting": True, "complete_target_reencoded": True, "prompt_reencoded": True, "target_truncated": False, "prediction_inputs_target_shaped_fields_absent": True, "source_train_split_and_provenance_checks": True, "cross_corpus_keys": ["row ID", "exact prompt+target", "same prompt/different target", "source hash+target", "package+source path+target"]}, "elapsed_seconds": time.monotonic() - started}
        atomic_json(temp / "manifest.json", manifest)
        fd = os.open(temp, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd); os.rename(temp, args.output); fd = os.open(output_parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--existing-rows", type=Path, required=True); p.add_argument("--existing-context", type=Path, required=True)
    p.add_argument("--training-sidecar", type=Path, required=True); p.add_argument("--prediction-inputs", type=Path, required=True)
    p.add_argument("--tokenrows16", type=Path, required=True); p.add_argument("--tokenrows32", type=Path, required=True)
    p.add_argument("--profiles16", type=Path, required=True); p.add_argument("--profiles32", type=Path, required=True)
    p.add_argument("--holds16", type=Path, required=True); p.add_argument("--holds32", type=Path, required=True); p.add_argument("--preparation-holds", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); print(json.dumps(integrate(args), sort_keys=True))


if __name__ == "__main__": main()
