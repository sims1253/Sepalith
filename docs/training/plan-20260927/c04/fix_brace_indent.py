#!/usr/bin/env python3
"""C04: re-indent the misindented closing brace of the 486 repaired finish_block rows.

Card C06 found 486 repaired finish_block TRAIN rows whose final `}` is
indented differently from the header of the block it closes (for example `  }`
under a column-0 `f <- function(x) {`). On 2026-10-12 the user decided to fix
them. This script writes a NEW data directory; v3 is read-only.

For each listed row:

1. Rebuild the applied document from the context sidecar (unchanged prompt
   and replacement range) and let R find the brace pair the target's final
   `}` closes, plus the line where the owning construct (the header) starts.
2. Replace only the leading whitespace of the target's last line (`}`) with
   the header line's leading whitespace.
3. Re-tokenize the target exactly as ``campaign_protocol.build_training_row``
   does: encode ``body + "\\n" + TERMINAL`` with the parent tokenizer
   (``add_special_tokens=False, split_special_tokens=True``), take the body
   tokens as the prefix before the unchanged terminal tokens, check that the
   joint prompt+target encoding keeps the boundary, and rebuild ``input_ids`` as
   BOS + prompt ids + target ids + EOS.

Every other row is copied byte for byte. The context sidecar is copied
unchanged; the repair ledger gets one appended entry per fixed row. The script
then verifies every fixed row (decode, R parse, brace column, air stability,
lengths) and writes ``brace-reindent-verification.json`` and
``materialization.json``.

Usage (training interpreter, CPU):
    fix_brace_indent.py --c06-receipt RECEIPT.json --output DIR [--air PATH]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3")
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
V3_PINS = {
    "train-token-rows.jsonl": "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e",
    "context-sidecar.jsonl": "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a",
    "repair-ledger.jsonl": "254d6ef418edec64c79047f18a546dc1af112a84e709b2fe6233d2453bc33cfa",
}
C06_RECEIPT_SHA256 = "91f456ca4f77e16fb47e45d06b6a4188d52b5ba788a83ef17173b14638b89fec"
TERMINAL = ">>>>>>> UPDATED"
MAX_SEQUENCE_TOKENS = 16384
GENERATION_MAX_NEW_TOKENS = 1024
CHANGED_FIELDS = ("input_ids", "target_body_text", "target_body_token_count", "target_body_tokens", "target_text",
                  "target_token_count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def require(condition: object, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def utf16_index(line: str, units: int) -> int:
    used = 0
    for index, char in enumerate(line):
        if used == units:
            return index
        used += len(char.encode("utf-16-le")) // 2
        if used > units:
            raise ValueError("utf16 split")
    if used != units:
        raise ValueError("utf16 overflow")
    return len(line)


def offsets(document: str, replacement_range: dict) -> tuple[int, int]:
    lines = document.split("\n")

    def offset(position: dict) -> int:
        return sum(len(x) + 1 for x in lines[: position["line"]]) + utf16_index(lines[position["line"]], position["character"])

    return offset(replacement_range["start"]), offset(replacement_range["end"])


def leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def serialize(row: dict, ensure_ascii: bool) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii) + "\n"


def brace_pairs(documents: dict[str, str], work: Path) -> dict[str, tuple[bool, list[tuple[int, int, int]], str]]:
    """R parse of each document plus its (open, close, owner) brace pairs, 0-based lines."""
    manifest = ["key\tpath"]
    for key, text in documents.items():
        path = work / f"{key}.R"
        path.write_text(text)
        manifest.append(f"{key}\t{path}")
    (work / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    subprocess.run(["Rscript", "--vanilla", str(HERE / "brace_pairs.R"), str(work / "manifest.tsv"), str(work / "pairs.tsv")],
                   check=True, timeout=3600)
    out = {}
    for line in (work / "pairs.tsv").read_text().splitlines()[1:]:
        key, ok, pairs, err = (line.split("\t") + [""] * 4)[:4]
        parsed = []
        if pairs:
            for item in pairs.split(","):
                o, c, w = item.split(":")
                parsed.append((int(o) - 1, int(c) - 1, int(w) - 1))
        out[key] = (ok == "1", parsed, err)
    return out


def final_pair(pairs: list[tuple[int, int, int]], first: int, last: int) -> tuple[int, int, int] | None:
    inside = [p for p in pairs if first <= p[1] <= last]
    return max(inside, key=lambda p: p[1], default=None)


def run_air(path: Path, air: str) -> tuple[bool, bool, str]:
    once, twice = path.with_suffix(".air1.R"), path.with_suffix(".air2.R")
    shutil.copyfile(path, once)
    first = subprocess.run([air, "format", str(once)], capture_output=True, text=True, timeout=120)
    if first.returncode != 0:
        return False, False, ""
    shutil.copyfile(once, twice)
    second = subprocess.run([air, "format", str(twice)], capture_output=True, text=True, timeout=120)
    return second.returncode == 0, once.read_text() == twice.read_text(), once.read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c06-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--air", default=os.path.expanduser("~/.local/bin/air"))
    args = parser.parse_args()
    out = args.output
    require(not out.exists(), f"output directory must be new: {out}")
    require(sha256_file(args.c06_receipt) == C06_RECEIPT_SHA256, "C06 receipt differs")
    require(sha256_file(PARENT / "tokenizer.json") == TOKENIZER_SHA256, "parent tokenizer differs")
    for name, expected in V3_PINS.items():
        require(sha256_file(V3 / name) == expected, f"v3 input differs: {name}")
    receipt = json.loads(args.c06_receipt.read_text())
    targets = receipt["population_census"]["repaired_final_brace_indent_mismatch_ids"]
    require(len(targets) == len(set(targets)) == 486, "expected 486 distinct row ids")
    wanted = set(targets)

    from transformers import AutoTokenizer

    from sepalith.campaign_protocol import validate_training_row

    tokenizer = AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True, use_fast=True, trust_remote_code=False)

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True))

    terminal_ids = encode(TERMINAL)
    ledger_lines = (V3 / "repair-ledger.jsonl").read_text().splitlines(keepends=True)
    repaired_ids = {json.loads(line)["row_id"] for line in ledger_lines}
    require(wanted <= repaired_ids, "a listed row is not a repaired row")

    contexts = {}
    with (V3 / "context-sidecar.jsonl").open() as stream:
        for line in stream:
            ctx = json.loads(line)
            if ctx["row_id"] in wanted:
                contexts[ctx["row_id"]] = ctx
    require(set(contexts) == wanted, "context join failed")

    raw_lines = (V3 / "train-token-rows.jsonl").read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines]
    ascii_style = all(serialize(r, True) == line for r, line in zip(rows, raw_lines, strict=True))
    require(ascii_style or all(serialize(r, False) == line for r, line in zip(rows, raw_lines, strict=True)),
            "cannot reproduce the v3 row serialization")
    # The encoder reproduces every v3 target exactly, so re-tokenized rows match the original build.
    encoder_matches = sum(encode(r["target_text"]) == r["input_ids"][r["target_start"]:-1] for r in rows)
    require(encoder_matches == len(rows), f"encoder reproduces only {encoder_matches} of {len(rows)} targets")

    by_id = {r["id"]: i for i, r in enumerate(rows)}
    require(wanted <= set(by_id), "listed rows missing from v3")
    work = Path(tempfile.mkdtemp(prefix="c04-brace-"))
    before_docs, plans = {}, {}
    for rid in targets:
        row, ctx = rows[by_id[rid]], contexts[rid]
        require(row["family"] == "finish_block" and row["split"] == "train", f"not a TRAIN finish row: {rid}")
        document = ctx["source_identity"]["source_provenance"]["selection_source"]["document_text"]
        rr = ctx["context"]["replacement_range"]
        require(sha256_text(document) == rr["content_sha256"], f"document hash differs: {rid}")
        left, right = offsets(document, rr)
        body = row["target_body_text"]
        applied = document[:left] + body + document[right:]
        first = applied[:left].count("\n")
        plans[rid] = {"document": document, "left": left, "right": right, "first": first, "last": first + body.count("\n")}
        before_docs[rid] = applied
    (work / "before").mkdir()
    parsed_before = brace_pairs(before_docs, work / "before")

    new_rows, ledger_add, header_ws = {}, [], {}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for rid in targets:
        row, plan = rows[by_id[rid]], plans[rid]
        ok, pairs, err = parsed_before[rid]
        require(ok, f"v3 applied document does not parse: {rid} {err}")
        pair = final_pair(pairs, plan["first"], plan["last"])
        require(pair is not None, f"no final brace pair: {rid}")
        lines = before_docs[rid].split("\n")
        opener, close, owner = pair
        body_lines = row["target_body_text"].split("\n")
        require(close == plan["last"] and body_lines[-1].strip() == "}", f"final brace is not the last target line: {rid}")
        header = leading_ws(lines[owner])
        header_ws[rid] = header
        old = leading_ws(lines[close])
        require(old not in (header, leading_ws(lines[opener])), f"row is not misindented: {rid}")
        body_lines[-1] = header + "}"
        body = "\n".join(body_lines)
        target_text = body + "\n" + TERMINAL
        prompt_ids = row["input_ids"][1:row["target_start"]]
        target_ids = encode(target_text)
        require(target_ids[-len(terminal_ids):] == terminal_ids == row["target_terminal_tokens"], f"terminal differs: {rid}")
        require(encode(row["prompt_text"] + target_text) == prompt_ids + target_ids, f"prompt/target boundary retokenizes: {rid}")
        body_ids = target_ids[:-len(terminal_ids)]
        require(0 not in body_ids and 1 not in body_ids, f"body encodes BOS/EOS: {rid}")
        new = dict(row)
        new.update(input_ids=[0, *prompt_ids, *target_ids, 1], target_body_text=body, target_body_tokens=body_ids,
                   target_body_token_count=len(body_ids), target_text=target_text, target_token_count=len(target_ids))
        validate_training_row(new)
        require({k for k in row if row[k] != new[k]} <= set(CHANGED_FIELDS), f"unexpected field change: {rid}")
        new_rows[rid] = new
        applied = plan["document"][:plan["left"]] + body + plan["document"][plan["right"]:]
        ledger_add.append({
            "row_id": rid, "repair": "final_brace_reindent_v4", "created_at_utc": now,
            "decided_by": "user decision 2026-10-12 (fix the C06-listed rows)",
            "c06_receipt_sha256": C06_RECEIPT_SHA256,
            "close_line_in_applied_document": close, "owner_line_in_applied_document": owner,
            "close_indent_before": len(old), "close_indent_after": len(header),
            "previous_target_body_sha256": sha256_text(row["target_body_text"]),
            "repaired_target_body_sha256": sha256_text(body),
            "previous_applied_document_sha256": sha256_text(before_docs[rid]),
            "applied_document_sha256": sha256_text(applied),
            "replacement_range": contexts[rid]["context"]["replacement_range"],
            "prompt_ids_unchanged": True,
            "target_token_count_before": row["target_token_count"], "target_token_count_after": len(target_ids),
        })

    # Write the new directory: unchanged rows are byte-identical copies.
    out.mkdir(parents=True)
    with (out / "train-token-rows.jsonl").open("w") as stream:
        for row, line in zip(rows, raw_lines, strict=True):
            stream.write(serialize(new_rows[row["id"]], ascii_style) if row["id"] in new_rows else line)
    shutil.copyfile(V3 / "context-sidecar.jsonl", out / "context-sidecar.jsonl")
    with (out / "repair-ledger.jsonl").open("w") as stream:
        stream.writelines(ledger_lines)
        for entry in ledger_add:
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")

    # Verify from the written files.
    written = [json.loads(line) for line in (out / "train-token-rows.jsonl").read_text().splitlines()]
    written_lines = (out / "train-token-rows.jsonl").read_text().splitlines(keepends=True)
    unchanged_identical = sum(line == raw for line, raw, r in zip(written_lines, raw_lines, written, strict=True)
                              if r["id"] not in wanted)
    after_docs = {}
    checks = collections.Counter()
    max_seq = max_target = 0
    for row in written:
        max_seq = max(max_seq, len(row["input_ids"]))
        max_target = max(max_target, len(row["input_ids"]) - row["target_start"])
        if row["id"] not in wanted:
            continue
        plan = plans[row["id"]]
        intended = rows[by_id[row["id"]]]["target_body_text"].split("\n")
        intended[-1] = header_ws[row["id"]] + "}"
        checks["decoded_target_equals_intended"] += (
            tokenizer.decode(row["target_body_tokens"], skip_special_tokens=False) == "\n".join(intended) + "\n"
            and tokenizer.decode(row["input_ids"][row["target_start"]:-1], skip_special_tokens=False) == row["target_text"]
            and row["target_body_text"] == "\n".join(intended))
        checks["layout_ok"] += row["input_ids"][row["target_start"]:] == row["target_body_tokens"] + row["target_terminal_tokens"] + [1]
        checks["prompt_unchanged"] += row["input_ids"][:row["target_start"]] == rows[by_id[row["id"]]]["input_ids"][:row["target_start"]]
        after_docs[row["id"]] = plan["document"][:plan["left"]] + row["target_body_text"] + plan["document"][plan["right"]:]
        checks["within_length_limits"] += (len(row["input_ids"]) <= MAX_SEQUENCE_TOKENS
                                           and len(row["input_ids"]) - row["target_start"] <= GENERATION_MAX_NEW_TOKENS)
    (work / "after").mkdir()
    parsed_after = brace_pairs(after_docs, work / "after")
    for rid, document in after_docs.items():
        ok, pairs, _err = parsed_after[rid]
        checks["r_parses"] += ok
        pair = final_pair(pairs, plans[rid]["first"], plans[rid]["last"]) if ok else None
        lines = document.split("\n")
        if pair is not None:
            checks["brace_column_equals_header_column"] += leading_ws(lines[pair[1]]) == leading_ws(lines[pair[2]])
        air_ok, idempotent, formatted = run_air(work / "after" / f"{rid}.R", args.air)
        checks["air_ok"] += air_ok
        checks["air_idempotent"] += air_ok and idempotent
        if air_ok:
            # Finish rows end at EOF, so the target's final `}` is the document's last non-blank line;
            # air must leave it exactly as re-indented.
            checks["air_keeps_final_brace_line"] += ([x for x in formatted.split("\n") if x.strip()][-1:]
                                                     == [x for x in lines if x.strip()][-1:])
    for key in ("decoded_target_equals_intended", "layout_ok", "prompt_unchanged", "within_length_limits", "r_parses",
                "brace_column_equals_header_column", "air_ok", "air_idempotent", "air_keeps_final_brace_line"):
        require(checks[key] == len(wanted), f"verification failed: {key} {checks[key]} of {len(wanted)}")
    require(unchanged_identical == len(rows) - len(wanted), "an unchanged row is not byte-identical")
    shutil.rmtree(work)

    files = {name: {"bytes": (out / name).stat().st_size, "sha256": sha256_file(out / name)}
             for name in ("train-token-rows.jsonl", "context-sidecar.jsonl", "repair-ledger.jsonl")}
    verification = {
        "schema": "sepalith.dat10.brace-reindent-verification.v1", "status": "pass", "created_at_utc": now,
        "fixed_rows": len(wanted), "rows": len(rows), "unchanged_rows_byte_identical": unchanged_identical,
        "checks": dict(sorted(checks.items())),
        "encoder_reproduces_all_v3_targets": encoder_matches,
        "max_sequence_tokens": max_seq, "max_target_tokens_including_eos": max_target,
        "limits": {"max_sequence_tokens": MAX_SEQUENCE_TOKENS, "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS},
        "target_token_delta": dict(sorted(collections.Counter(e["target_token_count_after"] - e["target_token_count_before"]
                                                              for e in ledger_add).items())),
        "indent_change": dict(sorted(collections.Counter(f"{e['close_indent_before']}->{e['close_indent_after']}"
                                                         for e in ledger_add).items())),
        "air_version": subprocess.run([args.air, "--version"], capture_output=True, text=True).stdout.strip(),
        "r_version": (subprocess.run(["Rscript", "--version"], capture_output=True, text=True).stdout
                      or subprocess.run(["Rscript", "--version"], capture_output=True, text=True).stderr).strip(),
    }
    (out / "brace-reindent-verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    materialization = {
        "schema": "sepalith.dat10.finish-source-repair.v4-braceindent", "created_at_utc": now,
        "parent_data": {"path": str(V3), "pins": V3_PINS},
        "operation": "re-indent the final closing brace of 486 C06-listed repaired finish_block targets to the header column; re-tokenize the target with the parent tokenizer exactly as campaign_protocol.build_training_row does",
        "c06_receipt": {"path": "docs/training/plan-20260927/receipts/C06-finish-repair-check.json (branch plan/C06-finish-repair-check)",
                        "sha256": C06_RECEIPT_SHA256},
        "tokenizer": {"path": str(PARENT / "tokenizer.json"), "sha256": TOKENIZER_SHA256},
        "script": {"path": "docs/training/plan-20260927/c04/fix_brace_indent.py", "sha256": sha256_file(Path(__file__))},
        "changed_fields": list(CHANGED_FIELDS), "prompt_and_range_unchanged": True,
        "artifacts": {**files, "brace-reindent-verification.json": {
            "bytes": (out / "brace-reindent-verification.json").stat().st_size,
            "sha256": sha256_file(out / "brace-reindent-verification.json")}},
        "coverage": {"rows": len(rows), "changed_rows": len(wanted), "unchanged_rows_byte_exact": unchanged_identical},
    }
    (out / "materialization.json").write_text(json.dumps(materialization, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": files, "verification": verification}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
