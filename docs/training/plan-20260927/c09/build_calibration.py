#!/usr/bin/env python3
"""C09: build the imatrix calibration text and the held-out KLD slice from TRAIN rows.

    uv run python docs/training/plan-20260927/c09/build_calibration.py \\
        --out-dir /mnt/e/sepalith/campaign-20260915/quant-work/calibration-v1 \\
        --verify-gguf /mnt/e/sepalith/campaign-20260915/quant-work/parent-11586/parent-11586-Q4_K_M.gguf

Inputs are TRAIN token rows only.  DEV75 and DEV250 are read to exclude
their ids and near-duplicates (normalized prompt, current-region and target
hashes, plus prompt line-shingle containment).  No DAT-07-final file is
opened.  See ``sepalith.quant.calibration`` for how token ids become text
that llama.cpp tokenizes back to the same ids; ``--verify-gguf`` proves it
with ``llama-tokenize`` against that GGUF's vocabulary (the vocabulary is the
same in every GGUF of this model; only the weights differ).

Writes into a new directory: ``calibration.txt`` (llama-imatrix ``-f``),
``calibration.ids.json``, ``kld-heldout.txt`` (llama-perplexity ``-f``),
``kld-heldout.ids.json`` and ``manifest.json`` (sha256 of every input and
output, selection, exclusions, verification and the exact tool flags).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages" / "sepalith" / "src"))
from sepalith.campaign_protocol import PromptContext, RENDERER_ID, TOKENIZATION_POLICY, render_prompt
from sepalith.quant import calibration as cal
from sepalith.quant.llama_tools import file_sha256, llama_bin, llama_version, refuse_sealed

TRAIN_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TRAIN_ROWS_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
DEV75 = Path("/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl")
DEV75_SHA256 = "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035"
DEV250 = Path("/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl")
DEV250_SHA256 = "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22"
TOKENIZER_JSON = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")
TOKENIZER_JSON_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
SCHEMA = "sepalith.c09.calibration.v1"


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    # Same as the training rows: split_special_tokens=True (special-looking text stays text).
    tokenizer.encode_special_tokens = True
    added = sorted({token.content for token in tokenizer.get_added_tokens_decoder().values()})

    def decode(ids):
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    def round_trips(ids):
        text = decode(ids)
        return (not cal.mentions_added_token(text, added)
                and tokenizer.encode(text, add_special_tokens=False).ids == list(ids))

    return decode, round_trips, added


def dev_cases(path: Path, expected_sha: str):
    if file_sha256(path) != expected_sha:
        raise SystemExit(f"DEV panel hash differs: {path}")
    for line in path.read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            if case.get("split") != "dev":
                raise SystemExit(f"non-DEV record in {path}")
            ids = {case["id"], case.get("source_ref", {}).get("row_id") or case["id"]}
            prompt = render_prompt(PromptContext.from_mapping(case["context"]))
            for case_id in ids:
                yield case_id, prompt, case["target_body_text"], case["operation"]


def llama_tokenize(gguf: Path, text_path: Path, *, parse_special: bool, threads: int) -> list[int]:
    command = ["nice", "-n", "10", str(llama_bin("llama-tokenize")), "-m", str(gguf), "-f", str(text_path),
               "--ids", "--no-escape", "--log-disable", "-t", str(threads)]
    if not parse_special:
        command.append("--no-parse-special")
    done = subprocess.run(command, check=True, capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    line = next(line for line in reversed(done.stdout.splitlines()) if line.startswith("["))
    return json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-rows", type=Path, default=TRAIN_ROWS)
    ap.add_argument("--train-rows-sha256", default=TRAIN_ROWS_SHA256)
    ap.add_argument("--tokenizer-json", type=Path, default=TOKENIZER_JSON)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--budget-tokens", type=int, default=524_288,
                    help="calibration tokens before packing (default 524,288 = 128 chunks of 4,096)")
    ap.add_argument("--n-ctx", type=int, default=4096, help="llama-imatrix context (chunk) size")
    ap.add_argument("--kld-n-ctx", type=int, default=1024, help="llama-perplexity context for the KLD slice")
    ap.add_argument("--kld-chunks", type=int, default=64, help="held-out windows (one per KLD chunk)")
    ap.add_argument("--verify-gguf", type=Path, help="GGUF whose vocabulary llama-tokenize checks the streams with")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    for path in (args.out_dir, args.train_rows, args.tokenizer_json, DEV75, DEV250):
        refuse_sealed(path)
    if not 1 <= args.threads <= 8:
        ap.error("daytime CPU rule: 1 to 8 threads")
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        ap.error(f"output directory exists and is not empty: {out}")
    started = time.monotonic()
    if file_sha256(args.tokenizer_json) != TOKENIZER_JSON_SHA256:
        ap.error("tokenizer.json differs from the pinned tokenizer")
    decode, round_trips, added = load_tokenizer(args.tokenizer_json)

    train_sha = file_sha256(args.train_rows)
    if args.train_rows_sha256 and train_sha != args.train_rows_sha256:
        ap.error("TRAIN rows hash differs from the pinned materialization")
    rows = [json.loads(line) for line in args.train_rows.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["split"] != "train" or row["renderer_id"] != RENDERER_ID or row["tokenization_policy"] != TOKENIZATION_POLICY:
            ap.error(f"row {row['id']} is not a PRM03 TRAIN row")
    if len({row["id"] for row in rows}) != len(rows):
        ap.error("duplicate TRAIN row ids")

    dev = cal.build_dev_index([*dev_cases(DEV75, DEV75_SHA256), *dev_cases(DEV250, DEV250_SHA256)])
    frequency = cal.shingle_document_frequency(row["prompt_text"] for row in rows)
    excluded: dict[str, list[str]] = {}
    skipped: Counter = Counter()
    eligible = []
    for row in rows:
        reasons = cal.exclusion_reasons(row, dev, frequency)
        if reasons:
            excluded[row["id"]] = reasons
            continue
        text = row["prompt_text"] + row["target_text"]
        if cal.mentions_added_token(text, added):
            skipped["mentions_added_token"] += 1
            continue
        if decode(row["input_ids"][1:-1]) != text:
            skipped["decode_differs_from_row_text"] += 1
            continue
        eligible.append(row)
    rows_by_id = {row["id"]: row for row in eligible}

    fits = [row for row in eligible if len(row["input_ids"]) <= args.n_ctx]
    selected = cal.family_balanced(fits, args.budget_tokens, args.seed)
    selected_ids = {row["id"] for row in selected}
    order = cal.seeded_order(selected, args.seed, "calibration-order")
    filler_pool = cal.seeded_order([row for row in eligible if row["id"] not in selected_ids], args.seed, "filler")
    chunks, fillers = cal.pack_chunks(order, args.n_ctx, iter(filler_pool), round_trips)
    calibration_text, calibration_ids = cal.render_calibration(chunks, rows_by_id, decode)
    if len(calibration_ids) != len(chunks) * args.n_ctx or calibration_text.endswith("\n"):
        raise SystemExit("calibration stream geometry is wrong")

    used = selected_ids | set(fillers)
    kld_pool = [row for row in eligible if row["id"] not in used]
    windows = cal.kld_windows(kld_pool, args.kld_n_ctx, args.kld_chunks, args.seed, round_trips)
    kld_text, kld_ids = cal.render_kld(windows, rows_by_id, decode)
    if len(kld_ids) != args.kld_chunks * args.kld_n_ctx or kld_text.endswith("\n"):
        raise SystemExit("KLD stream geometry is wrong")
    if {w.row_id for w in windows} & used:
        raise SystemExit("KLD slice overlaps the calibration set")

    out.mkdir(parents=True, exist_ok=True)
    files = {"calibration.txt": calibration_text, "calibration.ids.json": json.dumps(calibration_ids, separators=(",", ":")),
             "kld-heldout.txt": kld_text, "kld-heldout.ids.json": json.dumps(kld_ids, separators=(",", ":"))}
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")

    verification = None
    if args.verify_gguf:
        refuse_sealed(args.verify_gguf)
        actual_cal = llama_tokenize(args.verify_gguf, out / "calibration.txt", parse_special=True, threads=args.threads)
        actual_kld = llama_tokenize(args.verify_gguf, out / "kld-heldout.txt", parse_special=False, threads=args.threads)
        verification = {
            "tool": str(llama_bin("llama-tokenize")), "llama_cpp": llama_version("llama-tokenize"),
            "vocab_gguf": str(args.verify_gguf),
            "calibration": {"flags": "--ids --no-escape (special parsing on, as llama-imatrix --parse-special); BOS added",
                            "tokens": len(actual_cal), "identical": actual_cal == calibration_ids,
                            "first_mismatch": cal.first_mismatch(actual_cal, calibration_ids)},
            "kld": {"flags": "--ids --no-escape --no-parse-special (as llama-perplexity); BOS added",
                    "tokens": len(actual_kld), "identical": actual_kld == kld_ids,
                    "first_mismatch": cal.first_mismatch(actual_kld, kld_ids)},
        }

    def family_stats(pieces):
        counter: Counter = Counter()
        tokens: Counter = Counter()
        for row_id, length in pieces:
            family = rows_by_id[row_id]["family"]
            counter[family] += 1
            tokens[family] += length
        return {"rows": dict(sorted(counter.items())), "tokens": dict(sorted(tokens.items()))}

    whole = [(p.row_id, p.end - p.start) for chunk in chunks for p in chunk if p.whole]
    filler_pieces = [(p.row_id, p.end - p.start) for chunk in chunks for p in chunk if not p.whole]
    reason_counts = Counter(reason for reasons in excluded.values() for reason in reasons)
    manifest = {
        "schema": SCHEMA, "card": "C09", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_set_access": False, "split": "train_only",
        "inputs": {
            "train_rows": {"path": str(args.train_rows), "sha256": train_sha, "rows": len(rows)},
            "dev75": {"path": str(DEV75), "sha256": DEV75_SHA256, "role": "exclusion only"},
            "dev250": {"path": str(DEV250), "sha256": DEV250_SHA256, "role": "exclusion only"},
            "tokenizer_json": {"path": str(args.tokenizer_json), "sha256": TOKENIZER_JSON_SHA256},
        },
        "seed": args.seed,
        "exclusion": {
            "dev_ids": len(dev.ids), "dev_keys": len(dev.keys),
            "rule": ("TRAIN row excluded if its id is a DEV id, or it shares a normalized prompt, current-region "
                     f"(>= {cal.MIN_KEY_CHARS} chars) or non-no-op target (>= {cal.MIN_KEY_CHARS} chars) sha256 "
                     f"with a DEV case, or >= {cal.SHINGLE_MIN_SHARED} and >= {cal.SHINGLE_CONTAINMENT:.0%} of the "
                     f"smaller side's {cal.SHINGLE_LINES}-line prompt shingles (shingles in more than "
                     f"{cal.SHINGLE_MAX_DOCUMENT_FREQUENCY} TRAIN rows ignored)"),
            "excluded_rows": len(excluded), "reasons": dict(sorted(reason_counts.items())),
            "excluded_row_ids": dict(sorted(excluded.items())),
            "skipped": dict(skipped), "eligible_rows": len(eligible),
        },
        "calibration": {
            "file": {"path": str(out / "calibration.txt"), "sha256": file_sha256(out / "calibration.txt"),
                     "bytes": (out / "calibration.txt").stat().st_size},
            "ids": {"path": str(out / "calibration.ids.json"), "sha256": file_sha256(out / "calibration.ids.json"),
                    "ids_sha256": cal.ids_sha256(calibration_ids)},
            "n_ctx": args.n_ctx, "chunks": len(chunks), "tokens": len(calibration_ids),
            "budget_tokens": args.budget_tokens,
            "whole_rows": len(whole), "whole_row_tokens": sum(n for _, n in whole), "by_family": family_stats(whole),
            "filler_prefixes": len(filler_pieces), "filler_tokens": sum(n for _, n in filler_pieces),
            "filler_by_family": family_stats(filler_pieces),
            "selected_row_ids": sorted(selected_ids), "filler_row_ids": fillers,
            "chunks_layout": [[[p.row_id, p.end - p.start, p.whole] for p in chunk] for chunk in chunks],
            "llama_imatrix_flags": ["--parse-special", "--no-escape", "-c", str(args.n_ctx), "-b", str(args.n_ctx),
                                    "--chunks", str(len(chunks))],
            "feeding": ("Text = rows detokenized with the pinned HF tokenizer; '<s>'/'</s>' mark each row's BOS 0/EOS 1 "
                        "and are parsed as control tokens by --parse-special; the first '<s>' is omitted because "
                        "llama.cpp adds BOS. Every chunk starts at a row BOS, so llama-imatrix's per-chunk BOS "
                        "overwrite is a no-op. Rows mentioning any added-token string are skipped. --no-escape keeps "
                        "backslash sequences in R code literal."),
        },
        "kld": {
            "file": {"path": str(out / "kld-heldout.txt"), "sha256": file_sha256(out / "kld-heldout.txt"),
                     "bytes": (out / "kld-heldout.txt").stat().st_size},
            "ids": {"path": str(out / "kld-heldout.ids.json"), "sha256": file_sha256(out / "kld-heldout.ids.json"),
                    "ids_sha256": cal.ids_sha256(kld_ids)},
            "n_ctx": args.kld_n_ctx, "chunks": len(windows), "tokens": len(kld_ids),
            "scored_tokens": len(windows) * (args.kld_n_ctx - 1 - args.kld_n_ctx // 2),
            "by_family": family_stats([(w.row_id, w.end - w.start) for w in windows]),
            "windows": [[w.row_id, w.start, w.end] for w in windows],
            "disjoint_from_calibration": True,
            "llama_perplexity_flags": ["--no-escape", "-c", str(args.kld_n_ctx), "-b", str(args.kld_n_ctx)],
            "feeding": ("One window per chunk: the last n_ctx-1 tokens of a held-out row before its EOS (ends with the "
                        "gold target and terminal line). Chunk 0 relies on llama.cpp's added BOS; later chunks start "
                        "with '<think>' (user-defined id 8, matched without special parsing), which "
                        "llama-perplexity overwrites with BOS. llama-perplexity scores the second half of each chunk."),
        },
        "verification": verification,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {k: manifest["calibration"][k] for k in ("chunks", "tokens", "whole_rows", "filler_prefixes", "filler_tokens")}
    print(json.dumps({"out_dir": str(out), "calibration": summary, "calibration_sha256": manifest["calibration"]["file"]["sha256"],
                      "kld_chunks": len(windows), "kld_sha256": manifest["kld"]["file"]["sha256"],
                      "excluded_rows": len(excluded), "reasons": manifest["exclusion"]["reasons"],
                      "verification": None if verification is None else
                      {k: verification[k]["identical"] for k in ("calibration", "kld")}}, indent=2))
    if verification and not (verification["calibration"]["identical"] and verification["kld"]["identical"]):
        print("llama.cpp tokenization differs from the expected stream", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
