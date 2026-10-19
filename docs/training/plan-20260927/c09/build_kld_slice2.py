#!/usr/bin/env python3
"""C09 search: build a SECOND held-out KLD slice from TRAIN rows (robustness check).

The 2026-10-17/18 quant-autoresearch session ranked every recipe on
``calibration-v1/kld-heldout.txt``.  This tool builds a disjoint slice with the
same machinery (``sepalith.quant.calibration.kld_windows``/``render_kld``) so
the frontier ranking can be re-checked on text that played no role in any
search decision — guarding against slice-specific overfitting of the *search*,
not the gate (the gate keeps its own slice).

Exclusions (TRAIN-only, no DEV reads): the v1 manifest's excluded row, every
calibration row (imatrix, including fillers), every v1 KLD window row
(both read from the v1 ``manifest.json``) and the search panel's row ids.

    uv run python docs/training/plan-20260927/c09/build_kld_slice2.py \\
        --out-dir /mnt/e/.../autoresearch-20261017/kld2 \\
        --chunks 32 --verify-gguf <any arm-a GGUF>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages" / "sepalith" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sepalith.quant import calibration as cal
from sepalith.quant.llama_tools import file_sha256
from sepalith.quant.search_panel import manifest_row_ids
from build_calibration import TRAIN_ROWS, TRAIN_ROWS_SHA256, TOKENIZER_JSON, TOKENIZER_JSON_SHA256, llama_tokenize, load_tokenizer

V1_DIR = Path("/mnt/e/sepalith/campaign-20260915/quant-work/calibration-v1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--train-rows", type=Path, default=TRAIN_ROWS)
    ap.add_argument("--panel-ids-from", type=Path,
                    default=Path("/mnt/e/sepalith/campaign-20260915/quant-work/autoresearch-20261017/search-panel.json"))
    ap.add_argument("--verify-gguf", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    if file_sha256(args.train_rows) != TRAIN_ROWS_SHA256:
        raise SystemExit("TRAIN rows file differs from the pinned sha256")
    if file_sha256(TOKENIZER_JSON) != TOKENIZER_JSON_SHA256:
        raise SystemExit("tokenizer.json differs from the pinned tokenizer")

    reserved = manifest_row_ids(V1_DIR / "manifest.json")
    excluded = reserved["dev_excluded"]
    used = reserved["calibration"] | reserved["kld"]
    if args.panel_ids_from.is_file():
        used |= {case["id"] for case in json.loads(args.panel_ids_from.read_text())["cases"]}
    decode, round_trips, added = load_tokenizer(TOKENIZER_JSON)

    rows = [json.loads(line) for line in args.train_rows.read_text().splitlines() if line.strip()]
    pool = [row for row in rows if str(row["id"]) not in excluded | used
            and not cal.mentions_added_token(row["prompt_text"], added)]
    rows_by_id = {str(row["id"]): row for row in rows}
    windows = cal.kld_windows(pool, args.n_ctx, args.chunks, args.seed, round_trips)
    text, ids = cal.render_kld(windows, rows_by_id, decode)
    if len(ids) != args.chunks * args.n_ctx or text.endswith("\n"):
        raise SystemExit("rendered slice is malformed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "kld-slice2.txt").write_text(text)
    (args.out_dir / "kld-slice2.ids.json").write_text(json.dumps(ids, separators=(",", ":")))
    record = {"schema": "sepalith.c09.quant-kld-slice2.v1", "seed": args.seed, "chunks": args.chunks, "n_ctx": args.n_ctx,
              "window_rows": [w.row_id for w in windows], "excluded": {"v1_manifest": len(excluded), "used": len(used)},
              "text_sha256": cal.sha256_text(text), "ids_sha256": cal.ids_sha256(ids),
              "train_rows_sha256": TRAIN_ROWS_SHA256}
    if args.verify_gguf:
        actual = llama_tokenize(args.verify_gguf, args.out_dir / "kld-slice2.txt",
                                parse_special=False, threads=args.threads)
        mismatch = cal.first_mismatch(actual, ids)
        record["llama_tokenized_identical"] = mismatch is None
        record["llama_tokenized_first_mismatch"] = mismatch
    (args.out_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("chunks", "n_ctx", "seed") } | {"ok": record.get("llama_tokenized_identical")}))
    return 0 if record.get("llama_tokenized_identical", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
