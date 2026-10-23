#!/usr/bin/env python3
"""Build the PRM03 golden fixtures for the TypeScript port (TRAIN data only).

Every case comes from arm A's TRAIN inputs: the DAT10 v4 context sidecar and
token rows (the rows ``reward_v21.TRAIN_INPUTS`` and the SFT-v2 recipe pin) and
the C12 census samples (arm A outputs on those TRAIN prompts). DEV panels, the
sealed final set and pre-split pools are never read.

Outputs (``--out``, default next to this script):

* ``prm03-golden.json`` (committed): seeded, variant-stratified render cases
  with the Python ``render_prompt`` text, parse cases (TRAIN targets, census
  outputs, hand-written edge cases over TRAIN contexts) with the Python
  ``parse_output``/``valid_generation_tokens`` results, source-window cases
  with the Python ``select_source_window`` results, and history-diff cases.
* ``--census-all PATH`` (local only, large): Python parse results for every
  census sample, for ``npm run check-prm03-train``.

Run from the repository root:

    .venv/bin/python extensions/vscode-sepalith/tests/fixtures/build_prm03_fixtures.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))

from sepalith.campaign_protocol import (  # noqa: E402
    PromptContext, parse_output, render_prompt, valid_generation_tokens,
)
from sepalith.campaign_selection import select_source_window  # noqa: E402

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent")
SIDECAR = DATA / "context-sidecar.jsonl"
SIDECAR_SHA256 = "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a"
ROWS = DATA / "train-token-rows.jsonl"
ROWS_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
CENSUS = Path.home() / ".local/state/sepalith/c12/census-arm-a/samples.jsonl"
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536/tokenizer.json")
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
SEED = 20261023
FORBIDDEN = ("DAT-07-final", "sealed-final", "DAT-08", "spec_traces", "dev250", "dev75")


def guard(path: Path) -> Path:
    if any(part in str(path) for part in FORBIDDEN):
        raise SystemExit(f"refusing non-TRAIN input: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with guard(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def variant(ctx: dict) -> str:
    region = ctx["region_old"]
    if not region:
        shape = "empty"
    elif len(region) == 1:
        column = ctx["cursor"]["code_point_column"]
        shape = "one:" + ("col0" if column == 0 else "end" if column == len(region[0]) else "mid")
    else:
        shape = "multi"
    return f"{shape}|hist{len(ctx['history'])}|{ctx['document_eol']}"


def parse_record(text: str, context: PromptContext) -> dict:
    result = parse_output(text, context)
    return {"status": result.status, "operation": result.operation, "body": list(result.body),
            "body_text": result.body_text, "reason": result.reason}


def render_case(case_id: str, name: str, ctx: dict, prompt: str, used: dict[str, dict]) -> dict:
    used[case_id] = ctx
    return {"id": case_id, "variant": name, "context_id": case_id, "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt)}


def derived_variants(ctx: dict, rng: random.Random) -> list[tuple[str, dict]]:
    """Renderer branches arm A never saw in TRAIN, built from one TRAIN context."""
    out = []
    a = json.loads(json.dumps(ctx))
    a["diagnostics"] = [{"content": "warning: no visible binding for global variable 'x'", "path": a["path"]},
                        {"content": "style: line too long", "path": None}]
    a["selected_references"] = [{"content": "helper <- function(x) x + 1", "path": "R/helper.R"}]
    a["retrieval"] = [{"content": "", "path": None}]
    out.append(("derived:evidence", a))
    b = json.loads(json.dumps(ctx))
    b["scope_mode"] = "outline"
    b["scope_lines"] = ["<|outline|>", "12 f", "40 g"]
    out.append(("derived:scope_outline", b))
    c = json.loads(json.dumps(ctx))
    if c["region_old"]:
        # astral-plane text before the cursor: code-point and UTF-16 columns differ
        line = "  \U0001d465 <- \"é\U0001f600\" " + c["region_old"][0]
        c["region_old"] = [line] + c["region_old"][1:]
        column = 9
        c["cursor"] = {"region_line_index": 0, "code_point_column": column,
                       "utf16_column": sum(2 if ord(ch) > 0xFFFF else 1 for ch in line[:column])}
        rng_ = c["replacement_range"]
        width = sum(2 if ord(ch) > 0xFFFF else 1 for ch in c["region_old"][-1])
        if len(c["region_old"]) == 1:
            rng_["end"] = {"line": rng_["start"]["line"], "character": rng_["start"]["character"] + width}
        out.append(("derived:astral_cursor", c))
    return out


def edge_outputs(ctx: dict) -> list[str]:
    region = ctx["region_old"]
    body = "\n".join(region) if region else "x <- 1"
    return [
        "", ">>>>>>> UPDATED", "\n>>>>>>> UPDATED", "\n\n>>>>>>> UPDATED", "[NO_EDIT]\n>>>>>>> UPDATED",
        "[NO_EDIT]\r\n>>>>>>> UPDATED\r\n", " [NO_EDIT] \n>>>>>>> UPDATED", "[NO_EDIT]\nx\n>>>>>>> UPDATED",
        f"{body}\n>>>>>>> UPDATED", f"{body}\n>>>>>>> UPDATED\n", f"{body}\n>>>>>>> UPDATED\n\n",
        f"{body}\n>>>>>>> UPDATED\nmore", f"{body}\n>>>>>>> UPDATED \n", f"{body}\n >>>>>>> UPDATED",
        f"{body}\n>>>>>>> UPDATED\n>>>>>>> UPDATED", f"{body}\r\n>>>>>>> UPDATED", f"{body}\r\r\n>>>>>>> UPDATED",
        f"\n{body}\n\n>>>>>>> UPDATED", f"{body}  \n   \n>>>>>>> UPDATED",
        "  =======  \n>>>>>>> UPDATED", " =======\n>>>>>>> UPDATED", "\x1c<|user_cursor|>\x1f\n>>>>>>> UPDATED",
        "﻿=======\n>>>>>>> UPDATED", "​=======\n>>>>>>> UPDATED", "x <- '=======' \n>>>>>>> UPDATED",
        "<<<<<<< CURRENT\n>>>>>>> UPDATED", "<|outline|>\n>>>>>>> UPDATED", "<[fim-middle]>\n>>>>>>> UPDATED",
        "y <- '<unused_token_477>'\n>>>>>>> UPDATED", "y <- '<unused_token_478>'\n>>>>>>> UPDATED",
        "y <- '<unused_token_012>'\n>>>>>>> UPDATED", "\U0001f600 <- 1\n>>>>>>> UPDATED",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "prm03-golden.json")
    parser.add_argument("--per-variant", type=int, default=5)
    parser.add_argument("--census-sample", type=int, default=150)
    parser.add_argument("--census-all", type=Path, help="also write Python parse results for every census sample (JSONL)")
    args = parser.parse_args()
    for path, expected in ((SIDECAR, SIDECAR_SHA256), (ROWS, ROWS_SHA256), (TOKENIZER, TOKENIZER_SHA256)):
        if sha256_file(path) != expected:
            raise SystemExit(f"hash differs: {path}")
    rng = random.Random(SEED)

    contexts: dict[str, dict] = {}
    rows: dict[str, dict] = {}
    strata: dict[str, list[str]] = collections.defaultdict(list)
    with SIDECAR.open() as sidecar, ROWS.open() as row_file:
        for side_line, row_line in zip(sidecar, row_file, strict=True):
            side, row = json.loads(side_line), json.loads(row_line)
            if side["row_id"] != row["id"] or side["split"] != "train" or row["split"] != "train":
                raise SystemExit("sidecar and rows disagree or a non-TRAIN row appeared")
            rid = row["id"]
            contexts[rid] = side["context"]
            rows[rid] = {"family": row["family"], "prompt_text": row["prompt_text"], "target_text": row["target_text"],
                         "target_operation": row["target_operation"]}
            strata[f"{row['family']}|{variant(side['context'])}"].append(rid)

    chosen: list[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        chosen.extend(ids[: args.per_variant])

    render_cases, parse_cases = [], []
    used: dict[str, dict] = {}
    for rid in chosen:
        ctx = contexts[rid]
        prompt = render_prompt(PromptContext.from_mapping(ctx))
        if prompt != rows[rid]["prompt_text"]:
            raise SystemExit(f"{rid}: Python render differs from the TRAIN row")
        render_cases.append(render_case(rid, f"{rows[rid]['family']}|{variant(ctx)}", ctx, prompt, used))
        context = PromptContext.from_mapping(ctx)
        parse_cases.append({"id": f"{rid}:target", "context_id": rid, "text": rows[rid]["target_text"],
                            "expected": parse_record(rows[rid]["target_text"], context)})
    edge_ids = [rid for rid in chosen if contexts[rid]["region_old"]][:4] + [rid for rid in chosen if not contexts[rid]["region_old"]][:2]
    for rid in edge_ids:
        context = PromptContext.from_mapping(contexts[rid])
        for index, text in enumerate(edge_outputs(contexts[rid])):
            parse_cases.append({"id": f"{rid}:edge{index}", "context_id": rid, "text": text,
                                "expected": parse_record(text, context)})
    for rid in chosen[:: max(1, len(chosen) // 12)]:
        for name, ctx in derived_variants(contexts[rid], rng):
            prompt = render_prompt(PromptContext.from_mapping(ctx))
            render_cases.append(render_case(f"{rid}:{name}", name, ctx, prompt, used))

    # census outputs (arm A samples on TRAIN prompts), stratified by outcome
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    pools: dict[str, list[dict]] = collections.defaultdict(list)
    census_out = guard(args.census_all).open("w") if args.census_all else None
    census_total = 0
    with guard(CENSUS).open() as handle:
        for line in handle:
            sample = json.loads(line)
            rid = sample["id"]
            ids = sample["generated_ids"]
            complete = bool(ids) and ids[-1] in (1, 130073)
            text = tokenizer.decode(ids[:-1] if complete else ids, skip_special_tokens=False)
            context = PromptContext.from_mapping(contexts[rid])
            expected = parse_record(text, context)
            record = {"id": f"{rid}:s{sample['sample']}", "context_id": rid, "text": text, "tokens": ids,
                      "expected": expected, "valid_tokens": valid_generation_tokens(ids)}
            census_total += 1
            if census_out:
                census_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            key = (f"{expected['status']}:{expected['operation'] or expected['reason']}:"
                   f"{'ok' if record['valid_tokens'] else 'badtok'}")
            pool = pools[key]
            if len(pool) < 4 * args.census_sample:
                pool.append(record)
    if census_out:
        census_out.close()
    census_cases = []
    per_pool = max(1, args.census_sample // max(1, len(pools)))
    for key in sorted(pools):
        pool = pools[key]
        rng.shuffle(pool)
        census_cases.extend(pool[:per_pool])
    for record in census_cases:
        if record["context_id"] not in used:
            ctx = contexts[record["context_id"]]
            prompt = render_prompt(PromptContext.from_mapping(ctx))
            render_cases.append(render_case(record["context_id"], f"census|{variant(ctx)}", ctx, prompt, used))

    # source windows over documents rebuilt from TRAIN contexts
    selection_cases = []
    pool_ids = sorted(contexts)
    rng.shuffle(pool_ids)
    for index, rid in enumerate(pool_ids[:24]):
        ctx = contexts[rid]
        extra = contexts[pool_ids[(index + 1) % len(pool_ids)]]
        region = ctx["region_old"] or [""]
        lines = extra["prefix"] + ctx["prefix"] + region + ctx["suffix_lines"] + extra["suffix_lines"]
        start = len(extra["prefix"]) + len(ctx["prefix"])
        end = start + len(region) - 1
        results = []
        for budget in (6000, 1200, 64, 0):
            view = select_source_window(lines=lines, region_start_line=start, region_end_line=end,
                                        max_utf16_units=budget).to_dict()
            results.append({"budget": budget,
                            "expected": {k: view[k] for k in ("budget_utf16_units", "used_utf16_units",
                                                              "required_utf16_units", "overflow", "required_overflow",
                                                              "prefix", "region", "suffix")},
                            "omissions": [{"side": o["side"], "start_line": o["start_line"], "end_line": o["end_line"]}
                                          for o in view["omissions"]]})
        selection_cases.append({"id": rid, "lines": lines, "region_start_line": start, "region_end_line": end,
                                "results": results})
    # the TRAIN window reproduces itself: no budget pressure inside a selected window
    for rid in chosen:
        ctx = contexts[rid]
        region = ctx["region_old"] or [""]
        lines = ctx["prefix"] + region + ctx["suffix_lines"]
        start = len(ctx["prefix"])
        result = select_source_window(lines=lines, region_start_line=start, region_end_line=start + len(region) - 1)
        if list(result.prefix) != ctx["prefix"] or list(result.suffix) != ctx["suffix_lines"]:
            raise SystemExit(f"{rid}: TRAIN window does not reproduce under select_source_window")

    history_cases = []
    for rid in chosen:
        for event in contexts[rid]["history"]:
            history_cases.append({"id": event["event_id"], "path": event["path"],
                                  "start": event["range_utf16"]["start"]["line"],
                                  "old": event["old_text"].split("\n"), "new": event["new_text"].split("\n"),
                                  "event_diff": event["event_diff"]})

    fixture = {
        "schema": "sepalith.prm03-golden.v1",
        "generator": "extensions/vscode-sepalith/tests/fixtures/build_prm03_fixtures.py",
        "seed": SEED,
        "renderer_id": "zeta2-prm03-v1",
        "python_sources_sha256": {
            name: sha256_file(REPO / "packages/sepalith/src/sepalith" / name)
            for name in ("campaign_protocol.py", "campaign_selection.py")
        },
        "train_inputs": {"sidecar": {"path": str(SIDECAR), "sha256": SIDECAR_SHA256},
                         "rows": {"path": str(ROWS), "sha256": ROWS_SHA256},
                         "census_samples": {"path": str(CENSUS), "samples_read": census_total},
                         "tokenizer_json_sha256": TOKENIZER_SHA256},
        "data_rule": "TRAIN only: DAT10 v4 TRAIN sidecar/rows and C12 census samples on TRAIN prompts; no DEV, no sealed final set",
        "counts": {"render": len(render_cases), "parse": len(parse_cases) + len(census_cases),
                   "census": len(census_cases), "selection": len(selection_cases), "history": len(history_cases),
                   "variants": len({c["variant"] for c in render_cases})},
        "contexts": used,
        "render": render_cases,
        "parse": parse_cases + census_cases,
        "selection": selection_cases,
        "history": history_cases,
    }
    args.out.write_text(json.dumps(fixture, ensure_ascii=False, indent=0, sort_keys=True) + "\n")
    print(json.dumps(fixture["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
