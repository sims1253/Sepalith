#!/usr/bin/env python3
"""Read-only proof for the pinned serving fixtures and source snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

OWNER = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-e6aed5ff")
SNAPSHOT = Path("/mnt/e/sepalith/campaign-20260915/source-snapshots/20260912T005156Z/owner")
EVAL = OWNER / "experiments" / "eval"
EXTENSION = OWNER / "extensions/vscode-sepalith/src/extension.ts"

SOURCE_HASHES = {
    "extensions/vscode-sepalith/src/extension.ts":
        "368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285",
    "experiments/eval/eval_noop_fp.py":
        "947d5dc6280bf85b831012116541f067f1104c46a682b3e3dc7ef722893f637d",
    "experiments/eval/prediction_parser.py":
        "421ba755f26912a0e931e2363d2264e4748208511816a1478c0241f7fb0e6467",
    "experiments/eval/run_eval.py":
        "7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d",
    "experiments/eval/eval_scenarios.py":
        "da81802f553e91182ca7dbece30614ad5b12479f3f4e3f5a856a16dcd23fefc3",
    "experiments/post-processing/assemble_sft_v2.py":
        "dd519c61f0684040251f5f886c6f6307a6d0a99329fc7ac5792974a23cd43f12",
    "experiments/synthetic-data/scenarios.py":
        "cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ts_parse(text: str) -> list[str]:
    """The inline extension.parsePrediction algorithm, copied as a fixture."""
    marker_line = re.compile(
        r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|"
        r"suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$"
    )
    if ">>>>>>>" in text:
        text = text.split(">>>>>>>")[0]
    text = text.replace("<|user_cursor|>", "")
    lines = [line.rstrip("\r") for line in text.split("\n")]
    lines = [line for line in lines if not marker_line.match(line)]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    for index in range(2, len(lines)):
        if lines[index] == lines[index - 1] == lines[index - 2]:
            lines = lines[:index]
            break
    return lines


def main() -> None:
    for relative, expected in SOURCE_HASHES.items():
        active = OWNER / relative
        frozen = SNAPSHOT / relative
        assert sha256(active) == expected, f"active source hash changed: {active}"
        assert sha256(frozen) == expected, f"snapshot source hash changed: {frozen}"

    stops_match = re.search(r"const STOPS = (\[[^;]+\]);", EXTENSION.read_text())
    assert stops_match, "extension STOPS declaration not found"
    expected_stops = [
        ">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>",
        "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>",
    ]
    assert json.loads(stops_match.group(1)) == expected_stops

    sys.path.insert(0, str(EVAL))
    from prediction_parser import parse_prediction
    from run_eval import parse_pred, render_zeta2

    newline = chr(10)
    parser_fixtures = [
        (
            "markers_cursor_crlf",
            chr(13) + newline + newline.join([
                "<<<<<<< CURRENT", "  x <- 1", "<|user_cursor|>", "=======",
                "<[fim-middle]>", ">>>>>>> UPDATED", "ignored",
            ]),
            ["  x <- 1"],
        ),
        (
            "marker_echo_and_repetition",
            newline.join(["a", "<|outline|>", "b", "b", "b", "c"]),
            ["a", "b", "b"],
        ),
        (
            "blank_line_trim",
            newline.join(["", "", "  out <- 1", "", ""]),
            ["  out <- 1"],
        ),
    ]
    for name, raw, expected in parser_fixtures:
        assert ts_parse(raw) == expected, f"TypeScript fixture failed: {name}"
        assert parse_prediction(raw) == expected, f"helper fixture failed: {name}"

    example = {
        "suffix": ["tail()"],
        "event_diff": chr(96) * 3 + "diff" + newline + "User edited docs.R"
        + newline + "+new <- 1" + newline + chr(96) * 3,
        "path": "pkg/file.R",
        "prefix": ["old <- 0"],
        "region_old": ["cursor old"],
        "cursor_idx": 0,
    }
    expected_render = newline.join([
        "<[fim-suffix]>", "tail()", "<[fim-prefix]><filename>edit_history",
        "+new <- 1", "", "<filename>pkg/file.R", "old <- 0",
        "<<<<<<< CURRENT", "cursor old<|user_cursor|>", "=======",
        "<[fim-middle]>",
    ])
    assert render_zeta2(example) == expected_render
    assert parse_pred("zeta2", "  answer  " + newline
                      + ">>>>>>> UPDATED" + newline + "ignored") == ["  answer"]

    print(json.dumps({
        "source_snapshot": str(SNAPSHOT),
        "source_hashes": SOURCE_HASHES,
        "stop_contract": "PASS",
        "parser_fixtures": [name for name, _, _ in parser_fixtures],
        "renderer_fixture": "PASS",
        "server_started": False,
        "cuda_imported": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
