#!/usr/bin/env python3
"""Cross-check an e2e-cpu report in Python (TRAIN rows only).

For every case in ``npm run e2e-cpu -- --out FILE``: decode the server's
generated IDs with the pinned HF tokenizer (without the final end-of-generation
ID), require the text to equal the server content the extension parsed, and
re-classify it with ``campaign_protocol.parse_output`` and
``valid_generation_tokens`` (the gate's rule). Run from the repository root:

    .venv/bin/python extensions/vscode-sepalith/tests/e2e_crosscheck.py /tmp/e2e.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))
from sepalith.campaign_protocol import PromptContext, parse_output, valid_generation_tokens  # noqa: E402

SIDECAR = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/context-sidecar.jsonl")
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536/tokenizer.json")


def main() -> int:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    report = json.loads(Path(sys.argv[1]).read_text())
    wanted = {case["id"] for case in report["cases"]}
    contexts = {}
    with SIDECAR.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["row_id"] in wanted:
                contexts[row["row_id"]] = PromptContext.from_mapping(row["context"])
    agree = 0
    for case in report["cases"]:
        ids = case["tokens"]
        complete = bool(ids) and ids[-1] in (1, 130073)
        text = tokenizer.decode(ids[:-1] if complete else ids, skip_special_tokens=False)
        result = parse_output(text, contexts[case["id"]])
        valid = valid_generation_tokens(ids) and result.status == "accepted"
        python = "invalid" if not valid else ("no_edit" if result.operation == "no_op" else "edit")
        same = text == case["content"] and python == case["outcome"]
        agree += same
        print(case["id"], case.get("check", "case"), f"decode==content={text == case['content']}", f"python={python}", f"ts={case['outcome']}")
    print(f"cross-check {agree}/{len(report['cases'])}")
    return 0 if agree == len(report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
