#!/usr/bin/env python3
"""CPU-only tokenizer, geometry, and boundary checks for RUN-05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
MAX_OUTPUT = 192
VOCAB_SIZE = 130560
CONTROL_RANGES = ((0, 7), (10, 21), (130072, VOCAB_SIZE - 1))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def has_control(token: int) -> bool:
    return any(lo <= token <= hi for lo, hi in CONTROL_RANGES)


def check(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    fixture = json.loads(raw)
    assert fixture["schema"] == "sepalith.serving.context-stress-fixture.v1"
    assert fixture["task"] == "RUN-05"
    assert fixture["syntheticOnly"] is True
    assert fixture["selectorPolicy"]["productionDefaultBudgetUtf16Units"] == 6000
    assert fixture["selectorPolicy"]["stressOnly"] is True

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    lines = fixture["document"]["text"].splitlines()
    assert sha256_bytes(fixture["document"]["text"].encode("utf-8")) == fixture["document"]["contentSha256"]
    events = fixture["events"]
    assert len(events) == 6
    results: list[dict[str, object]] = []
    for case in fixture["cases"]:
        pair = [event for event in events if event["caseId"] == case["id"]]
        assert len(pair) == 2
        baseline, repeat = pair
        assert baseline["kind"] == "baseline"
        assert repeat["kind"] == "unchanged_repeat_control"
        assert baseline["after"]["promptText"] == repeat["after"]["promptText"]
        prompt = baseline["after"]["promptText"]
        prompt_bytes = prompt.encode("utf-8")
        ids = tokenizer.encode(prompt, add_special_tokens=False, split_special_tokens=True)
        assert all(isinstance(token, int) and 0 <= token < VOCAB_SIZE for token in ids)
        assert not any(has_control(token) for token in ids)
        selection = baseline["after"]["selection"]
        selected = selection["prefix"] + selection["region"] + selection["suffix"]
        assert selection["usedUtf16Units"] <= selection["budgetUtf16Units"]
        assert selection["requiredOverflow"] is False
        assert all(line in lines for line in selected)
        span_lines: list[str] = []
        for span in selection["spans"]:
            span_lines.extend(lines[span["startLine"] : span["endLine"] + 1])
        assert span_lines == selected
        assert prompt.endswith("<[fim-middle]>\n")
        bos_prompt_tokens = 1 + len(ids)
        total_with_output = bos_prompt_tokens + MAX_OUTPUT
        expected_context = int(case["contextSize"])
        expected_overflow = total_with_output > expected_context
        profile_overflow = []
        for size, name in ((2048, "native_diagnostic_2048"), (4096, "primary_editor_4096"), (8192, "stress_only_8192")):
            if total_with_output > size:
                profile_overflow.append(name)
        assert profile_overflow == list(case["expectedOverflowProfiles"])
        if case["id"] == "near-2k":
            assert 1700 <= len(ids) <= 1855 and not expected_overflow
        elif case["id"] == "near-4k":
            assert 3000 <= len(ids) <= 3904 and not expected_overflow and total_with_output > 2048
        elif case["id"] == "near-8k":
            assert 6000 <= len(ids) <= 8000 and not expected_overflow and total_with_output > 4096
        else:
            raise AssertionError(f"unexpected case {case['id']}")
        results.append({
            "caseId": case["id"],
            "tokenTarget": case["tokenTarget"],
            "selectorBudgetUtf16Units": selection["budgetUtf16Units"],
            "selectedSourceUtf16Units": selection["usedUtf16Units"],
            "selectedSourceLineCount": len(selected),
            "promptUtf16Units": len(prompt),
            "promptUtf8Bytes": len(prompt_bytes),
            "promptSha256": baseline["after"]["promptSha256"],
            "promptUtf8Sha256": sha256_bytes(prompt_bytes),
            "promptTokenCountWithoutBos": len(ids),
            "promptTokenCountWithManualBos": bos_prompt_tokens,
            "promptTokenIdsSha256": sha256_json(ids),
            "totalWithOutput192": total_with_output,
            "contextSize": expected_context,
            "expectedOverflowProfiles": profile_overflow,
            "repeatExactPrompt": repeat["after"]["promptSha256"] == baseline["after"]["promptSha256"],
        })
    return {
        "schema": "sepalith.serving.context-stress-fixture.token-report.v1",
        "fixtureSha256": sha256_bytes(raw),
        "fixtureBytes": len(raw),
        "documentSha256": fixture["document"]["contentSha256"],
        "documentUtf8Bytes": len(fixture["document"]["text"].encode("utf-8")),
        "documentLineCount": len(lines),
        "tokenizer": {
            "path": str(TOKENIZER_DIR),
            "revision": fixture["source"]["tokenizerRevision"],
            "tokenizerJsonSha256": fixture["source"]["tokenizerJsonSha256"],
            "tokenizerConfigSha256": fixture["source"]["tokenizerConfigSha256"],
            "addSpecialTokens": False,
            "splitSpecialTokens": True,
            "manualBosId": 0,
            "canonicalEosId": 1,
        },
        "events": len(events),
        "cases": len(results),
        "results": results,
        "checks": [
            "pinned tokenizer local-only encode",
            "manual BOS counted exactly once outside tokenizer",
            "no native control IDs in prompt tokenization",
            "complete source lines only in selected spans",
            "UTF-16 selector budget and renderer boundary",
            "2K/4K/8K context plus 192-output overflow controls",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("context-stress-fixture.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check(args.fixture)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
