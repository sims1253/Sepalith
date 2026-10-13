#!/usr/bin/env python3
"""CPU-only tokenizer and source-geometry audit for the RUN-05 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TOKENIZER_DIR = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
INPUT = Path(__file__).resolve().parents[1] / "serving-transition-long-panel" / "long-transition-fixture.json"
CONTROL_RANGES = ((0, 7), (10, 21), (130072, 130559))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_digest(tokens: list[int]) -> str:
    return digest(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))


def has_control(token: int) -> bool:
    return any(lo <= token <= hi for lo, hi in CONTROL_RANGES)


def check(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    fixture = json.loads(raw)
    original = json.loads(INPUT.read_text(encoding="utf-8"))
    assert fixture["schema"] == "sepalith.serving.long-context-budget-candidates.v1"
    assert fixture["task"] == "RUN-05"
    assert fixture["candidateBudgetsUtf16Units"] == [3000, 1500]
    assert fixture["nativeProfile"]["contextSize"] == 4096
    assert fixture["nativeProfile"]["maxOutputTokens"] == 192
    assert fixture["nativeProfile"]["launchesServer"] is False
    assert len(fixture["events"]) == 12
    assert len(original["events"]) == 6

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    original_by_id = {event["eventId"]: event for event in original["events"]}
    results: list[dict[str, object]] = []
    for event in fixture["events"]:
        original_event = original_by_id[event["originalEventId"]]
        budget = event["sourceBudgetUtf16Units"]
        assert budget in (3000, 1500)
        assert event["afterDocument"] == original_event["afterDocument"]
        assert event["contentChanges"] == original_event["contentChanges"]
        assert event["cursorAfter"] == original_event["cursorAfter"]
        assert event["scope"] == original_event["scope"]
        assert event["after"]["context"]["replacement_range"] == original_event["after"]["context"]["replacement_range"]
        assert event["after"]["context"]["cursor"] == original_event["after"]["context"]["cursor"]
        assert event["after"]["context"]["region_old"] == original_event["after"]["context"]["region_old"]
        for field in ("history", "diagnostics", "selected_references", "retrieval", "scope_mode", "scope_lines", "path", "document_eol"):
            assert event["after"]["context"][field] == original_event["after"]["context"][field]

        # JS production geometry uses text.split("\\n"), retaining the final
        # empty segment after the fixture's terminal LF.
        document_lines = event["afterDocument"]["text"].split("\n")
        selection = event["after"]["selection"]
        assert selection["budgetUtf16Units"] == budget
        assert selection["requiredOverflow"] is False
        assert selection["usedUtf16Units"] <= budget
        selected_lines = selection["prefix"] + selection["region"] + selection["suffix"]
        assert all(line in document_lines for line in selected_lines)
        span_lines: list[str] = []
        for span in selection["spans"]:
            span_lines.extend(document_lines[span["startLine"] : span["endLine"] + 1])
        assert span_lines == selected_lines

        prompt = event["after"]["promptText"]
        prompt_bytes = prompt.encode("utf-8")
        assert digest(prompt_bytes) == event["after"]["promptSha256"]
        assert prompt.endswith("<[fim-middle]>\n")
        tokens = tokenizer.encode(prompt, add_special_tokens=False, split_special_tokens=True)
        assert all(isinstance(token, int) and 0 <= token < 130560 for token in tokens)
        assert not any(has_control(token) for token in tokens)
        with_bos = len(tokens) + 1
        assert with_bos + 192 <= 4096
        results.append({
            "eventId": event["eventId"],
            "originalEventId": event["originalEventId"],
            "kind": event["kind"],
            "sourceBudgetUtf16Units": budget,
            "selectedSourceUtf16Units": selection["usedUtf16Units"],
            "requiredSourceUtf16Units": selection["requiredUtf16Units"],
            "selectedSourceLineCount": len(selected_lines),
            "promptUtf16Units": len(prompt),
            "promptUtf8Bytes": len(prompt_bytes),
            "promptSha256": event["after"]["promptSha256"],
            "promptTokenCountWithoutBos": len(tokens),
            "promptTokenCountWithManualBos": with_bos,
            "totalWithOutput192": with_bos + 192,
            "promptTokenIdsSha256": token_digest(tokens),
            "replacementRangePreserved": event["after"]["context"]["replacement_range"] == original_event["after"]["context"]["replacement_range"],
            "cursorPreserved": event["after"]["context"]["cursor"] == original_event["after"]["context"]["cursor"],
            "oldRegionPreserved": event["after"]["context"]["region_old"] == original_event["after"]["context"]["region_old"],
        })
    assert {item["originalEventId"] for item in results} == set(original_by_id)
    assert all(sum(1 for item in results if item["originalEventId"] == event_id) == 2 for event_id in original_by_id)
    return {
        "schema": "sepalith.serving.long-context-budget-candidates.token-report.v1",
        "fixtureSha256": digest(raw),
        "fixtureBytes": len(raw),
        "events": len(results),
        "originalEventCount": len(original["events"]),
        "candidateBudgetsUtf16Units": [3000, 1500],
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
        "results": results,
        "checks": [
            "exact production selector source budgets 3000 and 1500 UTF-16",
            "complete source spans and required-scope fit",
            "replacement range, cursor, old region, history, diagnostics, and scope preservation",
            "renderer boundary and local HF tokenizer split_special_tokens=True",
            "4096 context plus 192-output fit",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("long-context-budget-candidates.json"))
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
