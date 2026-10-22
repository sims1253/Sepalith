"""Mechanical verification of SYNTH-v1 cases.

Per case (all recorded in ``verification``):

- ``render``: the case renders through ``campaign_protocol.build_training_row``
  and the row passes ``validate_training_row``;
- ``length``: prompt <= 3,072 tokens and completion (with EOS) <= 1,024
  (the reviewed v4 length policy);
- ``parse_after``: the edited window parses in R;
- ``parse_before``: the pre-edit window parses (roxygen, format), or is
  exactly incomplete ("unexpected end of input", finish_block);
- ``air_idempotent``: air leaves the edited window unchanged, so the gold
  whitespace is canonical;
- ``reward_v21``: scoring the gold tokens with reward v2.1 gives
  ``exact`` = true and reward 1.0, with no reference flag other than
  ``formatting_only_reference`` for format_propagation;
- family checks: finish_block closes on its last line at the header column;
  roxygen tags are known and ``@param`` names match the signature in order;
  format_propagation's region differs from its target and its history edit
  parses.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import tempfile

from sepalith.campaign_protocol import PromptContext, TokenEncoder, build_training_row, validate_training_row
from sepalith.data.synth.cases import Case, document_text
from sepalith.data.synth.roxygen import check_block
from sepalith.training.rl.air_canonical import AirFormatter
from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec, RewardV21Scorer, Sample

PARSE_STATUS = Path(__file__).with_name("r_parse_status.R")
PROMPT_MAX = 3072
COMPLETION_MAX = 1024
CLOSE_RE = re.compile(r"^\}[)\]]*$")


def parse_status(texts: Sequence[str], *, batch: int = 200, workers: int = 4) -> list[str]:
    chunks = [list(range(i, min(i + batch, len(texts)))) for i in range(0, len(texts), batch)]
    out: list[str] = [""] * len(texts)

    def run(indexes: list[int]) -> None:
        with tempfile.TemporaryDirectory(prefix="synth-parse-") as tmp:
            paths = []
            for index in indexes:
                path = Path(tmp) / f"{index}.R"
                path.write_text(texts[index], encoding="utf-8")
                paths.append(str(path))
            proc = subprocess.run(["Rscript", "--vanilla", str(PARSE_STATUS), *paths], capture_output=True,
                                  text=True, check=True, timeout=900)
            lines = proc.stdout.strip().split("\n")
            if len(lines) != len(indexes):
                raise RuntimeError("parse harness returned the wrong number of verdicts")
            for line in lines:
                position, status = line.split("\t")
                out[indexes[int(position)]] = status

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, chunks))
    return out


def render(case: Case, tokenizer: TokenEncoder, row_id: str) -> tuple[dict | None, dict, str | None]:
    try:
        context = case.to_context()
        row = build_training_row(PromptContext.from_dict(context), operation="replace", region_new=case.target,
                                 tokenizer=tokenizer, row_id=row_id, family=case.family,
                                 package_id=case.package_id, split="train")
        validate_training_row(row)
    except Exception as error:  # noqa: BLE001 - recorded as a verification failure
        return None, {}, f"{type(error).__name__}: {error}"
    return row, context, None


def family_checks(case: Case) -> list[str]:
    problems = []
    if case.family == "finish_block":
        last = case.target[-1]
        if not CLOSE_RE.match(last.strip()):
            problems.append("target_does_not_end_with_close")
        # Units are top-level expressions: the header starts at column 0, so the
        # closing line must too (the v4-braceindent convention).
        if len(last) - len(last.lstrip()) != 0:
            problems.append("closing_brace_not_at_header_column")
    elif case.family == "roxygen_drafting":
        problems += [f"roxygen:{p}" for p in case.meta.get("roxygen_problems", [])]
        if not all(line.startswith("#'") for line in case.target):
            problems.append("roxygen_target_not_all_roxygen_lines")
    elif case.family == "format_propagation":
        if tuple(case.region_old) == tuple(case.target):
            problems.append("format_region_equals_target")
    return problems


def verify(cases: Sequence[Case], tokenizer: TokenEncoder, scorer: RewardV21Scorer, formatter: AirFormatter,
           decoder: Callable[[Sequence[int]], str], *, workers: int = 8) -> list[tuple[dict | None, dict, dict]]:
    """Return (row, context, verification) per case, in order."""
    results: list[tuple[dict | None, dict, dict]] = []
    for case in cases:
        row, context, error = render(case, tokenizer, case.row_id())
        checks: dict = {"render": error is None}
        if error or row is None:
            checks["render_error"] = (error or "no row")[:300]
        else:
            checks["prompt_tokens"] = row["prompt_token_count"]
            checks["completion_tokens"] = row["target_token_count"] + 1
            checks["length"] = (row["prompt_token_count"] <= PROMPT_MAX
                                and row["target_token_count"] + 1 <= COMPLETION_MAX)
        results.append((row, context, checks))
    after = [document_text(c.window_after()) for c in cases]
    before = [document_text(c.window_before()) if c.window_before() else "\n" for c in cases]
    events = [document_text(c.events[0].before_lines) if c.events else "\n" for c in cases]
    status_after = parse_status(after)
    status_before = parse_status(before)
    status_event = parse_status(events)

    def air_check(text: str) -> bool:
        formatted = formatter.format(text)
        return formatted is not None and formatted.rstrip("\n") == text.rstrip("\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        air_ok = list(pool.map(air_check, after))
    for index, case in enumerate(cases):
        checks = results[index][2]
        checks["parse_after"] = status_after[index] == "ok"
        expected_before = "incomplete" if case.family == "finish_block" else "ok"
        checks["parse_before"] = status_before[index] == expected_before
        checks["parse_before_status"] = status_before[index]
        checks["history_parses"] = status_event[index] == "ok"
        if case.family == "format_propagation":
            # The suffix holds later statements in the old layout by design, so the
            # window is not air-stable; the generator proved air(collapsed) == source.
            checks["air_idempotent"] = case.meta.get("air_roundtrip") is True
        else:
            checks["air_idempotent"] = air_ok[index]
        checks["family_problems"] = family_checks(case)
    # reward v2.1 on the gold completion
    specs = []
    samples = []
    for (row, context, _checks), case in zip(results, cases, strict=True):
        if row is None:
            continue
        specs.append(ReferenceSpec(row["id"], case.family, PromptContext.from_dict(context), "replace",
                                   row["target_body_text"], Evidence("unverified")))
        samples.append(Sample(row["id"], completion_ids=tuple(row["input_ids"][row["target_start"]:]),
                              sample_id="gold"))
    references = scorer.build_references(specs)
    records = {r["row_id"]: r for r in scorer.score(references, samples, decoder)}
    for (row, _context, checks), case in zip(results, cases, strict=True):
        if row is None:
            checks["reward_v21"] = False
            continue
        record = records[row["id"]]
        flags = record["flags"]
        assert isinstance(flags, list)
        allowed = {"formatting_only_reference"} if case.family == "format_propagation" else set()
        checks["reward_exact"] = bool(record["exact"])
        checks["reward"] = record["reward"]
        checks["reward_comparison"] = record["comparison"]
        checks["reward_flags"] = list(flags)
        checks["reward_v21"] = (bool(record["exact"]) and float(str(record["reward"])) == 1.0
                                and set(flags) <= allowed)
    for _row, _context, checks in results:
        checks["passed"] = bool(
            checks.get("render") and checks.get("length") and checks.get("parse_after")
            and checks.get("parse_before") and checks.get("history_parses") and checks.get("air_idempotent")
            and not checks.get("family_problems") and checks.get("reward_v21"))
    return results


__all__ = ["verify", "parse_status", "check_block"]
