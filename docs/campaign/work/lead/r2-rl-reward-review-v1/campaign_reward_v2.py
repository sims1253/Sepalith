#!/usr/bin/env python3
"""CPU-safe reward-v2 core for already parsed PRM03 TRAIN candidates.

This module does not execute R.  A caller may supply a parse-only probe whose
contract is syntax validation of text.  Semantic correctness remains exact
target equality only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Sequence


class RewardV2Error(ValueError):
    """A candidate or prepared buffer failed closed validation."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RewardV2Error(reason)


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utf16_index(line: str, units: int) -> int:
    require(units >= 0, "negative UTF-16 position")
    used = 0
    for index, char in enumerate(line):
        if used == units:
            return index
        used += len(char.encode("utf-16-le")) // 2
        require(used <= units, "UTF-16 position splits a code point")
    require(used == units, "UTF-16 position exceeds line")
    return len(line)


def position_offset(text: str, line_number: int, units: int) -> int:
    lines = text.splitlines(keepends=True)
    if line_number == len(lines) and text.endswith(("\n", "\r")):
        require(units == 0, "nonzero position after final EOL")
        return len(text)
    require(0 <= line_number < len(lines), "line position exceeds buffer")
    body = lines[line_number].rstrip("\r\n")
    return sum(len(line) for line in lines[:line_number]) + utf16_index(body, units)


def apply_region(baseline: str, replacement_range: Mapping[str, object], body_text: str,
                 document_eol: str) -> str:
    require(document_eol in {"lf", "crlf"}, "mixed or invalid EOL is unsupported")
    start = replacement_range["start"]
    end = replacement_range["end"]
    require(isinstance(start, Mapping) and isinstance(end, Mapping), "replacement positions missing")
    left = position_offset(baseline, int(start["line"]), int(start["character"]))
    right = position_offset(baseline, int(end["line"]), int(end["character"]))
    require(left <= right, "replacement range reversed")
    eol = "\r\n" if document_eol == "crlf" else "\n"
    return baseline[:left] + body_text.replace("\n", eol) + baseline[right:]


def severe_repetition(lines: Sequence[str]) -> dict[str, int | bool]:
    """Detect three adjacent copies of a substantive block.

    Exact targets bypass this diagnostic.  Requiring three copies and at least
    six repeated substantive lines avoids treating normal paired statements or
    closing braces as runaway repetition.
    """
    normalized = [line.rstrip() for line in lines]
    best = {"detected": False, "block_lines": 0, "copies": 0, "covered_lines": 0}
    for block_lines in range(1, min(8, len(normalized) // 3) + 1):
        for start in range(0, len(normalized) - 3 * block_lines + 1):
            block = normalized[start:start + block_lines]
            if not any(any(char.isalnum() for char in line) for line in block):
                continue
            copies = 1
            while normalized[start + copies * block_lines:start + (copies + 1) * block_lines] == block:
                copies += 1
            covered = copies * block_lines
            if copies >= 3 and covered >= 6 and covered > int(best["covered_lines"]):
                best = {"detected": True, "block_lines": block_lines,
                        "copies": copies, "covered_lines": covered}
    return best


@dataclass(frozen=True)
class BufferEvidence:
    """Prepared text identity for a syntax-only application check."""

    mode: str  # complete_document, completion_prefix, or unverified
    baseline_text: str | None
    baseline_sha256: str
    baseline_parse_ok: bool | None
    gold_applied_parse_ok: bool | None
    parser_identity: str | None

    def validate(self, expected_sha256: str) -> None:
        require(self.mode in {"complete_document", "completion_prefix", "unverified"},
                "buffer evidence mode differs")
        require(self.baseline_sha256 == expected_sha256, "buffer evidence context hash differs")
        if self.mode == "unverified":
            require(self.baseline_text is None and self.baseline_parse_ok is None
                    and self.gold_applied_parse_ok is None and self.parser_identity is None,
                    "unverified buffer carries unadmitted parse claims")
            return
        require(self.baseline_text is not None and sha_text(self.baseline_text) == self.baseline_sha256,
                "baseline buffer bytes differ")
        require(self.parser_identity is not None, "parse evidence lacks parser identity")
        require(self.gold_applied_parse_ok is True, "gold-applied buffer must parse")
        if self.mode == "complete_document":
            require(self.baseline_parse_ok is True, "complete baseline buffer must parse")
        else:
            # A completion prefix is expected to be incomplete.  Its failure is
            # evidence about construction, not a model defect.
            require(self.baseline_parse_ok in {False, None},
                    "completion-prefix baseline must not claim a complete parse")


def score_candidate(
    *,
    target_operation: str,
    target_body_text: str,
    region_old: Sequence[str],
    protocol_valid: bool,
    protocol_failure: str | None,
    operation: str | None,
    body_lines: Sequence[str],
    replacement_range: Mapping[str, object],
    document_eol: str,
    buffer: BufferEvidence,
    parse_probe: Callable[[str], bool] | None,
) -> tuple[float, dict[str, object]]:
    """Score one protocol-parsed completion without executing generated R."""
    require(target_operation in {"no_op", "replace", "delete"}, "target operation differs")
    buffer.validate(str(replacement_range["content_sha256"]))
    record: dict[str, object] = {
        "reward_policy": "protocol_restraint_syntax_v2_preparation",
        "protocol_valid": protocol_valid,
        "protocol_failure": protocol_failure,
        "operation": operation,
        "expected_operation": target_operation,
        "exact_region": False,
        "false_noop_edit": False,
        "repetition": {"detected": False, "block_lines": 0, "copies": 0, "covered_lines": 0},
        "candidate_parse": "not_checked",
        "semantic_correctness": "not_measured_beyond_exact_region",
    }
    if not protocol_valid:
        require(operation is None, "invalid protocol candidate carries an operation")
        record.update(reward=-1.0, outcome="invalid_or_unterminated")
        return -1.0, record
    require(protocol_failure is None and operation in {"no_op", "replace", "delete"},
            "valid protocol candidate metadata differs")
    actual = list(region_old) if operation == "no_op" else list(body_lines)
    expected = (list(region_old) if target_operation == "no_op" else
                [] if target_operation == "delete" else target_body_text.split("\n"))
    exact = actual == expected
    record["exact_region"] = exact
    if exact:
        record.update(reward=1.2, outcome="exact")
        return 1.2, record
    if target_operation == "no_op":
        record.update(reward=-1.0, outcome="false_noop_edit", false_noop_edit=True)
        return -1.0, record
    if operation == "no_op":
        record.update(reward=0.0, outcome="missed_edit")
        return 0.0, record
    repetition = severe_repetition(actual)
    record["repetition"] = repetition
    if repetition["detected"]:
        record.update(reward=-0.75, outcome="severe_repetition")
        return -0.75, record
    if buffer.mode != "unverified":
        require(parse_probe is not None and buffer.baseline_text is not None,
                "verified buffer lacks parse-only probe")
        candidate = apply_region(buffer.baseline_text, replacement_range,
                                 "\n".join(actual), document_eol)
        if not parse_probe(candidate):
            record.update(reward=-0.5, outcome="applied_syntax_invalid", candidate_parse="failed")
            return -0.5, record
        record["candidate_parse"] = "passed"
    else:
        require(parse_probe is None, "unverified window must not run a parse check")
        record["candidate_parse"] = "unavailable_unverified_buffer"
    record.update(reward=0.0, outcome="wrong_content_no_semantic_credit")
    return 0.0, record
