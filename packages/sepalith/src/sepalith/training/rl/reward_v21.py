"""Reward v2.1 for PRM03 editing samples (card C11).

The census (C12), RFT and DPO (C13) and any GRPO (C15) score samples with this
module. Every record carries two values:

- ``exact``: the sample is correct. For an edit case this means canonical
  equality with the reference edit; for a no-op case it means ``[NO_EDIT]``.
  Decision rules and pass rates use ``exact`` only.
- ``reward``: the graded training signal.

| Outcome | Reward |
| --- | --- |
| Correct edit (canonical ``exact``) | 1.0 |
| Edit that parses but is not exact | ``partial_weight`` (0.5) x credit in [0, 1] |
| Edit that does not parse | 0 |
| Edit case answered with ``[NO_EDIT]`` | 0 |
| No-op case answered with ``[NO_EDIT]`` | 1.0 |
| No-op case with any edit | -0.5 |
| Protocol-invalid output or cap hit | -0.5 |

Canonical comparison. The candidate and the reference are each applied to the
same pre-edit text: the hash-pinned reward buffer when one exists, otherwise
the prompt's context window (prefix, region, suffix). Both results go through
``air format`` and then line-ending and trailing-whitespace normalization.
``exact`` means the canonical texts are equal. If air cannot format either side
the comparison falls back to the normalized raw region and the record is
flagged. When the reference edit is itself formatting-only (air maps the
reference and the unedited text to the same canonical text, as in
format_propagation), canonical equality cannot tell a correct answer from an
unchanged one, so that row is compared raw and flagged as well.

Partial credit is secondary and can be switched off (``partial_credit="off"``).
The default ``"gain"`` credit is the canonical line-F1 between candidate and
reference regions, measured as improvement over leaving the region unchanged:
``max(0, (f1(candidate) - f1(old)) / (1 - f1(old)))``. Returning the old region
with cosmetic changes therefore earns nothing. ``"plain"`` uses the raw line-F1.
The weight is capped at 0.5, so a near miss always ranks below a correct edit.

Parse validity is decided by R (``RParseOnlyProbe``). Only the designated
syntax-error exit counts as invalid R. Infrastructure failures of R or air are
retried and then raised; they are never scored as model failures.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import difflib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

from sepalith.campaign_protocol import (
    EOS_ID,
    NO_EDIT,
    TOKENIZER_JSON_SHA256,
    VOCAB_SIZE,
    PromptContext,
    ProtocolError,
    is_native_control_token,
    parse_output,
    valid_generation_tokens,
)
from sepalith.training.rl.air_canonical import AirFormatter, normalize_lines, normalize_text
from sepalith.training.rl.campaign_r_parse_probe import DEFAULT_HARNESS, RParseOnlyProbe, sha_file
from sepalith.training.rl.campaign_reward_v2 import apply_region, position_offset
from sepalith.training.rl.length_policy import REVIEWED_POLICY

POLICY_ID = "sepalith.reward.v2.1"
RECORD_SCHEMA = "sepalith.reward-v21.record.v1"
NONCANONICAL_EOG_ID = 130073

REWARD_EXACT = 1.0
REWARD_CORRECT_NO_EDIT = 1.0
REWARD_FALSE_POSITIVE_EDIT = -0.5
REWARD_PROTOCOL_INVALID = -0.5
REWARD_CAP_HIT = -0.5
REWARD_SYNTAX_INVALID = 0.0
REWARD_MISSED_EDIT = 0.0
REWARD_UNVERIFIED_WRONG = 0.0
MAX_PARTIAL_WEIGHT = 0.5
PARTIAL_MODES = ("gain", "plain", "off")

VERIFIED_MODES = frozenset({"complete_document", "completion_prefix", "framed_fragment"})
BEGIN_MARKER = "# sepalith-reward-v21-region-begin"
END_MARKER = "# sepalith-reward-v21-region-end"

LOGGER = logging.getLogger("sepalith.training.rl.reward_v21")


class RewardV21Error(ValueError):
    """An input, reference or configuration failed closed validation."""


@dataclass(frozen=True)
class RewardV21Config:
    partial_credit: str = "gain"
    partial_weight: float = MAX_PARTIAL_WEIGHT
    max_completion_tokens: int = REVIEWED_POLICY.completion_max_tokens

    def __post_init__(self) -> None:
        if self.partial_credit not in PARTIAL_MODES:
            raise RewardV21Error(f"partial_credit must be one of {PARTIAL_MODES}")
        if not isinstance(self.partial_weight, (int, float)) or not 0 <= self.partial_weight <= MAX_PARTIAL_WEIGHT:
            raise RewardV21Error("partial_weight must lie in [0, 0.5]")
        if type(self.max_completion_tokens) is not int or self.max_completion_tokens < 1:
            raise RewardV21Error("max_completion_tokens must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {"partial_credit": self.partial_credit, "partial_weight": float(self.partial_weight),
                "max_completion_tokens": self.max_completion_tokens}


def line_f1(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Line-level F1 from difflib's matching blocks, in [0, 1]."""
    pred = [line.rstrip() for line in predicted]
    gold = [line.rstrip() for line in expected]
    while pred and pred[-1] == "":
        pred.pop()
    while gold and gold[-1] == "":
        gold.pop()
    if pred == gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    matches = sum(block.size for block in difflib.SequenceMatcher(a=pred, b=gold, autojunk=False).get_matching_blocks())
    if matches == 0:
        return 0.0
    precision = matches / len(pred)
    recall = matches / len(gold)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Surfaces: the text an edit is applied to.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Surface:
    """Rebuilds the edited text around the replacement region.

    ``document`` surfaces use the hash-pinned reward-buffer baseline and the
    exact replacement range. ``window`` surfaces splice whole lines into the
    prompt's context window when no baseline exists.
    """

    kind: str
    baseline: str | None = None
    replacement_range: Mapping[str, object] | None = None
    document_eol: str = "lf"
    diagnostic_suffix: str = ""
    start_line: int = 0
    start_character: int = 0
    prefix_lines: tuple[str, ...] = ()
    suffix_lines: tuple[str, ...] = ()

    @property
    def eol(self) -> str:
        return "\r\n" if self.kind == "document" and self.document_eol == "crlf" else "\n"

    def render(self, body: Sequence[str]) -> tuple[str, int, int]:
        """Return the edited text and the [left, right) offsets of the body."""
        if self.kind == "document":
            if self.baseline is None or self.replacement_range is None:
                raise RewardV21Error("document surface lacks its baseline")
            text = apply_region(self.baseline, self.replacement_range, "\n".join(body), self.document_eol)
            left = position_offset(self.baseline, self.start_line, self.start_character)
            right = left + len("\n".join(body).replace("\n", self.eol))
            return text + self.diagnostic_suffix, left, right
        if self.kind == "window":
            head = "\n".join(self.prefix_lines)
            left = len(head) + (1 if self.prefix_lines else 0)
            text = "\n".join((*self.prefix_lines, *body, *self.suffix_lines)) + "\n"
            return text, left, left + len("\n".join(body))
        raise RewardV21Error(f"unknown surface kind {self.kind!r}")

    def render_marked(self, body: Sequence[str]) -> str:
        """The edited text with comment-line markers around the region.

        The markers sit on their own lines just outside the region, so both
        sides of a comparison carry them in the same place. They are used only
        to locate the canonical region for partial credit, never for ``exact``.
        """
        text, left, right = self.render(body)
        eol = self.eol
        tail = text[right:]
        if tail and not tail.startswith(("\n", "\r")):
            tail = eol + tail
        return text[:left] + BEGIN_MARKER + eol + text[left:right] + eol + END_MARKER + tail


def canonical_region(formatted: str | None) -> tuple[str, ...] | None:
    if formatted is None:
        return None
    lines = normalize_text(formatted).split("\n")
    begins = [index for index, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    ends = [index for index, line in enumerate(lines) if line.strip() == END_MARKER]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        return None
    region = lines[begins[0] + 1:ends[0]]
    while region and region[0] == "":
        region.pop(0)
    while region and region[-1] == "":
        region.pop()
    return tuple(region)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    """Reward-buffer evidence for one row (``unverified`` when none exists)."""

    mode: str
    baseline_text: str | None = None
    diagnostic_suffix: str = ""
    parser_r_version: str | None = None

    @classmethod
    def from_buffer_envelope(cls, envelope: Mapping[str, object], baseline_text: str | None) -> Evidence:
        mode = envelope.get("syntax_evidence_mode")
        if mode == "unverified":
            return cls("unverified")
        if mode not in VERIFIED_MODES or baseline_text is None:
            raise RewardV21Error(f"reward-buffer evidence mode {mode!r} is unusable")
        identity = envelope.get("parser_identity")
        version = identity.get("r_version") if isinstance(identity, Mapping) else None
        return cls(str(mode), baseline_text, str(envelope.get("diagnostic_suffix") or ""),
                   str(version) if version is not None else None)


@dataclass(frozen=True)
class ReferenceSpec:
    row_id: str
    family: str
    context: PromptContext
    target_operation: str
    target_body_text: str
    evidence: Evidence


@dataclass(frozen=True)
class RowReference:
    row_id: str
    family: str
    context: PromptContext
    target_operation: str
    target_lines: tuple[str, ...]
    surface: Surface | None
    evidence_mode: str
    parse_available: bool
    gold_canonical: str | None
    gold_region: tuple[str, ...] | None
    old_region: tuple[str, ...] | None
    formatting_target: bool
    flags: tuple[str, ...] = ()

    @property
    def is_edit(self) -> bool:
        return self.target_operation != "no_op"


def _target_lines(operation: str, body_text: str) -> tuple[str, ...]:
    if operation == "no_op":
        if body_text != NO_EDIT:
            raise RewardV21Error("no-op target body differs from [NO_EDIT]")
        return ()
    if operation == "delete":
        if body_text:
            raise RewardV21Error("delete target carries a body")
        return ()
    if operation == "replace":
        if not body_text:
            raise RewardV21Error("replace target lacks a body")
        return tuple(body_text.split("\n"))
    raise RewardV21Error(f"target operation {operation!r} is invalid")


def _surface_for(spec: ReferenceSpec) -> Surface:
    context = spec.context
    if spec.evidence.mode in VERIFIED_MODES:
        return Surface(
            "document", baseline=spec.evidence.baseline_text,
            replacement_range=context.replacement_range.to_dict(),
            document_eol=context.document_eol,
            diagnostic_suffix=spec.evidence.diagnostic_suffix if spec.evidence.mode == "framed_fragment" else "",
            start_line=context.replacement_range.start.line,
            start_character=context.replacement_range.start.character,
        )
    # PromptContext already requires the range to end at the end of its last
    # region line; a window splice also needs it to start at column 0.
    if context.replacement_range.start.character != 0:
        raise RewardV21Error(f"row {spec.row_id}: a context window needs a whole-line region")
    return Surface("window", prefix_lines=tuple(context.prefix), suffix_lines=tuple(context.suffix_lines))


# ---------------------------------------------------------------------------
# Completion protocol checks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    """One completion to score. Give token IDs, or decoded text plus how it stopped."""

    row_id: str
    completion_ids: tuple[int, ...] | None = None
    completion_text: str | None = None
    stop: str = "eos"  # text path only: "eos" or "length"
    sample_id: str | None = None


@dataclass
class _Work:
    reference: RowReference
    sample: Sample
    record: dict[str, object] = field(default_factory=dict)
    body: tuple[str, ...] | None = None
    candidate_text: str | None = None
    parse_ok: bool | None = None
    done: bool = False


def _check_ids(ids: Sequence[int], cap: int) -> tuple[str | None, bool]:
    """Return (failure, cap_hit) for generated IDs; mirrors reward v2's checks."""
    if not ids:
        return "empty_completion", False
    if len(ids) > cap:
        return "completion_cap_exceeded", True
    if EOS_ID in ids[:-1]:
        return "premature_canonical_eos", False
    if any(is_native_control_token(token) for token in ids[:-1]):
        return "control_before_terminal", False
    if ids[-1] == NONCANONICAL_EOG_ID:
        return "noncanonical_eog", False
    if ids[-1] != EOS_ID:
        return "missing_canonical_eos", len(ids) >= cap
    if not valid_generation_tokens(list(ids)):
        return "invalid_generation_tokens", False
    return None, False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class RewardV21Scorer:
    """Batch scorer. Thread-safe; R and air run as subprocesses."""

    def __init__(
        self,
        probe: RParseOnlyProbe,
        formatter: AirFormatter,
        config: RewardV21Config | None = None,
        *,
        workers: int = 8,
        chunk_size: int = 32,
    ) -> None:
        if type(workers) is not int or not 1 <= workers <= 64:
            raise RewardV21Error("workers must be in [1, 64]")
        if type(chunk_size) is not int or chunk_size < 1:
            raise RewardV21Error("chunk_size must be positive")
        self.probe = probe
        self.formatter = formatter
        self.config = config or RewardV21Config()
        self.workers = workers
        self.chunk_size = chunk_size
        self.r_version = probe.r_version()
        self.air_version = formatter.version()

    def identity(self) -> dict[str, object]:
        return {
            "policy": POLICY_ID,
            "config": self.config.to_dict(),
            "parser": self.probe.identity(),
            "formatter": {"version": self.air_version, "command": " ".join(self.formatter.command),
                          "arguments": "format --no-configuration --stdin-file-path"},
            "rewards": {
                "exact": REWARD_EXACT, "correct_no_edit": REWARD_CORRECT_NO_EDIT,
                "false_positive_edit": REWARD_FALSE_POSITIVE_EDIT,
                "protocol_invalid": REWARD_PROTOCOL_INVALID, "cap_hit": REWARD_CAP_HIT,
                "syntax_invalid": REWARD_SYNTAX_INVALID, "missed_edit": REWARD_MISSED_EDIT,
                "unverified_wrong": REWARD_UNVERIFIED_WRONG,
            },
        }

    def _map(self, function: Callable[[Sequence[_Work]], None], items: Sequence[_Work]) -> None:
        chunks = [items[index:index + self.chunk_size] for index in range(0, len(items), self.chunk_size)]
        if self.workers == 1 or len(chunks) <= 1:
            for chunk in chunks:
                function(chunk)
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in pool.map(function, chunks):
                pass

    # -- references -------------------------------------------------------
    def build_references(self, specs: Sequence[ReferenceSpec]) -> dict[str, RowReference]:
        out: dict[str, RowReference] = {}
        lock = threading.Lock()

        def build(chunk: Sequence[ReferenceSpec]) -> None:
            prepared = []
            parse_texts = []
            for spec in chunk:
                target = _target_lines(spec.target_operation, spec.target_body_text)
                if spec.target_operation == "no_op":
                    prepared.append((spec, target, None, None))
                    continue
                if spec.evidence.mode in VERIFIED_MODES and spec.evidence.parser_r_version not in (None, self.r_version):
                    raise RewardV21Error(
                        f"row {spec.row_id}: reward-buffer parser {spec.evidence.parser_r_version!r} "
                        f"differs from the probe's {self.r_version!r}")
                surface = _surface_for(spec)
                gold_text, _, _ = surface.render(target)
                index = None
                if surface.kind == "window":
                    index = len(parse_texts)
                    parse_texts.append(gold_text)
                prepared.append((spec, target, surface, index))
            verdicts = self.probe.parse_many(parse_texts) if parse_texts else []
            built = []
            for spec, target, surface, index in prepared:
                if surface is None:
                    built.append(RowReference(spec.row_id, spec.family, spec.context, spec.target_operation,
                                              target, None, "operation_only", False, None, None, None, False))
                    continue
                flags = []
                parse_available = True if index is None else verdicts[index]
                mode = spec.evidence.mode if surface.kind == "document" else "context_window"
                if not parse_available:
                    flags.append("reference_window_does_not_parse")
                gold_text, _, _ = surface.render(target)
                old_text, _, _ = surface.render(spec.context.region_old)
                gold_formatted = self.formatter.format(gold_text)
                old_formatted = self.formatter.format(old_text)
                gold_canonical = normalize_text(gold_formatted) if gold_formatted is not None else None
                old_canonical = normalize_text(old_formatted) if old_formatted is not None else None
                gold_region = canonical_region(self.formatter.format(surface.render_marked(target)))
                old_region = (canonical_region(self.formatter.format(surface.render_marked(spec.context.region_old)))
                              if old_canonical is not None else None)
                if gold_canonical is None:
                    flags.append("reference_air_failed")
                elif gold_region is None:
                    flags.append("reference_region_markers_lost")
                formatting_target = gold_canonical is not None and gold_canonical == old_canonical
                if formatting_target:
                    flags.append("formatting_only_reference")
                built.append(RowReference(
                    spec.row_id, spec.family, spec.context, spec.target_operation, target, surface, mode,
                    parse_available, gold_canonical, gold_region, old_region, formatting_target, tuple(flags),
                ))
            with lock:
                for reference in built:
                    out[reference.row_id] = reference

        chunks = [specs[index:index + self.chunk_size] for index in range(0, len(specs), self.chunk_size)]
        if self.workers == 1 or len(chunks) <= 1:
            for chunk in chunks:
                build(chunk)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for _ in pool.map(build, chunks):
                    pass
        return out

    # -- scoring ----------------------------------------------------------
    def score(
        self,
        references: Mapping[str, RowReference],
        samples: Sequence[Sample],
        decoder: Callable[[Sequence[int]], str] | None = None,
    ) -> list[dict[str, object]]:
        work = []
        for sample in samples:
            reference = references.get(sample.row_id)
            if reference is None:
                raise RewardV21Error(f"no reference for row {sample.row_id}")
            work.append(_Work(reference, sample))

        def run(chunk: Sequence[_Work]) -> None:
            for item in chunk:
                self._protocol(item, decoder)
            pending = [item for item in chunk if not item.done and item.candidate_text is not None
                       and item.reference.parse_available]
            verdicts = self.probe.parse_many([str(item.candidate_text) for item in pending]) if pending else []
            for item, verdict in zip(pending, verdicts, strict=True):
                item.parse_ok = verdict
            for item in chunk:
                if not item.done:
                    self._finish(item)

        self._map(run, work)
        return [item.record for item in work]

    def _base_record(self, item: _Work) -> dict[str, object]:
        reference, sample = item.reference, item.sample
        return {
            "schema": RECORD_SCHEMA, "reward_policy": POLICY_ID,
            "row_id": reference.row_id, "sample_id": sample.sample_id, "family": reference.family,
            "expected_operation": reference.target_operation, "operation": None,
            "protocol_valid": False, "failure": None, "cap_hit": False, "generated_tokens": None,
            "exact": False, "exact_raw": False, "reward": 0.0, "outcome": None,
            "comparison": None, "canonical_fallback": False, "evidence_mode": reference.evidence_mode,
            "parse": "not_checked", "similarity": None, "baseline_similarity": None,
            "partial_credit": None, "similarity_space": None, "flags": list(reference.flags),
            "parser": self.r_version,
        }

    def _close(self, item: _Work, reward: float, outcome: str, *, exact: bool = False) -> None:
        item.record.update(reward=float(reward), outcome=outcome, exact=exact)
        item.done = True

    def _protocol(self, item: _Work, decoder: Callable[[Sequence[int]], str] | None) -> None:
        item.record = self._base_record(item)
        sample, reference = item.sample, item.reference
        cap = self.config.max_completion_tokens
        failure: str | None
        if sample.completion_ids is not None:
            ids = list(sample.completion_ids)
            if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
                failure, cap_hit = "invalid_token_ids", False
            else:
                failure, cap_hit = _check_ids(ids, cap)
            item.record["generated_tokens"] = len(ids)
            text = None
            if failure is None:
                if decoder is None:
                    raise RewardV21Error("token IDs need a decoder")
                text = decoder(ids[:-1])
        elif sample.completion_text is not None:
            if sample.stop not in ("eos", "length"):
                raise RewardV21Error("text samples need stop 'eos' or 'length'")
            cap_hit = sample.stop == "length"
            failure = "missing_canonical_eos" if cap_hit else None
            text = sample.completion_text
        else:
            raise RewardV21Error(f"sample for row {sample.row_id} has neither IDs nor text")
        item.record["cap_hit"] = cap_hit
        if failure is None and text is not None:
            try:
                parsed = parse_output(text, reference.context)
            except (ProtocolError, TypeError, ValueError) as error:
                failure = f"parse_error:{error}"
            else:
                if parsed.status != "accepted" or parsed.operation is None:
                    failure = parsed.reason or "invalid_protocol_output"
                else:
                    item.record.update(protocol_valid=True, operation=parsed.operation)
                    item.body = () if parsed.operation == "no_op" else tuple(parsed.body)
        if failure is not None:
            item.record["failure"] = failure
            if cap_hit:
                self._close(item, REWARD_CAP_HIT, "cap_hit")
            else:
                self._close(item, REWARD_PROTOCOL_INVALID, "protocol_invalid")
            return
        operation = item.record["operation"]
        if not reference.is_edit:
            item.record["comparison"] = "operation"
            if operation == "no_op":
                item.record["exact_raw"] = True
                self._close(item, REWARD_CORRECT_NO_EDIT, "correct_no_edit", exact=True)
            else:
                self._close(item, REWARD_FALSE_POSITIVE_EDIT, "false_positive_edit")
            return
        if operation == "no_op":
            item.record["comparison"] = "operation"
            self._close(item, REWARD_MISSED_EDIT, "missed_edit")
            return
        assert reference.surface is not None and item.body is not None
        item.candidate_text, _, _ = reference.surface.render(item.body)
        if not reference.parse_available:
            item.record["parse"] = "unavailable"

    def _finish(self, item: _Work) -> None:
        reference, record = item.reference, item.record
        body = item.body
        assert body is not None and reference.surface is not None and item.candidate_text is not None
        record["exact_raw"] = body == reference.target_lines
        if item.parse_ok is False:
            record["parse"] = "failed"
            self._close(item, REWARD_SYNTAX_INVALID, "applied_syntax_invalid")
            return
        if item.parse_ok is True:
            record["parse"] = "passed"
        raw_equal = normalize_lines(body) == normalize_lines(reference.target_lines)
        candidate_canonical = None
        if reference.formatting_target:
            record["comparison"] = "raw_formatting_target"
            exact = raw_equal
        elif reference.gold_canonical is None:
            record.update(comparison="raw_fallback_reference", canonical_fallback=True)
            exact = raw_equal
        else:
            formatted = self.formatter.format(item.candidate_text)
            if formatted is None:
                record.update(comparison="raw_fallback_candidate", canonical_fallback=True)
                exact = raw_equal
            else:
                candidate_canonical = normalize_text(formatted)
                record["comparison"] = "air"
                exact = candidate_canonical == reference.gold_canonical
        if exact:
            self._close(item, REWARD_EXACT, "exact", exact=True)
            return
        if item.parse_ok is None:
            self._close(item, REWARD_UNVERIFIED_WRONG, "wrong_parse_unverified")
            return
        config = self.config
        if config.partial_credit == "off" or config.partial_weight == 0:
            self._close(item, 0.0, "near_miss")
            record["partial_credit"] = 0.0
            return
        space = "raw"
        candidate_lines: tuple[str, ...] = normalize_lines(body)
        gold_lines: tuple[str, ...] = normalize_lines(reference.target_lines)
        old_lines: tuple[str, ...] = normalize_lines(reference.context.region_old)
        if record["comparison"] == "air" and reference.gold_region is not None:
            region = canonical_region(self.formatter.format(reference.surface.render_marked(body)))
            if region is not None:
                space = "canonical"
                candidate_lines, gold_lines = region, reference.gold_region
                if reference.old_region is not None:
                    old_lines = reference.old_region
                else:
                    space = "canonical_with_raw_baseline"
        similarity = line_f1(candidate_lines, gold_lines)
        baseline = line_f1(old_lines, gold_lines)
        if config.partial_credit == "plain":
            credit = similarity
        else:
            credit = 0.0 if baseline >= 1.0 else max(0.0, (similarity - baseline) / (1.0 - baseline))
        credit = min(1.0, max(0.0, credit))
        record.update(similarity=round(similarity, 6), baseline_similarity=round(baseline, 6),
                      partial_credit=round(credit, 6), similarity_space=space)
        self._close(item, round(config.partial_weight * credit, 6), "near_miss")


# ---------------------------------------------------------------------------
# TRL-compatible reward function
# ---------------------------------------------------------------------------
class CampaignPRM03RewardV21:
    """Reward v2.1 with the call signature of ``CampaignPRM03Reward`` (v2).

    ``reward_buffer_index`` is a ``RewardBufferIndex``; its envelopes come from
    ``RLTrainRecord.envelope()``. References are built once per row and cached.
    """

    __name__ = "campaign_prm03_reward_v21"

    def __init__(
        self,
        scorer: RewardV21Scorer,
        reward_buffer_index: object,
        *,
        decoder: Callable[[Sequence[int]], str],
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.scorer = scorer
        self.reward_buffer_index = reward_buffer_index
        self.decoder = decoder
        self.event_sink = event_sink
        self.references: dict[str, RowReference] = {}
        self.last_records: list[dict[str, object]] = []

    def _spec(self, row_id: str, family: str, context: object, operation: str, body: str,
              envelope: Mapping[str, object]) -> ReferenceSpec:
        prompt_context = context if isinstance(context, PromptContext) else PromptContext.from_mapping(context)
        if envelope.get("row_id") != row_id:
            raise RewardV21Error(f"reward buffer row identity differs for {row_id}")
        if envelope.get("context_sha256") != prompt_context.replacement_range.content_sha256:
            raise RewardV21Error(f"reward buffer context identity differs for {row_id}")
        baseline = None
        if envelope.get("syntax_evidence_mode") != "unverified":
            loader = getattr(self.reward_buffer_index, "baseline_text", None)
            if loader is None:
                raise RewardV21Error("reward buffer index cannot load baselines")
            baseline = loader(envelope)
        return ReferenceSpec(row_id, family, prompt_context, operation, body,
                             Evidence.from_buffer_envelope(envelope, baseline))

    def __call__(
        self,
        *,
        completion_ids: Sequence[Sequence[int]],
        context: Sequence[object],
        target_operation: Sequence[str],
        target_body_text: Sequence[str],
        reward_buffer: Sequence[Mapping[str, object]],
        id: Sequence[str],  # TRL passes dataset columns by name
        family: Sequence[str] | None = None,
        **_: object,
    ) -> list[float]:
        size = len(completion_ids)
        if any(len(value) != size for value in (context, target_operation, target_body_text, reward_buffer, id)):
            raise RewardV21Error("reward metadata and completion IDs are not aligned")
        families = family or [""] * size
        missing = {}
        for index in range(size):
            if id[index] not in self.references and id[index] not in missing:
                missing[id[index]] = self._spec(id[index], families[index], context[index], target_operation[index],
                                                target_body_text[index], reward_buffer[index])
        if missing:
            self.references.update(self.scorer.build_references(list(missing.values())))
        samples = []
        for index in range(size):
            ids = completion_ids[index]
            to_list = getattr(ids, "tolist", None)
            values = to_list() if callable(to_list) else list(ids)
            samples.append(Sample(id[index], completion_ids=tuple(int(token) for token in values)))
        records = self.scorer.score(self.references, samples, self.decoder)
        self.last_records = records
        if self.event_sink is not None:
            for record in records:
                self.event_sink(record)
        return [float(str(record["reward"])) for record in records]


# ---------------------------------------------------------------------------
# Campaign data and CLI
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("SEPALITH_CAMPAIGN_DATA", "/mnt/e/sepalith/campaign-20260915/data-work"))
TRAIN_INPUTS = {
    "rows": ("DAT10-finish-source-repair-v3/train-token-rows.jsonl",
             "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"),
    "context": ("DAT10-finish-source-repair-v3/context-sidecar.jsonl",
                "36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a"),
    "selected_ids": ("RL11-15006-context-v1/selected-train-ids.json",
                     "986a1f7910c44423e50988e76abb6671dedf8cd70f5542ff9efd7a03ce504462"),
    "reward_buffer_manifest": ("RL11-finish-source-repair-buffer-v1/materialization.json",
                               "84f19b3011501b9614f6794dd6635b0e0aef56d8a853cb94fbbda4f8aa0467d4"),
}


def load_train_specs(
    data_root: Path = DATA_ROOT,
) -> tuple[list[ReferenceSpec], dict[str, tuple[int, ...]], dict[str, object]]:
    """Load the 15,006 TRAIN rows; return specs, each row's target IDs, and the data identity."""
    from sepalith.training.rl.campaign_rl_buffer import RewardBufferIndex
    from sepalith.training.rl.campaign_rl_data import load_training_records

    paths = {name: data_root / relative for name, (relative, _) in TRAIN_INPUTS.items()}
    records, manifest = load_training_records(
        paths["rows"], TRAIN_INPUTS["rows"][1], paths["context"], TRAIN_INPUTS["context"][1],
        paths["selected_ids"], TRAIN_INPUTS["selected_ids"][1],
        reward_buffer_manifest_path=paths["reward_buffer_manifest"],
        reward_buffer_manifest_sha256=TRAIN_INPUTS["reward_buffer_manifest"][1],
    )
    index = RewardBufferIndex.load(paths["reward_buffer_manifest"], TRAIN_INPUTS["reward_buffer_manifest"][1],
                                   manifest.selected_ids)
    specs = []
    targets = {}
    for record in records:
        envelope = record.reward_buffer
        assert envelope is not None
        baseline = index.baseline_text(envelope) if envelope.get("syntax_evidence_mode") != "unverified" else None
        row = record.row
        specs.append(ReferenceSpec(record.row_id, str(row["family"]), record.context, str(row["target_operation"]),
                                   str(row["target_body_text"]), Evidence.from_buffer_envelope(envelope, baseline)))
        targets[record.row_id] = tuple(row["input_ids"][row["target_start"]:])
    identity: dict[str, object] = {name: {"path": str(paths[name]), "sha256": sha}
                                   for name, (_, sha) in TRAIN_INPUTS.items()}
    identity["rows"] = len(records)
    identity["length_policy"] = manifest.length_policy.policy_id
    return specs, targets, identity


def tokenizer_decoder(path: Path) -> Callable[[Sequence[int]], str]:
    if sha_file(path) != TOKENIZER_JSON_SHA256:
        raise RewardV21Error(f"tokenizer.json at {path} is not the pinned tokenizer")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    return lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False)


def make_scorer(args: argparse.Namespace) -> RewardV21Scorer:
    probe = RParseOnlyProbe(DEFAULT_HARNESS, sha_file(DEFAULT_HARNESS))
    formatter = AirFormatter((args.air,))
    config = RewardV21Config(partial_credit=args.partial_credit, partial_weight=args.partial_weight)
    return RewardV21Scorer(probe, formatter, config, workers=args.workers)


def summarize(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {"outcome": {}, "comparison": {}, "family_exact": {}, "family_rows": {},
                                         "flags": {}}
    exact = 0
    total = 0
    reward_sum = 0.0
    for record in records:
        total += 1
        exact += bool(record["exact"])
        reward_sum += float(str(record["reward"]))
        for key in ("outcome", "comparison"):
            value = str(record[key])
            counts[key][value] = counts[key].get(value, 0) + 1
        family = str(record["family"])
        counts["family_rows"][family] = counts["family_rows"].get(family, 0) + 1
        counts["family_exact"][family] = counts["family_exact"].get(family, 0) + bool(record["exact"])
        flags = record["flags"]
        assert isinstance(flags, list)
        for flag in map(str, flags):
            counts["flags"][flag] = counts["flags"].get(flag, 0) + 1
    return {"samples": total, "exact": exact, "mean_reward": round(reward_sum / total, 6) if total else None,
            **counts}


def _perturb(body: Sequence[str]) -> tuple[str, ...]:
    """A deterministic near miss: repeat the first nonblank body line."""
    lines = list(body)
    for index, line in enumerate(lines):
        if line.strip():
            return (*lines[:index + 1], line, *lines[index + 1:])
    return (*lines, "x")


def command_self_check(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    specs, targets, data_identity = load_train_specs(Path(args.data_root))
    if args.limit:
        specs = specs[:args.limit]
    loaded = time.perf_counter()
    scorer = make_scorer(args)
    decoder = tokenizer_decoder(Path(args.tokenizer))
    references = scorer.build_references(specs)
    built = time.perf_counter()
    samples = [Sample(spec.row_id, completion_ids=targets[spec.row_id], sample_id="target") for spec in specs]
    records = scorer.score(references, samples, decoder)
    scored = time.perf_counter()
    perturbed_records: list[dict[str, object]] = []
    perturbed_seconds = None
    if args.perturbed:
        edits = [spec for spec in specs if spec.target_operation != "no_op"]
        chosen = edits[::max(1, len(edits) // args.perturbed)][:args.perturbed]  # spread over families
        perturbed = []
        for spec in chosen:
            target = references[spec.row_id].target_lines
            # A near miss, and the correct edit with every line indented two
            # more spaces (a formatting-only difference for code outside strings).
            for kind, body in (("near_miss", _perturb(target)),
                               ("reindented", tuple("  " + line if line.strip() else line for line in target))):
                perturbed.append(Sample(spec.row_id, completion_text="\n".join((*body, ">>>>>>> UPDATED", "")),
                                        sample_id=kind))
        start = time.perf_counter()
        perturbed_records = scorer.score(references, perturbed, decoder)
        perturbed_seconds = time.perf_counter() - start
    failures = [record for record in records if not record["exact"]]
    receipt = {
        "schema": "sepalith.c11.reward-v21-self-check.v1",
        "scorer": scorer.identity(),
        "data": data_identity,
        "workers": args.workers,
        "rows_scored": len(records),
        "self_check": summarize(records),
        "non_exact_rows": [{key: record[key] for key in ("row_id", "family", "outcome", "comparison", "failure",
                                                           "parse", "flags")} for record in failures[:50]],
        "non_exact_count": len(failures),
        "reference_flags": summarize(records)["flags"],
        "seconds": {
            "load": round(loaded - started, 3), "build_references": round(built - loaded, 3),
            "score_targets": round(scored - built, 3),
            "score_perturbed": round(perturbed_seconds, 3) if perturbed_seconds is not None else None,
        },
        "seconds_per_1000": {
            "references": round(1000 * (built - loaded) / max(1, len(specs)), 3),
            "target_samples": round(1000 * (scored - built) / max(1, len(records)), 3),
            "perturbed_samples": (round(1000 * perturbed_seconds / len(perturbed_records), 3)
                                  if perturbed_records and perturbed_seconds is not None else None),
        },
        "near_miss_probe": summarize(r for r in perturbed_records if r["sample_id"] == "near_miss"),
        "reindented_probe": summarize(r for r in perturbed_records if r["sample_id"] == "reindented"),
        "infrastructure_failures": {"r": scorer.probe.infrastructure_failures,
                                    "air": scorer.formatter.infrastructure_failures},
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.records:
        with Path(args.records).open("w", encoding="utf-8") as stream:
            for record in records + perturbed_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    sys.stdout.write(text)
    return 0 if not failures else 1


def command_score(args: argparse.Namespace) -> int:
    specs, _, data_identity = load_train_specs(Path(args.data_root))
    by_id = {spec.row_id: spec for spec in specs}
    samples = []
    with Path(args.samples).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            row_id = value.get("row_id")
            if row_id not in by_id:
                raise RewardV21Error(f"sample line {line_number}: unknown TRAIN row {row_id!r}")
            ids = value.get("completion_ids")
            samples.append(Sample(row_id, completion_ids=tuple(ids) if ids is not None else None,
                                  completion_text=value.get("completion_text"), stop=value.get("stop", "eos"),
                                  sample_id=value.get("sample_id")))
    scorer = make_scorer(args)
    decoder = tokenizer_decoder(Path(args.tokenizer)) if any(s.completion_ids is not None for s in samples) else None
    references = scorer.build_references([by_id[row_id] for row_id in dict.fromkeys(s.row_id for s in samples)])
    records = scorer.score(references, samples, decoder)
    with Path(args.output).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {"scorer": scorer.identity(), "data": data_identity, "summary": summarize(records)}
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reward v2.1 scorer (card C11)")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--tokenizer", required=True, help="pinned tokenizer.json")
    parser.add_argument("--air", default="air")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--partial-credit", choices=PARTIAL_MODES, default="gain")
    parser.add_argument("--partial-weight", type=float, default=MAX_PARTIAL_WEIGHT)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("self-check", help="score every TRAIN target against itself")
    check.add_argument("--limit", type=int, default=0)
    check.add_argument("--perturbed", type=int, default=0,
                       help="also score a near miss and a re-indented correct edit for N edit rows")
    check.add_argument("--output")
    check.add_argument("--records")
    score = commands.add_parser("score", help="score a JSONL file of samples")
    score.add_argument("--samples", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.command == "self-check":
        return command_self_check(args)
    return command_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
