#!/usr/bin/env python3
"""CPU-only DSpark/Opus09 shape and KV-state invariant checks.

This file deliberately does not import llama.cpp, ggml, a framework, or a model.
It checks source anchors in the pinned public tree and runs a small reference
state machine for the capacity and suffix-removal invariants.  The reference
state machine is an executable design check; it does not prove the native
runtime, graph allocation, or numerical model behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PINNED_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
PINNED_FILES = {
    "common/speculative.cpp": "81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3",
    "src/llama-context.cpp": "a061f57bc0aaf55697e41952569835a5d4e65cc8d7a97c25410d940b47ee3044",
    "src/llama-context.h": "7cd1b39819442f1f3c556bec487fdd8d1501967cd1fbe33feea59d5bbc556fc8",
    "src/models/dflash.cpp": "58d5feeb6f5a5c459e438b0bc531e8d53da33d13962f4bf589071eb649a82251",
}


class InvariantError(AssertionError):
    """An expected capacity, position, or output invariant was violated."""


@dataclass(frozen=True)
class DecodeShape:
    """Logical shape of one speculative draft decode.

    `full_dspark_block` models the proposed Opus09 repair only.  It is false
    for the pinned code path, which uses `n_max` rows for DSpark and
    `n_max + 1` rows for legacy DFlash.
    """

    mode: str
    n_max: int
    block_size: int
    n_parallel: int
    n_batch: int
    n_ubatch: int
    n_outputs_max: int
    n_outputs_max_per_seq: int
    full_dspark_block: bool = False

    @property
    def decode_rows_per_sequence(self) -> int:
        if self.mode == "dspark":
            return self.block_size if self.full_dspark_block else self.n_max
        if self.mode == "dflash":
            return self.n_max + 1
        raise InvariantError(f"unknown speculative mode: {self.mode}")

    @property
    def decode_rows_total(self) -> int:
        return self.n_parallel * self.decode_rows_per_sequence


def check_decode_capacity(shape: DecodeShape) -> None:
    """Check the limits used by llama_context::decode/output_reserve.

    DFlash/DSpark builds one combined batch for all drafting sequences and
    flags every row as an output.  Therefore both total and per-sequence
    output limits must cover the actual rows, and n_batch must cover the
    combined batch.  n_ubatch may split the batch, so it only has to be a
    valid positive sub-batch here.
    """

    for name in (
        "n_max",
        "block_size",
        "n_parallel",
        "n_batch",
        "n_ubatch",
        "n_outputs_max",
        "n_outputs_max_per_seq",
    ):
        if getattr(shape, name) <= 0:
            raise InvariantError(f"{name} must be positive")
    if shape.n_ubatch > shape.n_batch:
        raise InvariantError("n_ubatch cannot exceed n_batch")

    rows = shape.decode_rows_per_sequence
    total = shape.decode_rows_total
    if shape.n_batch < total:
        raise InvariantError(f"n_batch={shape.n_batch} < combined decode rows={total}")
    if shape.n_outputs_max_per_seq < rows:
        raise InvariantError(
            f"per-sequence output capacity {shape.n_outputs_max_per_seq} < rows {rows}"
        )
    if shape.n_outputs_max < total:
        raise InvariantError(f"total output capacity {shape.n_outputs_max} < rows {total}")


def check_target_capacity(
    *, proposal_k: int, n_parallel: int, n_batch: int, n_outputs_max: int, n_outputs_max_per_seq: int
) -> None:
    """Check the separate target verify shape: anchor plus proposal prefix."""

    if proposal_k <= 0 or n_parallel <= 0:
        raise InvariantError("proposal_k and n_parallel must be positive")
    rows = 1 + proposal_k
    total = n_parallel * rows
    if n_batch < total:
        raise InvariantError(f"target n_batch={n_batch} < verify rows={total}")
    if n_outputs_max_per_seq < rows:
        raise InvariantError(f"target per-seq capacity {n_outputs_max_per_seq} < verify rows {rows}")
    if n_outputs_max < total:
        raise InvariantError(f"target total capacity {n_outputs_max} < verify rows {total}")


class KVState:
    """Minimal one-sequence suffix-removal model.

    The model intentionally exposes the boolean return and postcondition that
    native callers must check after llama_memory_seq_rm.  It does not emulate
    a backend KV implementation or claim that every llama memory type accepts
    suffix removal.
    """

    def __init__(self) -> None:
        self.positions: set[int] = set()

    def append(self, start: int, count: int) -> None:
        if count < 0:
            raise InvariantError("count must be non-negative")
        expected = list(range(start, start + count))
        if any(pos in self.positions for pos in expected):
            raise InvariantError("append overlaps an existing position")
        self.positions.update(expected)

    def seq_rm(self, p0: int, p1: int | None = None, *, supported: bool = True) -> bool:
        if not supported:
            return False
        end = float("inf") if p1 is None or p1 < 0 else p1
        self.positions = {pos for pos in self.positions if not (p0 <= pos <= end)}
        return True

    def max_position(self) -> int:
        return max(self.positions) if self.positions else -1

    def require_suffix_removed(self, p0: int, expected_last: int) -> None:
        if self.max_position() != expected_last:
            raise InvariantError(
                f"suffix cleanup postcondition max={self.max_position()} expected={expected_last}"
            )
        if any(pos >= p0 for pos in self.positions):
            raise InvariantError(f"stale KV position >= cleanup boundary {p0}")


def cleanup_after_prefix(
    state: KVState, *, block_start: int, generated_count: int, accepted_count: int
) -> int:
    """Retain the committed prefix and remove all rejected draft suffix rows."""

    if generated_count < 0 or not 0 <= accepted_count <= generated_count:
        raise InvariantError("accepted prefix is outside generated block")
    remove_from = block_start + accepted_count
    ok = state.seq_rm(remove_from, -1)
    if not ok:
        raise InvariantError("memory implementation rejected suffix removal")
    expected_last = remove_from - 1
    state.require_suffix_removed(remove_from, expected_last)
    return expected_last


def validate_eos_block(tokens: list[int], eos_id: int) -> int | None:
    """Return the first EOS index, rejecting tokens after EOS."""

    try:
        eos_index = tokens.index(eos_id)
    except ValueError:
        return None
    if any(token != eos_id for token in tokens[eos_index + 1 :]):
        raise InvariantError("draft block contains token rows after EOS")
    return eos_index


def run_reference_tests() -> list[dict[str, Any]]:
    """Run deterministic CPU-only tests and return compact case records."""

    records: list[dict[str, Any]] = []

    def case(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - report path
            records.append({"name": name, "status": "fail", "error": str(exc)})
        else:
            records.append({"name": name, "status": "pass"})

    # This is the previously rejected shape: seven full rows, but constructor
    # capacity derived from n_max + 1 = 4.  It must fail before any model load.
    def rejected_constructor_capacity() -> None:
        shape = DecodeShape("dspark", 3, 7, 1, 256, 256, 4, 4, True)
        try:
            check_decode_capacity(shape)
        except InvariantError as exc:
            if "per-sequence" not in str(exc):
                raise
            return
        raise InvariantError("unsafe full-block shape was accepted")

    case("rejected_full_block_with_nmax_plus_one_capacity", rejected_constructor_capacity)

    def accepted_full_block_capacity() -> None:
        check_decode_capacity(DecodeShape("dspark", 3, 7, 1, 256, 256, 7, 7, True))

    case("full_block_capacity_seven_rows", accepted_full_block_capacity)

    def accepted_parallel_full_block_capacity() -> None:
        check_decode_capacity(DecodeShape("dspark", 3, 7, 2, 14, 8, 14, 7, True))

    case("full_block_capacity_scales_with_parallel_sequences", accepted_parallel_full_block_capacity)

    def prefix_does_not_shrink_draft_shape() -> None:
        full = DecodeShape("dspark", 3, 7, 1, 7, 4, 7, 7, True)
        if full.decode_rows_per_sequence != 7:
            raise InvariantError("proposal prefix incorrectly changed full draft rows")
        check_target_capacity(proposal_k=3, n_parallel=1, n_batch=4, n_outputs_max=4, n_outputs_max_per_seq=4)

    case("proposal_prefix_has_separate_target_capacity", prefix_does_not_shrink_draft_shape)

    def pinned_dspark_nmax_shape() -> None:
        check_decode_capacity(DecodeShape("dspark", 3, 7, 1, 4, 2, 4, 4, False))

    case("pinned_dspark_nmax_shape", pinned_dspark_nmax_shape)

    def legacy_dflash_anchor_shape() -> None:
        check_decode_capacity(DecodeShape("dflash", 3, 7, 1, 4, 2, 4, 4, False))

    case("legacy_dflash_anchor_plus_nmax_shape", legacy_dflash_anchor_shape)

    def all_accept_and_all_reject_cycles() -> None:
        state = KVState()
        state.append(0, 8)
        block_start = 8
        state.append(block_start, 7)
        cleanup_after_prefix(state, block_start=block_start, generated_count=7, accepted_count=0)
        if state.max_position() != 7:
            raise InvariantError("full rejection left draft rows")
        state.append(block_start, 7)
        cleanup_after_prefix(state, block_start=block_start, generated_count=7, accepted_count=7)
        if state.max_position() != 14:
            raise InvariantError("full acceptance removed committed rows")

    case("kv_suffix_cleanup_repeated_reject_accept", all_accept_and_all_reject_cycles)

    def every_accepted_prefix() -> None:
        for accepted in range(8):
            state = KVState()
            state.append(0, 8)
            state.append(8, 7)
            cleanup_after_prefix(state, block_start=8, generated_count=7, accepted_count=accepted)
            if state.max_position() != 7 + accepted:
                raise InvariantError(f"prefix {accepted} retained wrong position")

    case("kv_cleanup_each_acceptance_prefix", every_accepted_prefix)

    def unsupported_memory_is_rejected() -> None:
        state = KVState()
        state.append(0, 8)
        state.append(8, 7)
        # A caller that ignores the false return would continue with stale
        # rows.  The wrapper used by this harness must turn it into a failure.
        original = state.seq_rm
        state.seq_rm = lambda p0, p1=None, *, supported=True: original(p0, p1, supported=False)  # type: ignore[method-assign]
        try:
            cleanup_after_prefix(state, block_start=8, generated_count=7, accepted_count=0)
        except InvariantError:
            return
        raise InvariantError("caller failed to reject unsupported suffix removal")

    case("kv_cleanup_requires_supported_memory", unsupported_memory_is_rejected)

    def eos_at_each_position() -> None:
        for eos_index in range(7):
            tokens = [100 + i for i in range(eos_index)] + [2]
            if validate_eos_block(tokens, 2) != eos_index:
                raise InvariantError("EOS index was not preserved")
            state = KVState()
            state.append(0, 8)
            state.append(8, len(tokens))
            # Accepting EOS retains the EOS row; any suffix after it is removed.
            cleanup_after_prefix(
                state,
                block_start=8,
                generated_count=len(tokens),
                accepted_count=len(tokens),
            )
            if state.max_position() != 8 + len(tokens) - 1:
                raise InvariantError("EOS row was incorrectly removed")

    case("eos_at_each_block_position", eos_at_each_position)

    def eos_suffix_rejected() -> None:
        try:
            validate_eos_block([101, 2, 103], 2)
        except InvariantError:
            return
        raise InvariantError("tokens after EOS were accepted")

    case("tokens_after_eos_rejected", eos_suffix_rejected)

    def stale_position_detected() -> None:
        state = KVState()
        state.append(0, 8)
        state.append(8, 7)
        # Simulate a buggy memory implementation that reports success but
        # leaves one rejected row.  The postcondition must catch it.
        if not state.seq_rm(11, -1):
            raise InvariantError("unexpected cleanup failure")
        state.positions.add(12)
        try:
            state.require_suffix_removed(11, 10)
        except InvariantError:
            return
        raise InvariantError("stale suffix position was not detected")

    case("kv_postcondition_detects_stale_row", stale_position_detected)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_for(text: str, needle: str) -> int | None:
    for index, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return index
    return None


def line_for_after(text: str, needle: str, minimum: int) -> int | None:
    for index, line in enumerate(text.splitlines(), 1):
        if index >= minimum and needle in line:
            return index
    return None


def source_audit(source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    for relative, expected_sha in PINNED_FILES.items():
        path = source_root / relative
        record: dict[str, Any] = {"path": str(path), "relative": relative}
        if not path.is_file():
            record.update({"status": "fail", "error": "missing"})
        else:
            actual = sha256_file(path)
            record.update({"status": "pass" if actual == expected_sha else "fail", "sha256": actual, "expected_sha256": expected_sha})
            texts[relative] = path.read_text(encoding="utf-8", errors="replace")
            if actual != expected_sha:
                record["error"] = "hash mismatch"
        files.append(record)

    anchors: dict[str, tuple[str, str]] = {
        "speculative.block_metadata": ("common/speculative.cpp", 'llama_model_meta_val_str(model_dft, "dflash.block_size"'),
        "speculative.dspark_capacity_bound": ("common/speculative.cpp", "const int32_t n_draft_max = is_dspark ? block_size : block_size - 1;"),
        "speculative.block_rows": ("common/speculative.cpp", "const int32_t n_block_tokens = n_draft + (is_dspark ? 0 : 1);"),
        "speculative.positioned_logits": ("common/speculative.cpp", "common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);"),
        "speculative.dspark_confidence": ("common/speculative.cpp", "llama_get_embeddings_nextn(ctx_dft)"),
        "speculative.dspark_sampling": ("common/speculative.cpp", "common_sampler_sample(smpl, ctx_dft, idx, true);"),
        "speculative.draft_pos_max_check": ("common/speculative.cpp", "llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id)"),
        "speculative.draft_suffix_cleanup": ("common/speculative.cpp", "llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);"),
        "context.per_sequence_output_guard": ("src/llama-context.cpp", "seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq"),
        "context.output_reserve_guard": ("src/llama-context.cpp", "if (output_reserve(n_outputs_all) < n_outputs_all)"),
        "context.scheduler_output_bound": ("src/llama-context.cpp", "std::min(n_tokens, cparams.n_outputs_max)"),
        "context.memory_suffix_api": ("src/llama-context.cpp", "return mem->seq_rm(seq_id, p0, p1);"),
        "context.header_decode": ("src/llama-context.h", "int decode(const llama_batch & batch_inp);"),
        "context.header_output_reserve": ("src/llama-context.h", "uint32_t output_reserve(int32_t n_outputs);"),
        "dflash.block_metadata": ("src/models/dflash.cpp", '"dflash.block_size"'),
        "dflash.equal_block_guard": ("src/models/dflash.cpp", "n_tok % n_blocks == 0"),
        "dflash.block_capacity_guard": ("src/models/dflash.cpp", "if (block_drafts > block_size)"),
        "dflash.markov_argmax": ("src/models/dflash.cpp", "prev = ggml_argmax(ctx0, col);"),
        "dflash_positions.use_rope_input": ("src/models/dflash.cpp", "ggml_rope_ext("),
    }
    for name, (relative, needle) in anchors.items():
        text = texts.get(relative, "")
        # The first nextn use is an unrelated deferred-boundary path.  Pin the
        # DSpark confidence use to the decode section after constructor setup.
        line = line_for_after(text, needle, 1100) if name == "speculative.dspark_confidence" else line_for(text, needle)
        evidence.append({"name": name, "relative": relative, "line": line, "needle": needle, "status": "pass" if line else "fail"})

    # There is no src/models/*dspark* file in this pinned tree.  DSpark is a
    # mode of the DFlash model implementation, identified by its Markov head.
    dspark_glob = sorted((source_root / "src/models").glob("*dspark*")) if (source_root / "src/models").is_dir() else []
    evidence.append({
        "name": "dflash_is_pinned_dspark_implementation",
        "status": "pass" if "build_dspark_markov_head" in texts.get("src/models/dflash.cpp", "") else "fail",
        "detail": "no src/models/*dspark* file; DSpark mode is implemented in src/models/dflash.cpp",
        "matching_paths": [str(path) for path in dspark_glob],
    })
    return files, evidence


def git_commit(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def inspect_candidate(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        record.update({"status": "pending", "detail": "response file does not exist"})
        return record
    size = path.stat().st_size
    record["size_bytes"] = size
    if size == 0:
        record.update({"status": "pending", "detail": "response.json exists but is empty; external response not ready"})
        return record
    record["sha256"] = sha256_file(path)
    if size > 2 * 1024 * 1024:
        record.update({"status": "pending", "detail": "candidate exceeds bounded review read limit; metadata only"})
        return record
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, dict):
        record["top_level_keys"] = sorted(str(key) for key in value.keys())[:80]
        parts: list[str] = []
        for key in ("diff", "patch", "response", "content", "output", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                parts.append(candidate)
        review_text = "\n".join(parts)
    elif isinstance(value, str):
        review_text = value
    else:
        review_text = raw
    review_text = review_text[:256 * 1024]
    required_groups = {
        "capacity": ("n_outputs_max", "n_outputs_max_per_seq", "output_reserve", "block_size", "n_batch"),
        "positions": ("n + i", "seq_rm", "seq_pos_max", "pos_max"),
        "eos": ("EOS", "eos", "eos_id"),
        "cleanup": ("rejected", "suffix", "restore", "checkpoint"),
    }
    record["keyword_groups"] = {
        name: [needle for needle in needles if needle in review_text]
        for name, needles in required_groups.items()
    }
    record["status"] = "reviewed_static_only"
    record["detail"] = "candidate is preserved; this harness does not apply or launch an untrusted patch"
    return record


def run(source_root: Path, candidate: Path, report_path: Path) -> int:
    files, evidence = source_audit(source_root)
    reference = run_reference_tests()
    commit = git_commit(source_root)
    candidate_record = inspect_candidate(candidate)
    static_ok = all(item["status"] == "pass" for item in files + evidence)
    reference_ok = all(item["status"] == "pass" for item in reference)
    overall = "pass_reference_harness_pending_candidate" if static_ok and reference_ok and candidate_record["status"] == "pending" else (
        "pass_reference_harness_candidate_static_reviewed" if static_ok and reference_ok and candidate_record["status"] == "reviewed_static_only" else "fail"
    )
    report = {
        "schema": "RUN-06-opus09-validation-v1",
        "status": overall,
        "scope": "CPU-only static source map and reference shape/KV/EOS invariants; no framework, model, GPU, server, network, or source edit",
        "source": {"root": str(source_root), "pinned_commit": PINNED_COMMIT, "observed_commit": commit, "commit_match": commit == PINNED_COMMIT},
        "files": files,
        "source_evidence": evidence,
        "reference_tests": reference,
        "candidate": candidate_record,
        "limits": [
            "Reference capacities model the output rows and limits visible in the pinned source; they do not allocate a ggml graph or invoke llama_context::decode.",
            "KVState checks the required seq_rm boolean/postcondition and suffix semantics for one sequence; it does not prove DFlash/ISWA or DSV4 backend support.",
            "EOS checks are token-row invariants and do not prove native sampler or target verification behavior.",
            "An empty Opus09 response was not reviewed as a patch and no candidate was applied.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "reference_pass": reference_ok, "static_pass": static_ok, "candidate": candidate_record["status"], "report": str(report_path)}, sort_keys=True))
    return 0 if overall != "fail" else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453"))
    parser.add_argument("--candidate", type=Path, default=Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-hillclimb/opus09-dspark-repair-run/response.json"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("validation-result.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args.source_root, args.candidate, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
