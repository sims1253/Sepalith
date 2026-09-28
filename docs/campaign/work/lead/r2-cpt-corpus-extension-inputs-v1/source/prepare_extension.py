#!/usr/bin/env python3
"""Prepare a lossless CPT corpus extension and its cursor-bound draw plan.

This packet separates two operations which were conflated by the earlier
augmentation helper:

* the recovery documents are reblocked to context 16,384 without reading or
  retokenizing the base corpus; and
* a later, root-supplied checkpoint binds a combined row stream and schedule.

The schedule keeps the complete original unique prefix, removes only the old
alignment tail, appends each new row once, and adds one final alignment tail.
That shape is accepted by the existing streaming cache contract while a
resumed trainer can keep its original draw cursor.  No schedule is emitted
without an explicit checkpoint cursor and a byte hash for the combined rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_INPUT = "sepalith.cpt.lossless-rechunk-input.v1"
SCHEMA_EXTENSION = "sepalith.dat10.cpt-corpus-extension-schedule.v1"
SCHEMA_PLAN = "sepalith.dat10.cpt-corpus-extension-input-plan.v1"
RESULT_SCHEMA = "sepalith.cpt.lossless-rechunk-result.v1"
CONTEXT = 16384
BATCH = 16
BOS = 0
EOS = 1
VOCAB_SIZE = 130560
CHUNK_SIZE = 2048
FRONTIER_GROUPS = 81
FRONTIER_DOCUMENTS = 2003
RECOVERY_DOCUMENTS = 1998
RECOVERY_ROWS = 6340
RECOVERY_CODE_TOKENS = 10699204
BUILDER_SHA256 = "b37ae0fe27015a6596e1e7b8750c0ab77daed4b8375e10603603407258393fc2"
FRONTIER_SHA256 = "4708d69f53e04c48754e48095497eb1c281acac998dd099620b8fae293a1019d"
RAW_CHUNKS_SHA256 = "84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab"
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
BASE_SCHEDULE_SHA256 = "78bc2f3ec17ece7ad56a6dd79d4a1369d58c2329b5fb6a282d5a613135517937"
BASE_CACHE_MANIFEST_SHA256 = "ac17fe1ced73efe37767bf28619b47c120f57be121c4648a1bf5fb0c426084ec"
BASE_ROWS_SHA256 = "c7c2bbcd64827cc2a06eca0dd4f3b1c404226f18019df61bf3d24a49e0b351b7"
BASE_ROWS_COUNT = 183084
BASE_DOCUMENTS = 177190
BASE_REPLAY_COUNT = 4


class ExtensionError(ValueError):
    """A fail-closed extension input or schedule violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtensionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def list_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(canonical(list(values))).hexdigest()


def pin(path: Path, *, expected_sha256: str | None = None, allow_symlink: bool = False) -> dict[str, Any]:
    require(path.is_file(), f"file_missing:{path}")
    if not allow_symlink:
        require(not path.is_symlink(), f"file_symlink:{path}")
    actual = sha256(path)
    require(expected_sha256 is None or actual == expected_sha256, f"file_hash_mismatch:{path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    require(not path.exists() and not path.is_symlink(), f"output_must_be_fresh:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return pin(path)


def replace_json(path: Path, value: Any) -> dict[str, Any]:
    """Atomically replace a fresh, unadmitted metadata file."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"replace_target_invalid:{path}")
    temporary = path.with_name(f".{path.name}.replace-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return pin(path)


def _regular_input(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    return pin(path, expected_sha256=expected_sha)


def _group_number(path: Path) -> int:
    match = re.match(r"^(\d{6})-", path.name)
    require(match is not None, f"recovery_group_name_invalid:{path.name}")
    return int(match.group(1))


def make_recovery_input_manifest(
    recovery_root: Path,
    output: Path,
    *,
    raw_chunk_contract: Path,
    frontier: Path,
    tokenizer: Path,
) -> dict[str, Any]:
    """Create the input manifest consumed by the reviewed lossless rechunker.

    Only group manifests and file metadata are read here.  The actual CPT row
    files are opened later by the pinned rechunker, which independently checks
    every row and document span.
    """
    recovery_root = Path(recovery_root)
    output = Path(output)
    require(recovery_root.is_dir(), f"recovery_root_missing:{recovery_root}")
    top_manifest_path = recovery_root / "manifest.json"
    progress_path = recovery_root / "progress.json"
    terminal_path = recovery_root.parent / f"{recovery_root.name}-root.terminal.json"
    top_manifest_pin = pin(top_manifest_path)
    progress_pin = pin(progress_path)
    terminal_pin = pin(terminal_path)
    top = json.loads(top_manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    require(top.get("schema") == 1 and top.get("status") == "CPU_materialized_candidate_not_training_admission", "recovery_manifest_status_invalid")
    require(top.get("builder", {}).get("sha256") == BUILDER_SHA256, "recovery_builder_identity_invalid")
    require(top.get("frontier", {}).get("sha256") == FRONTIER_SHA256, "recovery_frontier_identity_invalid")
    require(top.get("chunk_size") == CHUNK_SIZE and top.get("training_admission") is False and top.get("source_policy", {}).get("target_truncation") is False and top.get("truncation") in (None, False), "recovery_geometry_policy_invalid")
    require(top.get("groups_completed") == FRONTIER_GROUPS and top.get("frontier_groups") == FRONTIER_GROUPS, "recovery_group_completion_invalid")
    require(progress.get("status") == "complete_candidate" and progress.get("groups_completed") == FRONTIER_GROUPS and progress.get("groups_remaining") == 0, "recovery_progress_not_terminal")
    require(terminal.get("exit_code") == 0, "recovery_root_terminal_not_successful")
    counts = top.get("counts", {})
    require(counts.get("documents") == RECOVERY_DOCUMENTS and counts.get("rows") == RECOVERY_ROWS and counts.get("code_tokens") == RECOVERY_CODE_TOKENS, "recovery_top_counts_invalid")

    frontier_pin = pin(Path(frontier), expected_sha256=FRONTIER_SHA256)
    raw_pin = pin(Path(raw_chunk_contract), expected_sha256=RAW_CHUNKS_SHA256)
    tokenizer_pin = pin(Path(tokenizer), expected_sha256=TOKENIZER_SHA256)

    group_root = recovery_root / "groups"
    groups = sorted((path for path in group_root.iterdir() if path.is_dir() and not path.name.startswith(".")), key=_group_number)
    require(len(groups) == FRONTIER_GROUPS, f"recovery_group_directory_count:{len(groups)}")
    inputs: list[dict[str, Any]] = []
    total = {"rows": 0, "documents": 0, "payload_tokens": 0}
    seen_ordinals: list[int] = []
    for group_dir in groups:
        group_manifest_path = group_dir / "manifest.json"
        group_pin = pin(group_manifest_path)
        group = json.loads(group_manifest_path.read_text(encoding="utf-8"))
        require(group.get("schema") == 1 and group.get("status") == "complete_candidate", f"group_manifest_status_invalid:{group_dir.name}")
        require(group.get("builder_sha256") == BUILDER_SHA256 and group.get("frontier_sha256") == FRONTIER_SHA256, f"group_manifest_source_invalid:{group_dir.name}")
        require(group.get("chunk_size") == CHUNK_SIZE and group.get("truncation") is False, f"group_manifest_geometry_invalid:{group_dir.name}")
        artifact = group.get("artifacts", {}).get("cpt_train.jsonl")
        group_counts = group.get("counts", {})
        ordinals = group.get("frontier_ordinals")
        require(isinstance(artifact, dict) and isinstance(ordinals, list) and ordinals, f"group_manifest_artifacts_invalid:{group_dir.name}")
        require(all(type(value) is int for value in ordinals), f"group_manifest_ordinals_invalid:{group_dir.name}")
        seen_ordinals.extend(ordinals)
        row_path = group_dir / "cpt_train.jsonl"
        row_pin = pin(row_path, expected_sha256=artifact.get("sha256"))
        require(row_pin["bytes"] == artifact.get("bytes"), f"group_row_bytes_mismatch:{group_dir.name}")
        item = {
            "label": f"recovery-{group_dir.name}",
            "path": row_pin["path"],
            "bytes": row_pin["bytes"],
            "sha256": row_pin["sha256"],
            "rows": group_counts.get("rows"),
            "documents": group_counts.get("documents"),
            "payload_tokens": group_counts.get("code_tokens"),
            "group_id": group.get("group_id"),
            "frontier_ordinals": ordinals,
            "group_manifest": group_pin,
        }
        require(all(type(item[key]) is int and item[key] >= 0 for key in ("rows", "documents", "payload_tokens")), f"group_counts_invalid:{group_dir.name}")
        inputs.append(item)
        for key in total:
            total[key] += item[key]
    # The frontier has one entry per source path, not per emitted document.  It
    # therefore has 2,003 ordinals even though only 1,998 documents survived
    # terminal source deduplication.  The top-level counts are the authoritative
    # emitted document/row/token denominator.
    require(sorted(seen_ordinals) == list(range(1, FRONTIER_DOCUMENTS + 1)), "group_ordinal_accounting_invalid")
    require(total == {"rows": RECOVERY_ROWS, "documents": RECOVERY_DOCUMENTS, "payload_tokens": RECOVERY_CODE_TOKENS}, f"recovery_input_totals_invalid:{total}")
    require(len(set(seen_ordinals)) == len(seen_ordinals), "recovery_frontier_ordinals_duplicate")

    manifest = {
        "schema": SCHEMA_INPUT,
        "status": "complete_candidate_input_pending_lossless_rechunk",
        "context_sizes": [CONTEXT],
        "raw_chunks": {**raw_pin, "bos": BOS, "eos": EOS, "source_chunk_size": CHUNK_SIZE},
        "inputs": inputs,
        "expected_totals": total,
        "source": {
            "recovery_root": str(recovery_root.resolve()),
            "recovery_manifest": top_manifest_pin,
            "recovery_progress": progress_pin,
            "recovery_root_terminal": terminal_pin,
            "frontier": frontier_pin,
            "tokenizer": tokenizer_pin,
            "tokenizer_revision": TOKENIZER_REVISION,
            "builder_sha256": BUILDER_SHA256,
            "payload_opened_field_is_not_used": True,
        },
        "training_admission": False,
    }
    return {"manifest": manifest, "pin": atomic_json(output, manifest), "counts": total}


def _read_jsonl_row_ids(path: Path, expected_sha256: str | None = None) -> tuple[list[str], dict[str, Any]]:
    before = path.stat()
    digest = hashlib.sha256()
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            require(raw.endswith(b"\n"), f"rows_final_line_missing_newline:{path}:{line_number}")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ExtensionError(f"rows_json_invalid:{path}:{line_number}") from exc
            require(isinstance(row, dict), f"row_not_object:{path}:{line_number}")
            require(row.get("schema") == 1 and row.get("cpt_partition") == "cpt_train", f"row_partition_invalid:{path}:{line_number}")
            row_id = row.get("row_id")
            require(isinstance(row_id, str) and row_id and row_id not in seen, f"row_id_invalid_or_duplicate:{path}:{line_number}")
            ids.append(row_id)
            seen.add(row_id)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"rows_changed_during_read:{path}")
    actual = digest.hexdigest()
    require(expected_sha256 is None or actual == expected_sha256, f"rows_hash_mismatch:{path}")
    return ids, {"path": str(path.resolve()), "bytes": after.st_size, "sha256": actual, "rows": len(ids)}


def normalize_recovery_rows(input_manifest: Path, output_rows: Path, output_manifest: Path) -> dict[str, Any]:
    """Add the reviewed rechunker's source-range aliases without changing data.

    The recovery producer emitted ``source_token_start/end``.  The pinned
    closure rechunker also requires the equivalent ``token_start/end`` aliases
    used by its raw-row contract.  This derived stream changes only those two
    metadata keys; token IDs, labels, document IDs, and byte-level source
    identity remain bound to the producer artifacts.
    """
    input_manifest = Path(input_manifest)
    output_rows = Path(output_rows)
    output_manifest = Path(output_manifest)
    input_pin = pin(input_manifest)
    require(not output_rows.exists() and not output_rows.is_symlink(), f"normalized_rows_output_must_be_fresh:{output_rows}")
    require(not output_manifest.exists() and not output_manifest.is_symlink(), f"normalized_manifest_output_must_be_fresh:{output_manifest}")
    original = json.loads(input_manifest.read_text(encoding="utf-8"))
    require(original.get("schema") == SCHEMA_INPUT, "normalized_input_manifest_schema_invalid")
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_rows.with_name(f".{output_rows.name}.partial-{os.getpid()}")
    digest = hashlib.sha256()
    rows = 0
    documents: set[str] = set()
    current_document: str | None = None
    closed_documents: set[str] = set()
    payload_tokens = 0
    try:
        with temporary.open("wb") as target:
            for item in original.get("inputs", []):
                source = Path(item["path"])
                source_pin = pin(source, expected_sha256=item.get("sha256"))
                before = source.stat()
                local_rows = 0
                local_documents: set[str] = set()
                with source.open("rb", buffering=8 * 1024 * 1024) as stream:
                    for line_number, raw in enumerate(stream, 1):
                        require(raw.endswith(b"\n"), f"recovery_row_missing_newline:{source}:{line_number}")
                        try:
                            row = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise ExtensionError(f"recovery_row_json_invalid:{source}:{line_number}") from exc
                        require(isinstance(row, dict), f"recovery_row_not_object:{source}:{line_number}")
                        require(row.get("schema") == 1 and row.get("cpt_partition") == "cpt_train", f"recovery_row_contract_invalid:{source}:{line_number}")
                        document_id = row.get("document_id")
                        require(isinstance(document_id, str) and document_id, f"recovery_document_id_invalid:{source}:{line_number}")
                        start = row.get("source_token_start")
                        end = row.get("source_token_end")
                        require(type(start) is int and type(end) is int and 0 <= start < end, f"recovery_source_span_invalid:{source}:{line_number}")
                        if "token_start" in row:
                            require(row["token_start"] == start, f"recovery_token_start_alias_conflict:{source}:{line_number}")
                        if "token_end" in row:
                            require(row["token_end"] == end, f"recovery_token_end_alias_conflict:{source}:{line_number}")
                        row["token_start"] = start
                        row["token_end"] = end
                        encoded = canonical(row) + b"\n"
                        target.write(encoded)
                        digest.update(encoded)
                        rows += 1
                        local_rows += 1
                        payload_tokens += end - start
                        if current_document is None:
                            current_document = document_id
                        elif document_id != current_document:
                            closed_documents.add(current_document)
                            current_document = document_id
                        require(document_id not in closed_documents, f"recovery_document_reappears:{document_id}")
                        local_documents.add(document_id)
                        documents.add(document_id)
                after = source.stat()
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"recovery_source_changed_during_normalization:{source}")
                require(source_pin["sha256"] == item.get("sha256") and source_pin["bytes"] == item.get("bytes"), f"recovery_source_pin_changed:{source}")
                require(local_rows == item.get("rows") and len(local_documents) == item.get("documents"), f"recovery_input_counts_changed:{source}")
            target.flush()
            os.fsync(target.fileno())
        if current_document is not None:
            closed_documents.add(current_document)
        require(rows == original.get("expected_totals", {}).get("rows"), "normalized_row_count_differs")
        require(len(documents) == original.get("expected_totals", {}).get("documents"), "normalized_document_count_differs")
        require(payload_tokens == original.get("expected_totals", {}).get("payload_tokens"), "normalized_payload_count_differs")
        os.replace(temporary, output_rows)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_pin = pin(output_rows)
    normalized = {
        "schema": SCHEMA_INPUT,
        "status": "complete_candidate_input_pending_lossless_rechunk",
        "context_sizes": original.get("context_sizes"),
        "raw_chunks": original.get("raw_chunks"),
        "inputs": [{
            "label": "recovery-all-groups-source-range-aliases",
            **output_pin,
            "rows": rows,
            "documents": len(documents),
            "payload_tokens": payload_tokens,
        }],
        "expected_totals": {"rows": rows, "documents": len(documents), "payload_tokens": payload_tokens},
        "source": {
            "upstream_input_manifest": input_pin,
            "transformation": "metadata-only source_token_start/end aliases; token IDs/labels/source identities unchanged",
            "source_group_artifacts": len(original.get("inputs", [])),
            "tokenizer": original.get("source", {}).get("tokenizer"),
            "tokenizer_revision": original.get("source", {}).get("tokenizer_revision"),
            "builder_sha256": original.get("source", {}).get("builder_sha256"),
        },
        "training_admission": False,
    }
    manifest_pin = atomic_json(output_manifest, normalized)
    return {"rows": rows, "documents": len(documents), "payload_tokens": payload_tokens, "rows_artifact": output_pin, "manifest": manifest_pin}


def concat_rows(base_rows: Path, recovery_rows: Path, output: Path, *, base_sha256: str, recovery_sha256: str) -> dict[str, Any]:
    """Create the fresh combined JSONL stream without changing either input."""
    base_rows = Path(base_rows)
    recovery_rows = Path(recovery_rows)
    output = Path(output)
    require(str(output.resolve()).startswith("/mnt/e/"), "combined_rows_must_be_on_E")
    require(not output.exists() and not output.is_symlink(), f"combined_rows_output_must_be_fresh:{output}")
    base_pin = pin(base_rows, expected_sha256=base_sha256)
    recovery_pin = pin(recovery_rows, expected_sha256=recovery_sha256)
    require(base_rows.stat().st_size > 0 and recovery_rows.stat().st_size > 0, "combined_rows_input_empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    require(not temporary.exists(), f"combined_rows_temp_exists:{temporary}")
    digest = hashlib.sha256()
    lines = 0
    try:
        with temporary.open("wb") as target:
            for source_path, expected in ((base_rows, base_pin), (recovery_rows, recovery_pin)):
                before = source_path.stat()
                with source_path.open("rb", buffering=8 * 1024 * 1024) as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        target.write(block)
                        digest.update(block)
                        lines += block.count(b"\n")
                after = source_path.stat()
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"combined_source_changed:{source_path}")
                require(sha256(source_path) == expected["sha256"], f"combined_source_hash_changed:{source_path}")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": lines,
        "base": base_pin,
        "recovery": recovery_pin,
    }


def _load_checkpoint(path: Path, *, base_schedule_sha256: str, base_unique_count: int) -> dict[str, Any]:
    checkpoint_pin = pin(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "checkpoint_not_object")
    step = value.get("checkpoint_step", value.get("global_step", value.get("step")))
    cursor = value.get("source_cursor", value.get("draw_cursor"))
    checkpoint_id = value.get("checkpoint_id", value.get("id"))
    schedule_sha = value.get("source_schedule_sha256", value.get("schedule_sha256"))
    offset = value.get("global_optimizer_step_offset", value.get("source_global_optimizer_step_offset"))
    require(type(step) is int and step >= 0, "checkpoint_requires_nonnegative_step")
    require(type(cursor) is int and 0 <= cursor <= base_unique_count and cursor % BATCH == 0, "checkpoint_cursor_must_be_aligned_and_within_unique_prefix")
    require(isinstance(checkpoint_id, str) and checkpoint_id, "checkpoint_requires_id")
    require(isinstance(schedule_sha, str) and schedule_sha == base_schedule_sha256, "checkpoint_must_bind_base_schedule_sha256")
    require(type(offset) is int and offset >= 0, "checkpoint_requires_explicit_global_optimizer_step_offset")
    cursor_kind = value.get("source_cursor_kind", value.get("cursor_kind"))
    if cursor_kind is not None:
        require(cursor_kind == "draw_position_exclusive", "checkpoint_cursor_kind_invalid")
    checkpoint_record: dict[str, Any] = {}
    checkpoint_dir_value = value.get("checkpoint_path", value.get("checkpoint"))
    checkpoint_manifest_sha = value.get("checkpoint_manifest_sha256")
    if checkpoint_dir_value is not None or checkpoint_manifest_sha is not None:
        require(isinstance(checkpoint_dir_value, str) and checkpoint_dir_value, "checkpoint_path_required_with_manifest_binding")
        require(isinstance(checkpoint_manifest_sha, str) and len(checkpoint_manifest_sha) == 64, "checkpoint_manifest_sha256_required")
        checkpoint_dir = Path(checkpoint_dir_value)
        require(checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink(), "checkpoint_directory_invalid")
        manifest_path = checkpoint_dir / "campaign-manifest.json"
        manifest_pin = pin(manifest_path, expected_sha256=checkpoint_manifest_sha)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("checkpoint_kind") == "full_weights" and manifest.get("full") is True and manifest.get("step") == step, "checkpoint_manifest_state_invalid")
        files = manifest.get("files", {})
        for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "tokenizer.json"):
            require(isinstance(files.get(name), dict) and type(files[name].get("bytes")) is int and files[name]["bytes"] > 0 and isinstance(files[name].get("sha256"), str) and len(files[name]["sha256"]) == 64, f"checkpoint_manifest_file_missing:{name}")
        state_path = checkpoint_dir / "campaign-state.json"
        state_pin = pin(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        sampler = state.get("sampler", {})
        require(state.get("step") == step and sampler.get("global_step") == step and sampler.get("cursor") == cursor and sampler.get("stage_cursor") == cursor and sampler.get("global_optimizer_step_offset") == offset, "checkpoint_sampler_state_differs")
        require(sampler.get("draw_schedule_sha256") == base_schedule_sha256, "checkpoint_sampler_schedule_differs")
        checkpoint_record = {
            "path": str(checkpoint_dir.resolve()),
            "manifest": manifest_pin,
            "campaign_state": state_pin,
            "files_from_manifest": {name: files[name] for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "tokenizer.json")},
        }
    return {
        "path": str(path.resolve()),
        "bytes": checkpoint_pin["bytes"],
        "sha256": checkpoint_pin["sha256"],
        "checkpoint_step": step,
        "source_cursor": cursor,
        "checkpoint_id": checkpoint_id,
        "source_schedule_sha256": schedule_sha,
        "global_optimizer_step_offset": offset,
        "source_cursor_kind": "draw_position_exclusive",
        "checkpoint_artifacts": checkpoint_record,
        "raw": value,
    }


def make_extension_schedule(
    base_schedule: Path,
    base_cache_manifest: Path,
    base_rows: Path,
    recovery_result: Path,
    recovery_rows: Path,
    combined_rows: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind the exact base prefix, recovery rows, combined hash, and cursor."""
    base_schedule_pin = pin(Path(base_schedule), expected_sha256=BASE_SCHEDULE_SHA256)
    base_cache_pin = pin(Path(base_cache_manifest), expected_sha256=BASE_CACHE_MANIFEST_SHA256)
    base_rows_pin = pin(Path(base_rows), expected_sha256=BASE_ROWS_SHA256)
    recovery_result_pin = pin(Path(recovery_result))
    recovery_rows_pin = pin(Path(recovery_rows))
    combined_pin = pin(Path(combined_rows))
    base = json.loads(Path(base_schedule).read_text(encoding="utf-8"))
    cache = json.loads(Path(base_cache_manifest).read_text(encoding="utf-8"))
    recovered = json.loads(Path(recovery_result).read_text(encoding="utf-8"))
    require(base.get("schema") == "sepalith.sft11.cpt-draw-schedule.v1", "base_schedule_schema_invalid")
    require(base.get("effective_batch") == BATCH and base.get("method") in ("one_pass_plus_named_replay_v1", "one_pass_source_order_plus_named_replay_v1"), "base_schedule_contract_invalid")
    base_ids = base.get("row_ids")
    replay_count = base.get("replay_count")
    require(isinstance(base_ids, list) and all(isinstance(value, str) and value for value in base_ids), "base_schedule_row_ids_invalid")
    require(type(replay_count) is int and replay_count >= 0 and base.get("replay_row_ids") == (base_ids[-replay_count:] if replay_count else []), "base_schedule_replay_tail_invalid")
    require(base.get("max_steps") * BATCH == len(base_ids) and base.get("token_rows_sha256") == cache.get("source", {}).get("rows", {}).get("sha256") == base_rows_pin["sha256"], "base_schedule_cache_identity_invalid")
    base_unique_count = len(base_ids) - replay_count
    base_unique_ids = base_ids[:base_unique_count]
    require(len(set(base_unique_ids)) == base_unique_count and replay_count == BASE_REPLAY_COUNT, "base_schedule_unique_prefix_invalid")
    require(cache.get("schema") == "sepalith.sft11.cpt-streaming-cache.v1" and cache.get("status") == "complete", "base_cache_manifest_invalid")
    require(cache.get("counts", {}).get("rows") == base_unique_count == BASE_ROWS_COUNT and cache.get("counts", {}).get("documents") == BASE_DOCUMENTS, "base_cache_counts_invalid")
    require(recovered.get("schema") == RESULT_SCHEMA and recovered.get("status") == "complete", "recovery_result_invalid")
    recovery_artifact = recovered.get("artifacts", {}).get("cpt_train_ctx16384.jsonl")
    recovery_stats = recovered.get("outputs", {}).get(str(CONTEXT), {})
    require(isinstance(recovery_artifact, dict) and recovery_artifact.get("sha256") == recovery_rows_pin["sha256"], "recovery_result_rows_identity_invalid")
    require(type(recovery_stats.get("rows")) is int and recovery_stats.get("rows") > 0, "recovery_result_row_count_invalid")
    new_ids, observed_recovery = _read_jsonl_row_ids(Path(recovery_rows), expected_sha256=recovery_artifact.get("sha256"))
    require(len(new_ids) == recovery_stats.get("rows"), "recovery_row_count_differs_from_result")
    require(not set(base_unique_ids).intersection(new_ids), "recovery_row_id_overlaps_base_schedule")
    require(combined_pin["bytes"] == base_rows_pin["bytes"] + recovery_rows_pin["bytes"], "combined_rows_byte_conservation_invalid")
    checkpoint_info = _load_checkpoint(Path(checkpoint), base_schedule_sha256=base_schedule_pin["sha256"], base_unique_count=base_unique_count)

    # The base unique order is immutable.  Only its old alignment tail is
    # removed; therefore every source draw at/after the explicit cursor through
    # the old unique prefix remains byte-for-byte in the same order.
    new_alignment_count = (-(base_unique_count + len(new_ids))) % BATCH
    new_alignment_ids = new_ids[:new_alignment_count]
    combined_ids = base_unique_ids + new_ids + new_alignment_ids
    require(len(combined_ids) % BATCH == 0, "extension_schedule_not_batch_aligned")
    require(combined_ids[:base_unique_count] == base_unique_ids, "base_unique_prefix_changed")
    require(combined_ids[base_unique_count:base_unique_count + len(new_ids)] == new_ids, "recovery_order_changed")
    require(len(set(combined_ids[:base_unique_count + len(new_ids)])) == base_unique_count + len(new_ids), "extension_unique_prefix_invalid")
    if new_alignment_count:
        require(combined_ids[-new_alignment_count:] == new_alignment_ids, "extension_alignment_tail_invalid")
    remaining_original = base_unique_ids[checkpoint_info["source_cursor"]:]
    schedule = {
        "schema": SCHEMA_EXTENSION,
        "status": "candidate_pending_root_data_and_stage_admission",
        "split_id": "cpt_train_corpus_extension_v1",
        "method": "preserve_base_unique_prefix_append_new_rows_plus_named_replay_v1",
        "seed": base.get("seed", 3407),
        "effective_batch": BATCH,
        "max_steps": len(combined_ids) // BATCH,
        "token_rows_sha256": combined_pin["sha256"],
        "row_ids": combined_ids,
        "replay_count": new_alignment_count,
        "replay_row_ids": new_alignment_ids,
        "base_schedule": {**base_schedule_pin, "unique_rows": base_unique_count, "removed_old_alignment_rows": replay_count},
        "base_cache_manifest": base_cache_pin,
        "base_rows": base_rows_pin,
        "recovery_rows": {**recovery_rows_pin, "rows": len(new_ids), "row_ids_sha256": list_sha256(new_ids)},
        "combined_rows": combined_pin,
        "source_checkpoint": checkpoint_info,
        "coverage": {
            "base_unique_rows": base_unique_count,
            "new_unique_rows": len(new_ids),
            "combined_unique_rows": base_unique_count + len(new_ids),
            # Keep the canonical names used by the streaming trainer and the
            # prefix-extension verifier alongside the more descriptive
            # accounting fields above.  These are derived from the same ID
            # vector, so they cannot drift independently.
            "unique_rows": base_unique_count + len(new_ids),
            "draws": len(combined_ids),
            "updates": len(combined_ids) // BATCH,
            "remaining_original_draws_at_cursor": len(remaining_original),
            "old_alignment_rows_removed": replay_count,
            "new_alignment_replay_rows": new_alignment_count,
            "first_unique_draw_position_exclusive": base_unique_count + len(new_ids),
            "all_new_rows_before_replay": True,
            "no_consumed_base_draw_replayed": True,
            "base_unique_order_preserved": True,
            "complete_documents_required_in_combined_cache": True,
        },
        "stage_transition": {
            "source_cursor_kind": "draw_position_exclusive",
            "source_cursor": checkpoint_info["source_cursor"],
            "global_optimizer_step_offset": checkpoint_info["global_optimizer_step_offset"],
            "checkpoint_step": checkpoint_info["checkpoint_step"],
            "optimizer_scheduler_rng_state": "must_be_loaded_from_the_same_explicit_full_checkpoint",
            "destination_cache_cursor_policy": "retain_original_schedule_position; do not reset or replay consumed prefix",
        },
        "training_admission": False,
    }
    result = atomic_json(Path(output), schedule)
    return {"schedule": schedule, "pin": result, "observed_recovery_rows": observed_recovery, "checkpoint": checkpoint_info}


def finalize_cache_schedule_compatibility(
    schedule: Path,
    output_schedule: Path,
    cache_manifest: Path,
) -> dict[str, Any]:
    """Add canonical verifier coverage fields and rebind a fresh cache.

    The running cache builder may have consumed the original schedule before
    this packet learned the prefix verifier's canonical coverage names.  This
    operation changes only equivalent JSON metadata: the ID vector, replay
    tail, and source-row hash must remain unchanged.  It then updates the
    fresh, unadmitted cache manifest's schedule identity without touching any
    cache payload.  Existing accepted caches are never valid targets.
    """
    schedule = Path(schedule)
    output_schedule = Path(output_schedule)
    cache_manifest = Path(cache_manifest)
    source_pin = pin(schedule)
    require(not output_schedule.exists() and not output_schedule.is_symlink(), f"compatibility_schedule_output_must_be_fresh:{output_schedule}")
    value = json.loads(schedule.read_text(encoding="utf-8"))
    require(value.get("schema") == SCHEMA_EXTENSION, "compatibility_schedule_schema_invalid")
    ids = value.get("row_ids")
    replay_count = value.get("replay_count")
    require(isinstance(ids, list) and ids and all(isinstance(row_id, str) and row_id for row_id in ids), "compatibility_schedule_ids_invalid")
    require(type(replay_count) is int and 0 <= replay_count <= 15, "compatibility_schedule_replay_count_invalid")
    unique_rows = len(ids) - replay_count
    unique_ids = ids[:unique_rows]
    replay_ids = ids[unique_rows:]
    require(len(set(unique_ids)) == unique_rows, "compatibility_schedule_unique_prefix_invalid")
    require(replay_ids == value.get("replay_row_ids", []), "compatibility_schedule_replay_tail_invalid")
    require(len(ids) % BATCH == 0 and value.get("max_steps") == len(ids) // BATCH, "compatibility_schedule_geometry_invalid")
    coverage = dict(value.get("coverage", {}))
    coverage.update({"unique_rows": unique_rows, "draws": len(ids), "updates": len(ids) // BATCH})
    value["coverage"] = coverage
    value["compatibility"] = {
        "verifier": "r2-cpt-prefix-extension-v1/prefix_extension_contract.py",
        "derived_from": source_pin,
        "payload_rows_unchanged": True,
        "training_admission": False,
    }
    schedule_pin = atomic_json(output_schedule, value)

    require(cache_manifest.is_file() and not cache_manifest.is_symlink(), f"cache_manifest_invalid:{cache_manifest}")
    manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "sepalith.sft11.cpt-streaming-cache.v1" and manifest.get("status") == "complete", "cache_manifest_not_complete")
    source = manifest.get("source", {})
    cache_schedule_sha = source.get("draw_schedule", {}).get("sha256")
    allowed_cache_schedule_shas = {source_pin["sha256"]}
    # A later runtime checkpoint rebind changes schedule metadata but not its
    # draw vector.  The cache may still carry the pre-rebind builder pin; bind
    # that predecessor here before publishing the final schedule pin.
    derived_from = value.get("runtime_checkpoint_rebind", {}).get("derived_from", {})
    if isinstance(derived_from, dict) and isinstance(derived_from.get("sha256"), str):
        allowed_cache_schedule_shas.add(derived_from["sha256"])
    require(cache_schedule_sha in allowed_cache_schedule_shas, "cache_manifest_old_schedule_identity_differs")
    require(source.get("rows", {}).get("sha256") == value.get("token_rows_sha256"), "cache_manifest_rows_identity_differs")
    require(manifest.get("counts", {}).get("rows") == unique_rows and manifest.get("counts", {}).get("draws") == len(ids), "cache_manifest_counts_differ")
    # Avoid a second multi-gigabyte read here; the builder has already hashed
    # every payload, and the root prefix verifier will rehash them before
    # admission.  Size checks catch an incomplete or replaced payload set.
    for name, record in manifest.get("files", {}).items():
        payload = cache_manifest.parent / name
        require(payload.is_file() and payload.stat().st_size == record.get("bytes"), f"cache_payload_size_invalid:{name}")
    previous_cache_schedule = dict(source.get("draw_schedule", {}))
    manifest["source"]["draw_schedule"] = schedule_pin
    manifest["source"]["schedule_metadata_rebind"] = {
        "previous": previous_cache_schedule,
        "input_schedule": source_pin,
        "reason": "equivalent coverage aliases; row IDs and cache draw ordinals unchanged",
        "payloads_rehashed_by_builder": True,
    }
    manifest_pin = replace_json(cache_manifest, manifest)
    return {"schedule": schedule_pin, "cache_manifest": manifest_pin, "previous_schedule": source_pin, "counts": {"unique_rows": unique_rows, "draws": len(ids), "updates": len(ids) // BATCH}}


def rebind_runtime_checkpoint(schedule: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Bind a later root-verified full checkpoint without changing draw IDs.

    Step 82 is only the preparation reference that allowed the first cache
    build.  A native continuation may reach a later verified checkpoint before
    the extension is admitted.  This command derives a new schedule metadata
    file from that exact binding and leaves the combined rows, ID order, and
    cache payload independent of the runtime checkpoint.
    """
    schedule = Path(schedule)
    checkpoint = Path(checkpoint)
    output = Path(output)
    schedule_pin = pin(schedule)
    value = json.loads(schedule.read_text(encoding="utf-8"))
    require(value.get("schema") == SCHEMA_EXTENSION, "runtime_schedule_schema_invalid")
    ids = value.get("row_ids")
    replay_count = value.get("replay_count")
    require(isinstance(ids, list) and ids and all(isinstance(row_id, str) and row_id for row_id in ids), "runtime_schedule_ids_invalid")
    require(type(replay_count) is int and 0 <= replay_count <= 15, "runtime_schedule_replay_count_invalid")
    unique_rows = len(ids) - replay_count
    require(len(set(ids[:unique_rows])) == unique_rows, "runtime_schedule_unique_prefix_invalid")
    require(ids[unique_rows:] == value.get("replay_row_ids", []), "runtime_schedule_replay_tail_invalid")
    base = value.get("base_schedule", {})
    base_schedule_sha = base.get("sha256")
    require(isinstance(base_schedule_sha, str) and len(base_schedule_sha) == 64, "runtime_schedule_base_pin_missing")
    checkpoint_info = _load_checkpoint(checkpoint, base_schedule_sha256=base_schedule_sha, base_unique_count=int(value.get("coverage", {}).get("base_unique_rows", unique_rows)))
    value["source_checkpoint"] = checkpoint_info
    value.setdefault("stage_transition", {}).update({
        "source_cursor_kind": "draw_position_exclusive",
        "source_cursor": checkpoint_info["source_cursor"],
        "global_optimizer_step_offset": checkpoint_info["global_optimizer_step_offset"],
        "checkpoint_step": checkpoint_info["checkpoint_step"],
    })
    value["runtime_checkpoint_rebind"] = {
        "derived_from": schedule_pin,
        "checkpoint_binding": checkpoint_info["path"],
        "training_admission": False,
        "row_ids_unchanged": True,
    }
    value.setdefault("coverage", {}).update({
        "unique_rows": unique_rows,
        "draws": len(ids),
        "updates": len(ids) // BATCH,
        "runtime_source_cursor": checkpoint_info["source_cursor"],
    })
    output_pin = atomic_json(output, value)
    return {"schedule": output_pin, "checkpoint": checkpoint_info, "previous_schedule": schedule_pin, "counts": {"unique_rows": unique_rows, "draws": len(ids), "updates": len(ids) // BATCH}}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("recovery-manifest")
    manifest.add_argument("--recovery-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--raw-chunk-contract", type=Path, required=True)
    manifest.add_argument("--frontier", type=Path, required=True)
    manifest.add_argument("--tokenizer", type=Path, required=True)
    concat = sub.add_parser("concat")
    concat.add_argument("--base-rows", type=Path, required=True)
    concat.add_argument("--recovery-rows", type=Path, required=True)
    concat.add_argument("--base-sha256", required=True)
    concat.add_argument("--recovery-sha256", required=True)
    concat.add_argument("--output", type=Path, required=True)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("--input-manifest", type=Path, required=True)
    normalize.add_argument("--output-rows", type=Path, required=True)
    normalize.add_argument("--output-manifest", type=Path, required=True)
    schedule = sub.add_parser("schedule")
    for name in ("base-schedule", "base-cache-manifest", "base-rows", "recovery-result", "recovery-rows", "combined-rows", "checkpoint", "output"):
        schedule.add_argument(f"--{name}", type=Path, required=True)
    compatibility = sub.add_parser("finalize-cache-compatibility")
    compatibility.add_argument("--schedule", type=Path, required=True)
    compatibility.add_argument("--output-schedule", type=Path, required=True)
    compatibility.add_argument("--cache-manifest", type=Path, required=True)
    runtime = sub.add_parser("rebind-runtime-checkpoint")
    runtime.add_argument("--schedule", type=Path, required=True)
    runtime.add_argument("--checkpoint", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "recovery-manifest":
        result = make_recovery_input_manifest(args.recovery_root, args.output, raw_chunk_contract=args.raw_chunk_contract, frontier=args.frontier, tokenizer=args.tokenizer)
    elif args.command == "normalize":
        result = normalize_recovery_rows(args.input_manifest, args.output_rows, args.output_manifest)
    elif args.command == "concat":
        result = concat_rows(args.base_rows, args.recovery_rows, args.output, base_sha256=args.base_sha256, recovery_sha256=args.recovery_sha256)
    elif args.command == "schedule":
        result = make_extension_schedule(args.base_schedule, args.base_cache_manifest, args.base_rows, args.recovery_result, args.recovery_rows, args.combined_rows, args.checkpoint, args.output)
    elif args.command == "finalize-cache-compatibility":
        result = finalize_cache_schedule_compatibility(args.schedule, args.output_schedule, args.cache_manifest)
    else:
        result = rebind_runtime_checkpoint(args.schedule, args.checkpoint, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtensionError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        raise SystemExit(3)
