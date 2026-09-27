"""Fail-closed raw-R causal-LM chunk and package-holdout contracts.

The canonical row has only the small training identity and a source provenance
object. The adapter also accepts the raw_cpt materializer's explicit derived
``labels``/``attention_mask`` fields. Every chunk has manual BOS 0 and EOS 1;
internal EOS labels are masked, while the actual document-final EOS is learned.
Source ownership spans are contiguous and non-overlapping. A continuation may
carry one prior code token, but that context token is explicitly unsupervised.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BOS_ID = 0
EOS_ID = 1
PAD_ID = 1
VOCAB_SIZE = 130560
DEFAULT_MAX_SEQUENCE_TOKENS = 2048
TOKENIZER_REVISION = "8dc5f6055b90fe4b9422340810b270b9569f37f3"
TOKENIZATION_POLICY = (
    "hf_add_special_tokens_false_split_special_tokens_true_manual_bos0_eos1"
)
ROW_FIELDS = frozenset(("id", "split", "package_id", "input_ids", "source"))
SOURCE_FIELDS = frozenset((
    "source_kind", "source_id", "group_id", "document_sha256",
    "token_stream_sha256", "document_token_count", "token_start", "token_end",
    "chunk_index", "chunk_count", "tokenizer_revision", "builder_id",
    "builder_sha256", "overlap_context_tokens",
))
MATERIALIZED_REQUIRED_FIELDS = frozenset((
    "schema", "row_id", "document_id", "package", "group_id", "cpt_partition",
    "source_path", "source_sha256", "chunk_index", "input_ids", "labels",
    "attention_mask", "source_token_start", "source_token_end",
    "overlap_context_tokens", "is_document_end", "supervised_tokens",
))
# The v2 packer may retain these aliases/identity fields for an independent
# source audit. Unknown fields still fail closed.
MATERIALIZED_OPTIONAL_FIELDS = frozenset((
    "token_start", "token_end", "document_token_count", "token_stream_sha256",
    "builder_id", "builder_sha256", "tokenizer_revision", "package_id",
))
_HEX64 = frozenset("0123456789abcdef")


class CptDataError(ValueError):
    """A raw causal-pretraining row or split violates its immutable contract."""


def _sha256_text(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value):
        raise CptDataError(f"{name} must be a lowercase SHA256 digest")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CptDataError(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CptDataError(f"{name} must be an integer >= {minimum}")
    return value


def canonical_token_stream_sha256(tokens: Sequence[int]) -> str:
    """Hash the exact JSON token sequence, without BOS/EOS or overlap."""
    if any(type(token) is not int for token in tokens):
        raise CptDataError("token stream contains a non-integer token")
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_ids(ids: Any, ident: str, *, max_sequence_tokens: int, vocab_size: int) -> list[int]:
    if not isinstance(ids, list) or not 3 <= len(ids) <= max_sequence_tokens:
        raise CptDataError(f"{ident}: input_ids length must be 3..{max_sequence_tokens}")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise CptDataError(f"{ident}: input_ids contains an invalid token id")
    if ids[0] != BOS_ID or ids[-1] != EOS_ID or ids.count(BOS_ID) != 1 or ids.count(EOS_ID) != 1:
        raise CptDataError(f"{ident}: input_ids must have one manual BOS0 and one manual EOS1")
    return list(ids)


def _validate_labels(ids: list[int], labels: Any, ident: str, *, final: bool, overlap: int) -> list[int]:
    if not isinstance(labels, list) or len(labels) != len(ids):
        raise CptDataError(f"{ident}: labels must align with input_ids")
    if any(type(token) is not int or token < -100 or token >= VOCAB_SIZE for token in labels):
        raise CptDataError(f"{ident}: labels contains an invalid token id")
    if labels[0] != -100:
        raise CptDataError(f"{ident}: BOS label must be masked explicitly")
    if not 0 <= overlap <= len(ids) - 2:
        raise CptDataError(f"{ident}: overlap_context_tokens is outside input_ids")
    if any(value != -100 for value in labels[1:1 + overlap]):
        raise CptDataError(f"{ident}: overlap context labels must be masked")
    owned_start, owned_end = 1 + overlap, len(ids) - 1
    if any(labels[index] != ids[index] for index in range(owned_start, owned_end)):
        raise CptDataError(f"{ident}: code labels must equal their supplied input token")
    # The final EOS is the only EOS target. Internal EOS is visible context but
    # cannot be a stop target, avoiding artificial document boundaries.
    if labels[-1] != (EOS_ID if final else -100):
        raise CptDataError(f"{ident}: EOS label disagrees with document-final status")
    return list(labels)


def validate_row(row: Mapping[str, Any], *, max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
                 vocab_size: int = VOCAB_SIZE) -> dict[str, Any]:
    """Validate the normalized five-field row contract."""
    if not isinstance(row, Mapping):
        raise CptDataError("raw row must be an object")
    if set(row) != ROW_FIELDS:
        raise CptDataError(f"raw row fields mismatch: expected {sorted(ROW_FIELDS)}")
    ident = _nonempty(row["id"], "id")
    if row["split"] != "train":
        raise CptDataError(f"{ident}: split must be train")
    package_id = _nonempty(row["package_id"], f"{ident}.package_id")
    ids = _validate_ids(row["input_ids"], ident, max_sequence_tokens=max_sequence_tokens, vocab_size=vocab_size)
    source = row["source"]
    if not isinstance(source, Mapping) or set(source) != SOURCE_FIELDS:
        raise CptDataError(f"{ident}: source provenance fields mismatch")
    if source["source_kind"] != "raw_r_document":
        raise CptDataError(f"{ident}: source_kind is not raw_r_document")
    for name in ("source_id", "group_id", "builder_id", "tokenizer_revision"):
        _nonempty(source[name], f"{ident}.source.{name}")
    _sha256_text(source["document_sha256"], f"{ident}.source.document_sha256")
    _sha256_text(source["token_stream_sha256"], f"{ident}.source.token_stream_sha256")
    _sha256_text(source["builder_sha256"], f"{ident}.source.builder_sha256")
    if source["tokenizer_revision"] != TOKENIZER_REVISION:
        raise CptDataError(f"{ident}: tokenizer revision differs from the pinned contract")
    document_count = _int(source["document_token_count"], f"{ident}.source.document_token_count", 1)
    start = _int(source["token_start"], f"{ident}.source.token_start")
    end = _int(source["token_end"], f"{ident}.source.token_end")
    chunk_index = _int(source["chunk_index"], f"{ident}.source.chunk_index")
    chunk_count = _int(source["chunk_count"], f"{ident}.source.chunk_count", 1)
    if not 0 <= start < end <= document_count:
        raise CptDataError(f"{ident}: source token range is outside the document")
    if chunk_index >= chunk_count:
        raise CptDataError(f"{ident}: chunk_index is outside chunk_count")
    overlap = source.get("overlap_context_tokens", 0)
    if type(overlap) is not int or overlap < 0 or overlap > len(ids) - 2:
        raise CptDataError(f"{ident}: invalid overlap_context_tokens")
    if end - start != len(ids) - 2 - overlap:
        raise CptDataError(f"{ident}: source span length does not match owned payload length")
    return {"id": ident, "split": "train", "package_id": package_id,
            "input_ids": ids, "source": dict(source)}


def validate_materialized_row(row: Mapping[str, Any], *, max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
                              vocab_size: int = VOCAB_SIZE) -> dict[str, Any]:
    """Validate one row emitted by raw_cpt.py and normalize its identity.

    The materializer's source_path is retained as provenance for an admitted
    manifest. This validator never opens that path. ``labels`` and
    ``attention_mask`` are required so a caller cannot silently infer EOS
    supervision at internal chunk boundaries.
    """
    if not isinstance(row, Mapping) or not MATERIALIZED_REQUIRED_FIELDS <= set(row):
        raise CptDataError(f"materialized row fields missing: expected {sorted(MATERIALIZED_REQUIRED_FIELDS)}")
    unknown = set(row) - MATERIALIZED_REQUIRED_FIELDS - MATERIALIZED_OPTIONAL_FIELDS
    if unknown:
        raise CptDataError(f"materialized row has unsupported fields: {sorted(unknown)}")
    ident = _nonempty(row["row_id"], "row_id")
    if row["schema"] != 1 or row["cpt_partition"] not in ("cpt_train", "cpt_validation"):
        raise CptDataError(f"{ident}: unsupported materialized schema or partition")
    document_id = _nonempty(row["document_id"], f"{ident}.document_id")
    package = _nonempty(row["package"], f"{ident}.package")
    group_id = _nonempty(row["group_id"], f"{ident}.group_id")
    _nonempty(row["source_path"], f"{ident}.source_path")
    source_sha = _sha256_text(row["source_sha256"], f"{ident}.source_sha256")
    ids = _validate_ids(row["input_ids"], ident, max_sequence_tokens=max_sequence_tokens, vocab_size=vocab_size)
    start = _int(row["source_token_start"], f"{ident}.source_token_start")
    end = _int(row["source_token_end"], f"{ident}.source_token_end")
    if "token_start" in row and row["token_start"] != start:
        raise CptDataError(f"{ident}: token_start alias differs from source_token_start")
    if "token_end" in row and row["token_end"] != end:
        raise CptDataError(f"{ident}: token_end alias differs from source_token_end")
    overlap = _int(row["overlap_context_tokens"], f"{ident}.overlap_context_tokens")
    final = row["is_document_end"]
    if not isinstance(final, bool):
        raise CptDataError(f"{ident}: is_document_end must be boolean")
    if end <= start or end - start != len(ids) - 2 - overlap:
        raise CptDataError(f"{ident}: owned source span disagrees with input_ids/overlap")
    labels = _validate_labels(ids, row["labels"], ident, final=final, overlap=overlap)
    attention = row["attention_mask"]
    if attention != [1] * len(ids):
        raise CptDataError(f"{ident}: stored attention_mask must cover unpadded input")
    supervised = sum(value != -100 for value in labels)
    if row["supervised_tokens"] != supervised:
        raise CptDataError(f"{ident}: supervised_tokens disagrees with labels")
    if row["chunk_index"] < 0:
        raise CptDataError(f"{ident}: chunk_index must be nonnegative")
    if "document_token_count" in row and (type(row["document_token_count"]) is not int
                                            or row["document_token_count"] < end):
        raise CptDataError(f"{ident}: invalid document_token_count")
    if "token_stream_sha256" in row:
        _sha256_text(row["token_stream_sha256"], f"{ident}.token_stream_sha256")
    if "builder_sha256" in row:
        _sha256_text(row["builder_sha256"], f"{ident}.builder_sha256")
    if "tokenizer_revision" in row and row["tokenizer_revision"] != TOKENIZER_REVISION:
        raise CptDataError(f"{ident}: tokenizer revision differs from the pinned contract")
    return {
        "id": ident, "split": "train", "package_id": package, "input_ids": ids,
        "labels": labels, "attention_mask": list(attention),
        "source": {
            "source_kind": "raw_r_document", "source_id": document_id,
            "group_id": group_id, "document_sha256": source_sha,
            # Filled from the complete row group by validate_materialized_rows.
            "token_stream_sha256": row.get("token_stream_sha256"),
            "document_token_count": row.get("document_token_count"),
            "token_start": start, "token_end": end,
            "chunk_index": row["chunk_index"], "chunk_count": None,
            "tokenizer_revision": TOKENIZER_REVISION,
            "builder_id": row.get("builder_id", "raw_cpt.py"),
            "builder_sha256": row.get("builder_sha256"),
            "overlap_context_tokens": overlap,
            "source_path": row["source_path"],
            "is_document_end": final,
        },
        "_materialized": dict(row),
    }


def _document_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = row["source"]
    return (row["package_id"], source["source_id"], source["document_sha256"])


def _validate_complete_groups(rows: Sequence[dict[str, Any]], *, require_complete_documents: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_document_key(row)].append(row)
    documents = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item["source"]["chunk_index"])
        indexes = [item["source"]["chunk_index"] for item in ordered]
        if indexes != list(range(len(indexes))):
            raise CptDataError(f"{key}: chunk indexes must start at zero without duplicates")
        previous_end = 0
        payload = []
        for item in ordered:
            source = item["source"]
            if source["token_start"] != previous_end:
                raise CptDataError(f"{key}: owned source chunks have a gap or overlap")
            payload.extend(item["input_ids"][1 + source.get("overlap_context_tokens", 0):-1])
            previous_end = source["token_end"]
            declared_count = source.get("document_token_count")
            final_flag = source.get("is_document_end")
            # Canonical rows do not duplicate the materializer's boolean. Their
            # terminal status is established by the complete declared span.
            if final_flag is None:
                final_flag = declared_count is not None and source["token_end"] == declared_count
            if bool(final_flag) != (item is ordered[-1]):
                raise CptDataError(f"{key}: only final chunk may mark document end")
        if not payload or previous_end <= 0:
            raise CptDataError(f"{key}: empty document payload")
        final = ordered[-1]
        # For canonical rows, declared document length/hash are required. For
        # materialized rows, the complete span and raw packer's round-trip audit
        # establish coverage; the independently reproducible stream hash is
        # recorded in the normalized summary.
        declared_count = final["source"].get("document_token_count")
        if declared_count is not None and previous_end != declared_count:
            raise CptDataError(f"{key}: final chunk does not cover document end")
        stream_hash = canonical_token_stream_sha256(payload)
        declared_hash = final["source"].get("token_stream_sha256")
        if declared_hash is not None and stream_hash != declared_hash:
            raise CptDataError(f"{key}: reconstructed payload hash differs")
        if require_complete_documents and not final["source"].get("is_document_end", True):
            raise CptDataError(f"{key}: document has no terminal chunk")
        documents.append({
            "package_id": key[0], "source_id": key[1], "document_sha256": key[2],
            "chunks": len(ordered), "document_token_count": previous_end,
            "token_stream_sha256": stream_hash,
        })
    return rows, documents


def validate_rows(rows: Iterable[Mapping[str, Any]], *, max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
                  vocab_size: int = VOCAB_SIZE, require_complete_documents: bool = True) -> dict[str, Any]:
    """Validate canonical rows and prove contiguous document coverage."""
    checked, seen = [], set()
    for row in rows:
        value = validate_row(row, max_sequence_tokens=max_sequence_tokens, vocab_size=vocab_size)
        if value["id"] in seen:
            raise CptDataError(f"duplicate row id: {value['id']}")
        seen.add(value["id"])
        checked.append(value)
    if not checked:
        raise CptDataError("raw row stream is empty")
    checked, documents = _validate_complete_groups(checked, require_complete_documents=require_complete_documents)
    return {
        "rows": len(checked), "documents": len(documents),
        "packages": sorted({row["package_id"] for row in checked}),
        "input_tokens": sum(len(row["input_ids"]) for row in checked),
        "payload_tokens": sum(len(row["input_ids"]) - 2 - row["source"].get("overlap_context_tokens", 0) for row in checked),
        "loss_tokens": sum(sum(value != -100 for value in _derived_labels(row))
                            for row in checked),
        "documents_detail": documents, "rows_checked": checked,
    }


def validate_materialized_rows(rows: Iterable[Mapping[str, Any]], *, max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
                               vocab_size: int = VOCAB_SIZE) -> dict[str, Any]:
    """Validate raw_cpt rows, including derived labels and no-loss spans."""
    checked, seen = [], set()
    for row in rows:
        value = validate_materialized_row(row, max_sequence_tokens=max_sequence_tokens, vocab_size=vocab_size)
        if value["id"] in seen:
            raise CptDataError(f"duplicate row id: {value['id']}")
        seen.add(value["id"]); checked.append(value)
    if not checked:
        raise CptDataError("materialized row stream is empty")
    checked, documents = _validate_complete_groups(checked)
    return {
        "rows": len(checked), "documents": len(documents),
        "packages": sorted({row["package_id"] for row in checked}),
        "input_tokens": sum(len(row["input_ids"]) for row in checked),
        "payload_tokens": sum(len(row["input_ids"]) - 2 - row["source"]["overlap_context_tokens"] for row in checked),
        "loss_tokens": sum(sum(value != -100 for value in row["labels"][1:]) for row in checked),
        "documents_detail": documents, "rows_checked": checked,
    }


def validate_package_holdout(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], *,
                             max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS, materialized: bool = False) -> dict[str, Any]:
    """Require package and document-disjoint holdout; both files retain TRAIN split."""
    validator = validate_materialized_rows if materialized else validate_rows
    train = validator(train_rows, max_sequence_tokens=max_sequence_tokens)
    validation = validator(validation_rows, max_sequence_tokens=max_sequence_tokens)
    train_packages, validation_packages = set(train["packages"]), set(validation["packages"])
    if train_packages & validation_packages:
        raise CptDataError(f"package leakage between CPT train and holdout: {sorted(train_packages & validation_packages)}")
    train_docs = {d["document_sha256"] for d in train["documents_detail"]}
    validation_docs = {d["document_sha256"] for d in validation["documents_detail"]}
    if train_docs & validation_docs:
        raise CptDataError(f"document leakage between CPT train and holdout: {sorted(train_docs & validation_docs)}")
    return {
        "train": {k: v for k, v in train.items() if k != "rows_checked"},
        "validation": {k: v for k, v in validation.items() if k != "rows_checked"},
        "train_packages": sorted(train_packages), "validation_packages": sorted(validation_packages),
        "package_disjoint": True, "document_disjoint": True,
        "train_rows_checked": train["rows_checked"], "validation_rows_checked": validation["rows_checked"],
    }


def validate_draw_schedule(schedule: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
                           token_rows_sha256: str, max_steps: int, effective_batch: int = 16) -> dict[str, Any]:
    """Validate the compact CPT schedule or an accepted schedule wrapper.

    Root may wrap the declared sequence in the existing campaign schedule
    envelope. Only these immutable fields affect training: split_id, method,
    seed, max_steps, effective_batch, token_rows_sha256, and row_ids.
    """
    if not isinstance(schedule, Mapping):
        raise CptDataError("CPT draw schedule must be an object")
    core = schedule.get("schedule", schedule)
    if not isinstance(core, Mapping):
        raise CptDataError("CPT draw schedule core must be an object")
    required = {"split_id", "max_steps", "effective_batch", "token_rows_sha256", "row_ids"}
    if not required <= set(core):
        raise CptDataError("CPT draw schedule fields are incomplete")
    method = core.get("method", schedule.get("method", "sequential"))
    seed = core.get("seed", schedule.get("seed", 3407))
    if method not in ("sequential", "without_replacement", "one_deterministic_without_replacement"):
        raise CptDataError("CPT draw schedule must be deterministic")
    if type(seed) is not int:
        raise CptDataError("CPT draw schedule seed must be an integer")
    if core["max_steps"] != max_steps or core["effective_batch"] != effective_batch:
        raise CptDataError("CPT draw schedule differs from recipe")
    if core["token_rows_sha256"] != token_rows_sha256:
        raise CptDataError("CPT draw schedule refers to different raw rows")
    row_ids = core["row_ids"]
    if not isinstance(row_ids, list) or len(row_ids) != max_steps * effective_batch:
        raise CptDataError("CPT draw count must equal max_steps * effective_batch")
    known = {row["id"] for row in rows}
    if any(not isinstance(ident, str) or ident not in known for ident in row_ids):
        raise CptDataError("CPT draw schedule refers to an absent row")
    return {"max_steps": max_steps, "effective_batch": effective_batch, "draws": len(row_ids),
            "unique_rows": len(known), "split_id": core["split_id"], "method": method, "seed": seed,
            "row_ids": list(row_ids)}


def read_jsonl(path: str | Path, *, max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
               materialized: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise CptDataError(f"raw rows path must be an existing absolute file: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise CptDataError(f"invalid JSON at {path}:{line_number}") from error
    (validate_materialized_rows if materialized else validate_rows)(rows, max_sequence_tokens=max_sequence_tokens)
    return rows


def causal_lm_collator(batch: Sequence[Mapping[str, Any]], *, pad_token_id: int = PAD_ID,
                       max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS) -> dict[str, Any]:
    """Pad explicit causal rows without masking actual EOS by token value."""
    if not batch:
        raise CptDataError("CPT batch cannot be empty")
    minimal = set(batch[0]) == {"input_ids", "labels", "attention_mask"}
    materialized = "labels" in batch[0] and not minimal
    if materialized:
        rows = [validate_materialized_row(row, max_sequence_tokens=max_sequence_tokens) for row in batch]
    elif minimal:
        rows = []
        for index, row in enumerate(batch):
            ids, labels, attention = row["input_ids"], row["labels"], row["attention_mask"]
            ident = f"batch[{index}]"
            _validate_ids(ids, ident, max_sequence_tokens=max_sequence_tokens, vocab_size=VOCAB_SIZE)
            if not isinstance(labels, list) or len(labels) != len(ids):
                raise CptDataError(f"{ident}: labels must align with input_ids")
            if not isinstance(attention, list) or attention != [1] * len(ids):
                raise CptDataError(f"{ident}: attention_mask must cover unpadded input")
            if any(type(value) is not int or value < -100 or value >= VOCAB_SIZE for value in labels):
                raise CptDataError(f"{ident}: labels contains an invalid token id")
            rows.append({"input_ids": list(ids), "labels": list(labels), "attention_mask": list(attention)})
    else:
        rows = [validate_row(row, max_sequence_tokens=max_sequence_tokens) for row in batch]
    import torch
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        ids = row["input_ids"]
        values = torch.tensor(ids, dtype=torch.long)
        length = len(ids)
        input_ids[index, :length] = values
        attention[index, :length] = 1
        if materialized or minimal:
            labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long)
        else:
            # Canonical rows carry complete document metadata. The final EOS
            # is supervised; internal EOS is visible but masked.
            labels[index, :length] = torch.tensor(_derived_labels(row), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def verify_causal_batch(batch: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> int:
    """Check actual collator tensors and return the shifted loss denominator."""
    import torch
    expected = causal_lm_collator(rows)
    for name in ("input_ids", "attention_mask", "labels"):
        if not torch.equal(batch[name].cpu(), expected[name]):
            raise CptDataError(f"actual collator changed {name}")
    denominator = int(batch["labels"][:, 1:].ne(-100).sum())
    expected_denominator = int(expected["labels"][:, 1:].ne(-100).sum())
    if denominator != expected_denominator:
        raise CptDataError("causal loss denominator changed")
    return denominator


def causal_loss_denominator(batch: Mapping[str, Any]) -> int:
    return int(batch["labels"][:, 1:].ne(-100).sum())


def _derived_labels(row: Mapping[str, Any]) -> list[int]:
    """Build canonical-row labels with explicit overlap/EOS policy."""
    ids = row["input_ids"]
    source = row["source"]
    overlap = source.get("overlap_context_tokens", 0)
    final = source["token_end"] == source["document_token_count"]
    return ([-100] + [-100] * overlap + ids[1 + overlap:-1]
            + ([EOS_ID] if final else [-100]))
