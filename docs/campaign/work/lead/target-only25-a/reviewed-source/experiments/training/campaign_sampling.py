"""Finite, deterministic draw manifests for the Tuesday SFT campaign.

DAT-06 deliberately works on *admission metadata*, rather than prompt or target
text.  It therefore cannot create a training row and cannot repair a bad row.
The caller supplies the identity of the already admitted token-row registry and
this module only chooses finite presentations of those rows.

The sampler is intentionally a manifest builder.  A manifest with a non-zero
deficit has ``status == "infeasible"`` and must not be handed to the trainer.
There is no fallback loop which silently violates a row, family, or length cap.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "sepalith.dat06.sampler.v1"
DEFAULT_NOOP_FRACTION = 0.10
DEFAULT_FAMILY_CEILING = 0.25
DEFAULT_SMALL_PACK_CAP = 3
DEFAULT_ORDINARY_REPLAY_CAP = 8
DEFAULT_NATURALLY_LONG_FRACTION = 0.20
DEFAULT_SHORT_MAX_TOKENS = 2048
DEFAULT_LONG_MAX_TOKENS = 4096


class SamplerInputError(ValueError):
    """Raised when an input row is not an admitted metadata record."""


@dataclass(frozen=True)
class AdmittedRow:
    """The metadata needed by the finite sampler.

    No prompt, target, token-ID array, or future target text is accepted here.
    ``prompt_tokens`` and ``target_tokens`` are counts from the admission
    record.  They are used for exposure accounting only.
    """

    row_id: str
    family: str
    source_id: str
    package_id: str
    split: str
    semantic_noop: bool
    prompt_tokens: int
    target_tokens: int
    total_tokens: int
    length_bucket: str
    naturally_long: bool
    source_kind: str
    provenance: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdmittedRow":
        if not isinstance(value, Mapping):
            raise SamplerInputError("each sampler row must be an admission mapping")

        # A sampler input must stay metadata-only.  In particular, accepting an
        # input_ids or target field here would make it easy to accidentally build
        # a schedule from an unverified/future row.
        raw_keys = {
            "prompt", "prompt_text", "target", "target_text", "region_new",
            "input_ids", "target_body_tokens", "target_terminal_tokens",
        }
        present_raw = sorted(raw_keys.intersection(value))
        if present_raw:
            raise SamplerInputError(
                "raw prompt/target fields are forbidden in sampler metadata: "
                + ", ".join(present_raw)
            )

        missing = [
            key for key in (
                "row_id", "family", "source_id", "package_id", "split",
                "semantic_noop", "prompt_tokens", "target_tokens", "total_tokens",
                "length_bucket", "naturally_long", "source_kind",
            ) if key not in value
        ]
        # ``id`` is the spelling used by the token-row JSONL.  It is accepted as
        # a metadata alias, but the normalized record always exposes row_id.
        if "row_id" not in value and "id" in value:
            missing = [key for key in missing if key != "row_id"]
        if missing:
            raise SamplerInputError("missing admission metadata: " + ", ".join(missing))

        def string(name: str, *, alias: str | None = None, default: str | None = None) -> str:
            candidate = value.get(name)
            if candidate is None and alias is not None:
                candidate = value.get(alias)
            if candidate is None:
                candidate = default
            if not isinstance(candidate, str) or not candidate:
                raise SamplerInputError(f"{name} must be a non-empty string")
            return candidate

        row_id = string("row_id", alias="id")
        family = string("family")
        source_id = string("source_id")
        package_id = string("package_id")
        split = string("split")
        length_bucket = string("length_bucket").lower()
        source_kind = string("source_kind").lower().replace("-", "_")
        if source_kind in {"smallpack", "new_small_pack", "new_smallpack"}:
            source_kind = "small_pack"
        if source_kind not in {"ordinary", "small_pack"}:
            raise SamplerInputError(
                f"{row_id}: source_kind must be ordinary or small_pack, got {source_kind!r}"
            )
        if length_bucket not in {"short", "long"}:
            raise SamplerInputError(
                f"{row_id}: length_bucket must be short or long, got {length_bucket!r}"
            )

        semantic_noop = value.get("semantic_noop")
        if type(semantic_noop) is not bool:
            raise SamplerInputError(f"{row_id}: semantic_noop must be a boolean")
        operation = value.get("operation")
        if operation is not None:
            if not isinstance(operation, str) or not operation:
                raise SamplerInputError(f"{row_id}: operation must be a non-empty string")
            operation_noop = operation.lower() in {"no_op", "noop", "no-op"}
            if operation_noop != semantic_noop:
                raise SamplerInputError(
                    f"{row_id}: operation and semantic_noop disagree"
                )

        naturally_long = value.get("naturally_long")
        if type(naturally_long) is not bool:
            raise SamplerInputError(f"{row_id}: naturally_long must be a boolean")
        if (length_bucket == "long") != naturally_long:
            raise SamplerInputError(
                f"{row_id}: long bucket and naturally_long must agree"
            )

        numbers: dict[str, int] = {}
        for name in ("prompt_tokens", "target_tokens", "total_tokens"):
            number = value.get(name)
            if type(number) is not int or number < 0:
                raise SamplerInputError(f"{row_id}: {name} must be a non-negative integer")
            numbers[name] = number
        if numbers["total_tokens"] != numbers["prompt_tokens"] + numbers["target_tokens"]:
            raise SamplerInputError(f"{row_id}: total_tokens does not equal prompt + target")

        for flag in ("admitted", "heldout", "is_heldout", "prompt_truncated", "target_truncated"):
            if flag in value and type(value[flag]) is not bool:
                raise SamplerInputError(f"{row_id}: {flag} must be a boolean")
        if value.get("admitted", True) is not True:
            raise SamplerInputError(f"{row_id}: row is not admitted")
        if value.get("heldout", False) or value.get("is_heldout", False):
            raise SamplerInputError(f"{row_id}: heldout row is forbidden in train schedule")
        if value.get("prompt_truncated", False) or value.get("target_truncated", False):
            raise SamplerInputError(f"{row_id}: truncated rows are forbidden")

        provenance = value.get("provenance", "")
        if not isinstance(provenance, str):
            raise SamplerInputError(f"{row_id}: provenance must be a string")

        return cls(
            row_id=row_id,
            family=family,
            source_id=source_id,
            package_id=package_id,
            split=split,
            semantic_noop=semantic_noop,
            prompt_tokens=numbers["prompt_tokens"],
            target_tokens=numbers["target_tokens"],
            total_tokens=numbers["total_tokens"],
            length_bucket=length_bucket,
            naturally_long=naturally_long,
            source_kind=source_kind,
            provenance=provenance,
        )

    def metadata(self, *, capacity: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "row_id": self.row_id,
            "family": self.family,
            "source_id": self.source_id,
            "package_id": self.package_id,
            "split": self.split,
            "semantic_noop": self.semantic_noop,
            "prompt_tokens": self.prompt_tokens,
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "length_bucket": self.length_bucket,
            "naturally_long": self.naturally_long,
            "source_kind": self.source_kind,
        }
        if self.provenance:
            result["provenance"] = self.provenance
        if capacity is not None:
            result["presentation_capacity"] = capacity
        return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = _canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


def canonical_token_rows_sha256(
    rows: Sequence[AdmittedRow | Mapping[str, Any]],
    *,
    short_max_tokens: int = DEFAULT_SHORT_MAX_TOKENS,
    long_max_tokens: int = DEFAULT_LONG_MAX_TOKENS,
) -> str:
    """Hash normalized admission metadata when no registry digest is supplied.

    This is a metadata registry digest, not a claim that token IDs were
    re-read.  Production callers should pass the independently verified
    ``token_rows_sha256`` from ``campaign_sft_data``.
    """

    normalized = _normalize_rows(
        rows, short_max_tokens=short_max_tokens, long_max_tokens=long_max_tokens,
    )
    records = [row.metadata() for row in sorted(normalized, key=lambda item: item.row_id)]
    return _sha256(records)


def _normalize_rows(
    rows: Iterable[AdmittedRow | Mapping[str, Any]],
    *,
    short_max_tokens: int = DEFAULT_SHORT_MAX_TOKENS,
    long_max_tokens: int = DEFAULT_LONG_MAX_TOKENS,
) -> list[AdmittedRow]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise SamplerInputError("rows must be an iterable of admission metadata records")
    result: list[AdmittedRow] = []
    seen: set[str] = set()
    for value in rows:
        # Re-run the same strict checks for a manually constructed dataclass;
        # callers must not bypass admission validation by choosing another
        # input spelling.
        row = (AdmittedRow.from_mapping(value.metadata())
               if isinstance(value, AdmittedRow)
               else AdmittedRow.from_mapping(value))
        if row.row_id in seen:
            raise SamplerInputError(f"duplicate row_id: {row.row_id}")
        seen.add(row.row_id)
        if row.split != "train":
            raise SamplerInputError(
                f"{row.row_id}: split {row.split!r} is not admitted train data"
            )
        if row.length_bucket == "short" and row.total_tokens > short_max_tokens:
            raise SamplerInputError(
                f"{row.row_id}: short row exceeds {short_max_tokens} tokens"
            )
        if row.length_bucket == "long":
            if row.total_tokens <= short_max_tokens:
                raise SamplerInputError(
                    f"{row.row_id}: long row is not naturally beyond short limit"
                )
            if row.total_tokens > long_max_tokens:
                raise SamplerInputError(
                    f"{row.row_id}: long row exceeds {long_max_tokens} token ceiling"
                )
        result.append(row)
    return result


def validate_admitted_rows(
    rows: Iterable[AdmittedRow | Mapping[str, Any]],
    *,
    short_max_tokens: int = DEFAULT_SHORT_MAX_TOKENS,
    long_max_tokens: int = DEFAULT_LONG_MAX_TOKENS,
) -> tuple[AdmittedRow, ...]:
    """Normalize and validate rows without selecting any draw."""

    return tuple(_normalize_rows(rows, short_max_tokens=short_max_tokens,
                                 long_max_tokens=long_max_tokens))


def _stable_key(seed: int, *parts: str) -> str:
    encoded = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_key(row: AdmittedRow) -> str:
    return f"{row.package_id}::{row.source_id}"


def _row_capacity(
    row: AdmittedRow,
    *,
    small_pack_cap: int,
    ordinary_replay_cap: int,
) -> int:
    return small_pack_cap if row.source_kind == "small_pack" else ordinary_replay_cap


def _family_quotas(
    family_sizes: Mapping[str, int],
    slots: int,
    cap: int | Mapping[str, int],
) -> tuple[dict[str, int], int]:
    """Allocate slots by sqrt(row count), with deterministic largest remainder."""

    families = sorted(family_sizes)
    quotas = {family: 0 for family in families}
    if slots <= 0 or not families:
        return quotas, max(0, slots)
    weights = {family: math.sqrt(family_sizes[family]) for family in families}
    caps = {
        family: (cap[family] if isinstance(cap, Mapping) else cap)
        for family in families
    }
    total_weight = sum(weights.values())
    raw = {family: slots * weights[family] / total_weight for family in families}
    for family in families:
        quotas[family] = min(caps[family], int(math.floor(raw[family])))
    remaining = slots - sum(quotas.values())
    while remaining:
        candidates = [family for family in families if quotas[family] < caps[family]]
        if not candidates:
            break
        # Recompute the largest remainder after every placement.  This is
        # finite, deterministic, and keeps the family ceiling explicit.
        family = max(
            candidates,
            key=lambda item: (raw[item] - quotas[item], weights[item], item),
        )
        quotas[family] += 1
        remaining -= 1
    return quotas, max(0, remaining)


def _rotation_order(rows: Sequence[AdmittedRow], seed: int, family: str) -> list[AdmittedRow]:
    """Return a source-round-robin order, with each row appearing once per cycle."""

    by_source: dict[str, list[AdmittedRow]] = defaultdict(list)
    for row in rows:
        by_source[_source_key(row)].append(row)
    for source, source_rows in by_source.items():
        source_rows.sort(key=lambda row: _stable_key(seed, "row", family, source, row.row_id))
    sources = sorted(by_source, key=lambda source: _stable_key(seed, "source", family, source))
    order: list[AdmittedRow] = []
    for position in range(max((len(values) for values in by_source.values()), default=0)):
        for source in sources:
            source_rows = by_source[source]
            if position < len(source_rows):
                order.append(source_rows[position])
    return order


def _interleave_draws(draws: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Spread semantic/family queues across the schedule with exact totals.

    Each queue already has its source-rotation order.  Deficit round-robin
    chooses the queue furthest below its proportional prefix target, preserving
    every queue's internal order while preventing an all-no-op or all-family
    prefix.  The integer score avoids floating-point tie drift.
    """

    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for draw in draws:
        category = "__semantic_noop__" if draw["semantic_noop"] else draw["family"]
        queues[category].append(draw)
    categories = sorted(
        queues,
        key=lambda category: _stable_key(seed, "interleave-category", category),
    )
    targets = {category: len(queue) for category, queue in queues.items()}
    total = len(draws)
    emitted = Counter()
    queue_positions = Counter()
    result: list[dict[str, Any]] = []
    for position in range(total):
        available = [
            category for category in categories
            if queue_positions[category] < targets[category]
        ]
        category = max(
            available,
            key=lambda item: (
                targets[item] * (position + 1) - emitted[item] * total,
                item,
            ),
        )
        emitted[category] += 1
        result.append(queues[category][queue_positions[category]])
        queue_positions[category] += 1
    for index, draw in enumerate(result):
        draw["draw_index"] = index
    return result


def _prefix_milestones(draws: Sequence[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    """Summarize mixture and token exposure at fixed schedule prefixes."""

    milestones: list[dict[str, Any]] = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        prefix_length = min(len(draws), int(math.floor(requested * fraction)))
        prefix = draws[:prefix_length]
        families = Counter(draw["family"] for draw in prefix)
        sources = {draw["source_identity"] for draw in prefix}
        milestones.append({
            "fraction_of_requested": fraction,
            "draws": prefix_length,
            "semantic_noop": sum(draw["semantic_noop"] for draw in prefix),
            "semantic_noop_fraction": (
                sum(draw["semantic_noop"] for draw in prefix) / prefix_length
                if prefix_length else 0.0
            ),
            "families": {family: families[family] for family in sorted(families)},
            "distinct_rows": len({draw["row_id"] for draw in prefix}),
            "distinct_sources": len(sources),
            "long": sum(draw["naturally_long"] for draw in prefix),
            "prompt_tokens": sum(draw["prompt_tokens"] for draw in prefix),
            "target_tokens": sum(draw["target_tokens"] for draw in prefix),
        })
    return milestones


def _as_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SamplerInputError(f"{name} must be an integer >= {minimum}")
    return value


def build_draw_manifest(
    rows: Sequence[AdmittedRow | Mapping[str, Any]],
    *,
    max_steps: int,
    effective_batch: int,
    split_id: str,
    seed: int,
    token_rows_sha256: str | None = None,
    requested_draws: int | None = None,
    noop_fraction: float = DEFAULT_NOOP_FRACTION,
    family_ceiling: float = DEFAULT_FAMILY_CEILING,
    small_pack_cap: int = DEFAULT_SMALL_PACK_CAP,
    ordinary_replay_cap: int = DEFAULT_ORDINARY_REPLAY_CAP,
    naturally_long_fraction: float = DEFAULT_NATURALLY_LONG_FRACTION,
    short_max_tokens: int = DEFAULT_SHORT_MAX_TOKENS,
    long_max_tokens: int = DEFAULT_LONG_MAX_TOKENS,
) -> dict[str, Any]:
    """Build a finite predeclared schedule from admitted row metadata.

    ``requested_draws`` defaults to ``max_steps * effective_batch``.  If it is
    supplied explicitly it must equal that product, because
    ``campaign_sft_data`` rejects a row-id list with a different optimizer
    shape.  An incomplete schedule is returned with explicit deficits so a
    caller can inspect the reason and stop before training.
    """

    max_steps = _as_int(max_steps, "max_steps", minimum=1)
    effective_batch = _as_int(effective_batch, "effective_batch", minimum=1)
    if not isinstance(split_id, str) or not split_id:
        raise SamplerInputError("split_id must be a non-empty string")
    if type(seed) is not int:
        raise SamplerInputError("seed must be an integer")
    optimizer_draws = max_steps * effective_batch
    requested = optimizer_draws if requested_draws is None else _as_int(
        requested_draws, "requested_draws", minimum=1
    )
    if requested != optimizer_draws:
        raise SamplerInputError(
            "requested_draws must equal max_steps * effective_batch for the SFT schedule"
        )
    if not isinstance(noop_fraction, (int, float)) or not 0 <= noop_fraction <= 1:
        raise SamplerInputError("noop_fraction must be between 0 and 1")
    if not isinstance(family_ceiling, (int, float)) or not 0 < family_ceiling <= 1:
        raise SamplerInputError("family_ceiling must be in (0, 1]")
    if not isinstance(naturally_long_fraction, (int, float)) or not 0 <= naturally_long_fraction <= 1:
        raise SamplerInputError("naturally_long_fraction must be between 0 and 1")
    small_pack_cap = _as_int(small_pack_cap, "small_pack_cap", minimum=1)
    ordinary_replay_cap = _as_int(ordinary_replay_cap, "ordinary_replay_cap", minimum=1)
    short_max_tokens = _as_int(short_max_tokens, "short_max_tokens", minimum=1)
    long_max_tokens = _as_int(long_max_tokens, "long_max_tokens", minimum=short_max_tokens + 1)

    admitted = list(validate_admitted_rows(
        rows, short_max_tokens=short_max_tokens, long_max_tokens=long_max_tokens,
    ))
    computed_rows_hash = canonical_token_rows_sha256(
        admitted, short_max_tokens=short_max_tokens, long_max_tokens=long_max_tokens,
    )
    if token_rows_sha256 is None:
        registry_hash = computed_rows_hash
        registry_hash_mode = "computed_from_admission_metadata"
    else:
        if not isinstance(token_rows_sha256, str) or len(token_rows_sha256) != 64:
            raise SamplerInputError("token_rows_sha256 must be a 64-character hex digest")
        try:
            int(token_rows_sha256, 16)
        except ValueError as error:
            raise SamplerInputError("token_rows_sha256 must be hexadecimal") from error
        registry_hash = token_rows_sha256
        registry_hash_mode = "caller_supplied_verified_token_registry"

    noop_slots = int(math.floor(requested * float(noop_fraction)))
    non_noop_slots = requested - noop_slots
    family_cap = max(1, int(math.floor(requested * float(family_ceiling))))
    naturally_long_cap = int(math.floor(requested * float(naturally_long_fraction)))

    capacities = {
        row.row_id: _row_capacity(
            row, small_pack_cap=small_pack_cap, ordinary_replay_cap=ordinary_replay_cap,
        ) for row in admitted
    }
    non_noop_by_family: dict[str, list[AdmittedRow]] = defaultdict(list)
    noop_rows: list[AdmittedRow] = []
    for row in admitted:
        if row.semantic_noop:
            noop_rows.append(row)
        else:
            non_noop_by_family[row.family].append(row)

    # State is deliberately local to one build.  No stateful iterator can
    # accidentally continue a previous schedule or replay forever.
    exposure = Counter()  # row_id -> number of presentations
    family_exposure = Counter()
    source_exposure = Counter()
    long_draws = 0
    draws: list[dict[str, Any]] = []

    def append(row: AdmittedRow) -> bool:
        nonlocal long_draws
        count = exposure[row.row_id]
        if count >= capacities[row.row_id]:
            return False
        # The family ceiling applies to every semantic class, including a
        # no-op row whose family also contains edits.
        if family_exposure[row.family] >= family_cap:
            return False
        if row.naturally_long and long_draws >= naturally_long_cap:
            return False
        exposure[row.row_id] += 1
        family_exposure[row.family] += 1
        source_exposure[_source_key(row)] += 1
        if row.naturally_long:
            long_draws += 1
        draws.append({
            "draw_index": len(draws),
            "row_id": row.row_id,
            "family": row.family,
            "source_id": row.source_id,
            "package_id": row.package_id,
            "source_identity": _source_key(row),
            "source_kind": row.source_kind,
            "semantic_noop": row.semantic_noop,
            "presentation": count + 1,
            "prompt_tokens": row.prompt_tokens,
            "target_tokens": row.target_tokens,
            "total_tokens": row.total_tokens,
            "length_bucket": row.length_bucket,
            "naturally_long": row.naturally_long,
        })
        return True

    order_cache: dict[tuple[str, str], list[AdmittedRow]] = {}
    rotation_cursor: dict[tuple[str, str], int] = defaultdict(int)

    def take(
        candidates: Sequence[AdmittedRow], requested_count: int, family: str,
        *, category: str = "family",
    ) -> int:
        """Take at most requested_count with finite source/row rotation."""

        if requested_count <= 0 or not candidates:
            return 0
        cache_key = (category, family)
        if cache_key not in order_cache:
            order_cache[cache_key] = _rotation_order(candidates, seed, family)
        order = order_cache[cache_key]
        # A row can be considered at most once for every finite capacity.  The
        # cursor persists between quota and backfill calls, so a later call
        # starts at the next unused source/row instead of repeatedly revisiting
        # the first row.
        max_attempts = len(order) * max((capacities[row.row_id] for row in order), default=0)
        cursor = rotation_cursor[cache_key]
        selected = 0
        idle = 0
        attempts = 0
        while selected < requested_count and attempts < max_attempts:
            row = order[cursor % len(order)]
            cursor += 1
            attempts += 1
            if append(row):
                selected += 1
                idle = 0
            else:
                idle += 1
                # A complete pass with no append proves that all remaining
                # candidates are at capacity or blocked by the long/family
                # policy.  Stop without an unbounded search.
                if idle >= len(order):
                    break
        rotation_cursor[cache_key] = cursor
        return selected

    # Reserve no-op slots first.  If the finite no-op pool cannot supply them,
    # the shortage is explicitly converted into eligible non-no-op backfill.
    noop_achieved_initial = take(
        noop_rows, noop_slots, "__semantic_noop__", category="semantic_noop",
    )
    noop_shortage = noop_slots - noop_achieved_initial

    family_sizes = {family: len(values) for family, values in non_noop_by_family.items()}
    family_target_slots = non_noop_slots + noop_shortage
    remaining_family_capacity = {
        family: max(0, family_cap - family_exposure[family])
        for family in family_sizes
    }
    family_quotas, allocator_deficit = _family_quotas(
        family_sizes, family_target_slots, remaining_family_capacity,
    )
    initial_family_exposure: Counter[str] = Counter()
    initial_draw_count = len(draws)
    for family in sorted(family_quotas):
        selected = take(non_noop_by_family[family], family_quotas[family], family)
        # No-op rows are excluded above, so this is the exact count before
        # cross-family backfill.
        initial_family_exposure[family] = selected
    initial_non_noop_draws = len(draws) - initial_draw_count

    # Fill unallocated or quota-deficit slots by deterministic family/source
    # rotation.  The family ceiling and every row capacity still apply.
    desired_after_quota = min(requested, noop_achieved_initial + family_target_slots)
    backfill_draws = 0
    family_order = sorted(
        non_noop_by_family,
        key=lambda family: _stable_key(seed, "backfill-family", family),
    )
    while len(draws) < requested:
        progress = 0
        for family in family_order:
            if len(draws) >= requested:
                break
            if family_exposure[family] >= family_cap:
                continue
            progress += take(non_noop_by_family[family], 1, family)
            if len(draws) >= requested:
                break
        if progress == 0:
            break
        backfill_draws += progress

    # ``desired_after_quota`` is useful for diagnosing a quota allocator cap;
    # ``backfill_draws`` is the actual number selected in the second phase.
    del desired_after_quota, initial_non_noop_draws

    # Selection quotas are computed before ordering.  Interleave their queues
    # only after finite capacities are applied, so prefix distributions are
    # smooth while final exposure and family caps remain exact.
    draws = _interleave_draws(draws, seed)

    family_ledger: dict[str, dict[str, int]] = {}
    for family in sorted(family_sizes):
        requested_family = family_quotas.get(family, 0)
        initial = initial_family_exposure[family]
        achieved = family_exposure[family]
        family_ledger[family] = {
            "sqrt_quota": requested_family,
            "initial_achieved": initial,
            "achieved": achieved,
            "backfill": max(0, achieved - initial),
            "remaining_deficit": max(0, requested_family - achieved),
            "family_ceiling": family_cap,
            "capacity_after_noop_reservation": remaining_family_capacity[family],
        }

    global_deficit = requested - len(draws)
    if global_deficit:
        reasons = ["finite_row_or_family_capacity"]
        if any(row.naturally_long for row in admitted) and long_draws >= naturally_long_cap:
            reasons.append("naturally_long_fraction_ceiling")
        if allocator_deficit:
            reasons.append("family_ceiling_allocator_limit")
    else:
        reasons = []

    row_exposure = {row.row_id: exposure[row.row_id] for row in sorted(admitted, key=lambda r: r.row_id)}
    row_eligibility = {
        row.row_id: {
            "eligible": True,
            "split": row.split,
            "heldout": False,
            "capacity": capacities[row.row_id],
            "length_bucket": row.length_bucket,
            "naturally_long": row.naturally_long,
        }
        for row in sorted(admitted, key=lambda r: r.row_id)
    }
    source_keys = sorted({_source_key(row) for row in admitted})
    source_exposure_result = {source: source_exposure[source] for source in source_keys}
    family_keys = sorted({row.family for row in admitted})
    family_exposure_result = {family: family_exposure[family] for family in family_keys}
    long_count = sum(1 for draw in draws if draw["naturally_long"])
    noop_count = sum(1 for draw in draws if draw["semantic_noop"])

    schedule = {
        "row_ids": [draw["row_id"] for draw in draws],
        "max_steps": max_steps,
        "effective_batch": effective_batch,
        "split_id": split_id,
        "token_rows_sha256": registry_hash,
    }
    schedule_hash = _sha256(schedule)
    admitted_records = [
        row.metadata(capacity=capacities[row.row_id])
        for row in sorted(admitted, key=lambda item: item.row_id)
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if global_deficit == 0 else "infeasible",
        "schema_contract": {
            "input": "admission metadata only; no prompt, target, or future target fields",
            "draw_order": "predeclared row_ids; source round-robin before finite replay",
            "incomplete_use": "reject manifests whose status is infeasible",
        },
        "policy": {
            "seed": seed,
            "requested_draws": requested,
            "max_steps": max_steps,
            "effective_batch": effective_batch,
            "split_id": split_id,
            "noop_fraction": float(noop_fraction),
            "noop_slots": noop_slots,
            "family_allocation": "sqrt_admitted_family_row_count",
            "family_ceiling_fraction": float(family_ceiling),
            "effective_family_ceiling": family_cap,
            "small_pack_replay_cap": small_pack_cap,
            "ordinary_replay_cap": ordinary_replay_cap,
            "naturally_long_fraction_ceiling": float(naturally_long_fraction),
            "naturally_long_slots": naturally_long_cap,
            "short_max_tokens": short_max_tokens,
            "long_max_tokens": long_max_tokens,
            "truncation": "forbidden; admission metadata must represent full prompt and target",
        },
        "registry": {
            "token_rows_sha256": registry_hash,
            "token_rows_sha256_mode": registry_hash_mode,
            "computed_admission_metadata_sha256": computed_rows_hash,
            "admitted_row_count": len(admitted),
            "admitted_rows": admitted_records,
        },
        "schedule": schedule,
        "schedule_sha256": schedule_hash,
        # Keep row_ids at top level as well as under schedule.  This matches the
        # existing campaign_sft_data schedule interface exactly.
        "row_ids": schedule["row_ids"],
        "max_steps": max_steps,
        "effective_batch": effective_batch,
        "split_id": split_id,
        "token_rows_sha256": registry_hash,
        "draw_count": len(draws),
        "draws": draws,
        "prefix_milestones": _prefix_milestones(draws, requested),
        "exposure": {
            "row": row_exposure,
            "row_eligibility": row_eligibility,
            "source": source_exposure_result,
            "family": family_exposure_result,
            "distinct_rows": sum(count > 0 for count in row_exposure.values()),
            "distinct_sources": sum(count > 0 for count in source_exposure_result.values()),
            "prompt_tokens": sum(draw["prompt_tokens"] for draw in draws),
            "target_tokens": sum(draw["target_tokens"] for draw in draws),
            "total_tokens": sum(draw["total_tokens"] for draw in draws),
        },
        "achieved_mixture": {
            "semantic_noop": {
                "reserved": noop_slots,
                "achieved": noop_count,
                "deficit": max(0, noop_slots - noop_count),
                "fraction": noop_count / requested if requested else 0.0,
            },
            "families": {
                family: {
                    "count": family_exposure_result.get(family, 0),
                    "fraction": family_exposure_result.get(family, 0) / requested
                    if requested else 0.0,
                } for family in family_keys
            },
            "length": {
                "short": sum(draw["length_bucket"] == "short" for draw in draws),
                "long": long_count,
                "long_fraction": long_count / requested if requested else 0.0,
                "long_ceiling": naturally_long_fraction,
            },
        },
        "quota_ledger": {
            "semantic_noop": {
                "reserved": noop_slots,
                "initial_achieved": noop_achieved_initial,
                "achieved": noop_count,
                "remaining_deficit": max(0, noop_slots - noop_count),
                "backfill_slots_created": noop_shortage,
            },
            "families": family_ledger,
            "allocator_unallocated": allocator_deficit,
            "backfill_draws": backfill_draws,
        },
        "deficits": {
            "requested_draws": requested,
            "achieved_draws": len(draws),
            "global": global_deficit,
            "no_op_reserved": noop_slots,
            "no_op_achieved": noop_count,
            "no_op_shortage": max(0, noop_slots - noop_count),
            "family_quota": {
                family: values["remaining_deficit"]
                for family, values in family_ledger.items()
                if values["remaining_deficit"]
            },
            "allocator_unallocated": allocator_deficit,
            "reasons": reasons,
        },
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def generate_draw_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility spelling for callers that call this a generator."""

    return build_draw_manifest(*args, **kwargs)


def validate_draw_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the structural and exposure invariants of one generated manifest."""

    if not isinstance(manifest, Mapping):
        raise SamplerInputError("manifest must be a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SamplerInputError("unsupported sampler manifest schema")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping):
        raise SamplerInputError("manifest schedule is missing")
    row_ids = manifest.get("row_ids")
    if row_ids != schedule.get("row_ids"):
        raise SamplerInputError("top-level row_ids differ from schedule row_ids")
    draws = manifest.get("draws")
    if not isinstance(draws, list) or [draw["row_id"] for draw in draws] != row_ids:
        raise SamplerInputError("draw records differ from schedule row_ids")
    if schedule.get("max_steps", 0) * schedule.get("effective_batch", 0) != len(row_ids) \
            and manifest.get("status") == "complete":
        raise SamplerInputError("complete schedule does not match optimizer draw shape")
    if manifest.get("schedule_sha256") != _sha256(schedule):
        raise SamplerInputError("schedule hash mismatch")
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != _sha256(body):
        raise SamplerInputError("manifest hash mismatch")
    exposure = manifest.get("exposure", {})
    row_counts = exposure.get("row", {})
    observed = Counter(draw["row_id"] for draw in draws)
    if any(row_counts.get(row_id, 0) != count for row_id, count in observed.items()):
        raise SamplerInputError("row exposure does not match draw records")
    registry_rows = {
        row["row_id"]: row for row in manifest.get("registry", {}).get("admitted_rows", [])
    }
    if set(row_counts) != set(registry_rows):
        raise SamplerInputError("row exposure and admitted registry differ")
    policy = manifest.get("policy", {})
    requested = policy.get("requested_draws")
    if manifest.get("draw_count") != len(draws):
        raise SamplerInputError("draw_count does not match draw records")
    if manifest.get("status") == "complete" and len(draws) != requested:
        raise SamplerInputError("complete manifest does not cover requested draws")
    if manifest.get("prefix_milestones") != _prefix_milestones(draws, requested):
        raise SamplerInputError("prefix milestone distribution does not match draws")
    family_counts = Counter()
    source_counts = Counter()
    noop_count = 0
    long_count = 0
    token_totals = Counter()
    presentation_seen = Counter()
    for draw in draws:
        if draw.get("presentation", 0) < 1:
            raise SamplerInputError("draw presentation number must be positive")
        row_id = draw.get("row_id")
        source = registry_rows.get(row_id)
        if source is None:
            raise SamplerInputError(f"draw refers to an absent admitted row: {row_id}")
        family_counts[draw["family"]] += 1
        source_counts[draw["source_identity"]] += 1
        noop_count += int(draw["semantic_noop"])
        long_count += int(draw["naturally_long"])
        token_totals["prompt_tokens"] += draw["prompt_tokens"]
        token_totals["target_tokens"] += draw["target_tokens"]
        token_totals["total_tokens"] += draw["total_tokens"]
        presentation_seen[row_id] += 1
        if draw["presentation"] != presentation_seen[row_id]:
            raise SamplerInputError(f"presentation order is not finite/monotonic: {row_id}")
        if draw["family"] != source["family"] or draw["source_kind"] != source["source_kind"]:
            raise SamplerInputError(f"draw metadata differs from admitted row: {row_id}")
        if draw["source_id"] != source["source_id"] or draw["package_id"] != source["package_id"]:
            raise SamplerInputError(f"draw source identity differs from admitted row: {row_id}")
        if draw["semantic_noop"] != source["semantic_noop"]:
            raise SamplerInputError(f"draw no-op label differs from admitted row: {row_id}")
        for field in ("prompt_tokens", "target_tokens", "total_tokens", "length_bucket", "naturally_long"):
            if draw[field] != source[field]:
                raise SamplerInputError(f"draw token/length metadata differs from admitted row: {row_id}")
    if any(
        row_counts[row_id] > registry_rows[row_id]["presentation_capacity"]
        for row_id in row_counts
    ):
        raise SamplerInputError("row presentation capacity exceeded")
    family_cap = policy.get("effective_family_ceiling")
    if any(count > family_cap for count in family_counts.values() if family_cap is not None):
        raise SamplerInputError("family ceiling exceeded")
    if noop_count > policy.get("noop_slots", noop_count):
        raise SamplerInputError("no-op reservation exceeded")
    if long_count > policy.get("naturally_long_slots", long_count):
        raise SamplerInputError("naturally-long ceiling exceeded")
    family_expected = {
        family: family_counts[family] for family in exposure.get("family", {})
    }
    source_expected = {
        source: source_counts[source] for source in exposure.get("source", {})
    }
    if family_expected != exposure.get("family"):
        raise SamplerInputError("family exposure does not match draw records")
    if source_expected != exposure.get("source"):
        raise SamplerInputError("source exposure does not match draw records")
    if dict(token_totals) != {
        "prompt_tokens": exposure.get("prompt_tokens"),
        "target_tokens": exposure.get("target_tokens"),
        "total_tokens": exposure.get("total_tokens"),
    }:
        raise SamplerInputError("token exposure does not match draw records")


__all__ = [
    "AdmittedRow",
    "SamplerInputError",
    "SCHEMA_VERSION",
    "build_draw_manifest",
    "canonical_token_rows_sha256",
    "generate_draw_manifest",
    "validate_admitted_rows",
    "validate_draw_manifest",
]
