"""Importance-matrix calibration text and the held-out KLD slice (card C09).

Both are built from editing-SFT TRAIN token rows (PRM03 render: BOS 0,
prompt, gold target, EOS 1).  llama.cpp tools read text, not token ids, so
each file is a detokenization whose llama.cpp tokenization must equal a
known id stream.  The builder records that stream and the CLI checks it
with ``llama-tokenize`` against the GGUF vocabulary.

Calibration stream (``llama-imatrix --parse-special --no-escape``):

- chunks of exactly ``n_ctx`` tokens, each starting at a row's BOS, so the
  BOS that llama-imatrix writes over every chunk's first token changes
  nothing;
- whole rows, ``<s>`` + body + ``</s>``, with their BOS 0 and EOS 1 parsed
  as control tokens (that is what ``--parse-special`` is for);
- a chunk that cannot be filled with whole rows starts with a prefix of an
  extra TRAIN row ("filler"), cut so that it round-trips exactly;
- the very first ``<s>`` is left out of the text: llama.cpp adds that BOS
  itself (``tokenizer.ggml.add_bos_token`` is true).

Held-out KLD stream (``llama-perplexity``, which never parses control
tokens): one window per chunk, the last ``n_ctx - 1`` tokens of a row before
its EOS, so each window ends with the gold target and its terminal line, and
the second half of the chunk (the part llama-perplexity scores) covers the
target.  Chunks after the first start with ``<think>``: a user-defined
token (id 8) that llama.cpp matches even without special parsing, so it cuts
the text at the chunk boundary, and llama-perplexity replaces it with BOS.

Rows that mention any added-token string are skipped: with special parsing
on, such text would become a control token, while training rows were
tokenized with special tokens split.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

BOS_ID = 0
EOS_ID = 1
BOS_TEXT = "<s>"
EOS_TEXT = "</s>"
# User-defined in the pinned GGUF (token type 4): llama.cpp splits on it even
# with parse_special off, and llama-perplexity overwrites it with BOS.
KLD_CHUNK_MARK_TEXT = "<think>"
KLD_CHUNK_MARK_ID = 8
CURSOR_MARKER = "<|user_cursor|>"
CURRENT_OPEN = "<<<<<<< CURRENT\n"
CURRENT_CLOSE = "\n=======\n"
MIN_KEY_CHARS = 40
SHINGLE_LINES = 3
MIN_SHINGLE_LINE_CHARS = 8
SHINGLE_MIN_SHARED = 3
SHINGLE_CONTAINMENT = 0.5
# Shingles present in more TRAIN rows than this are boilerplate (protocol
# scaffolding, common roxygen tags) and say nothing about duplication.
SHINGLE_MAX_DOCUMENT_FREQUENCY = 25
FILLER_LOOKAHEAD = 64


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha256(ids: Sequence[int]) -> str:
    """Canonical hash of a token id list (compact JSON)."""
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def normalize(text: str) -> str:
    """NFKC, all whitespace runs collapsed to one space, stripped."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def current_region(prompt_text: str) -> str:
    """The ``<<<<<<< CURRENT`` block of a PRM03 prompt, without the cursor marker."""
    start = prompt_text.rfind(CURRENT_OPEN)
    if start < 0:
        return ""
    start += len(CURRENT_OPEN)
    end = prompt_text.find(CURRENT_CLOSE, start)
    if end < 0:
        end = len(prompt_text)
    return prompt_text[start:end].replace(CURSOR_MARKER, "")


def near_duplicate_keys(prompt_text: str, target_body_text: str, operation: str) -> set[str]:
    """Exact hashes of normalized text that identify a near-duplicate case.

    - ``p:`` the whole prompt (cursor marker removed);
    - ``r:`` the current region, if it has at least ``MIN_KEY_CHARS`` characters;
    - ``t:`` the gold edit body of a non-no-op case of that length.
    """
    keys = {"p:" + sha256_text(normalize(prompt_text.replace(CURSOR_MARKER, "")))}
    region = normalize(current_region(prompt_text))
    if len(region) >= MIN_KEY_CHARS:
        keys.add("r:" + sha256_text(region))
    target = normalize(target_body_text)
    if operation != "no_op" and len(target) >= MIN_KEY_CHARS:
        keys.add("t:" + sha256_text(target))
    return keys


def shingles(prompt_text: str) -> set[str]:
    """Hashes of ``SHINGLE_LINES`` consecutive non-trivial normalized prompt lines."""
    lines = [normalize(line) for line in prompt_text.replace(CURSOR_MARKER, "").split("\n")]
    lines = [line for line in lines if len(line) >= MIN_SHINGLE_LINE_CHARS]
    return {hashlib.sha1("\n".join(lines[i:i + SHINGLE_LINES]).encode()).hexdigest()[:16]
            for i in range(len(lines) - SHINGLE_LINES + 1)}


@dataclass(frozen=True)
class DevIndex:
    ids: frozenset[str]
    keys: frozenset[str]
    shingle_owners: Mapping[str, frozenset[str]]
    shingle_counts: Mapping[str, int]


def build_dev_index(cases: Iterable[tuple[str, str, str, str]]) -> DevIndex:
    """Index DEV cases given as ``(id, prompt_text, target_body_text, operation)``."""
    ids, keys, owners, counts = set(), set(), defaultdict(set), {}
    for case_id, prompt_text, body, operation in cases:
        ids.add(case_id)
        keys |= near_duplicate_keys(prompt_text, body, operation)
        found = shingles(prompt_text)
        counts[case_id] = len(found)
        for shingle in found:
            owners[shingle].add(case_id)
    return DevIndex(frozenset(ids), frozenset(keys), {k: frozenset(v) for k, v in owners.items()}, counts)


def shingle_document_frequency(prompts: Iterable[str]) -> Counter:
    frequency: Counter = Counter()
    for prompt_text in prompts:
        frequency.update(shingles(prompt_text))
    return frequency


def exclusion_reasons(row: Mapping[str, Any], dev: DevIndex, frequency: Mapping[str, int]) -> list[str]:
    """Why a TRAIN row must stay out of calibration and KLD text (empty: it may enter)."""
    reasons = []
    row_id = str(row["id"])
    if row_id in dev.ids:
        reasons.append("dev_id")
    prompt_text, body, operation = str(row["prompt_text"]), str(row["target_body_text"]), str(row["target_operation"])
    for key in sorted(near_duplicate_keys(prompt_text, body, operation) & dev.keys):
        reasons.append({"p": "dev_prompt_hash", "r": "dev_region_hash", "t": "dev_target_hash"}[key[0]])
    found = {s for s in shingles(prompt_text) if frequency.get(s, 0) <= SHINGLE_MAX_DOCUMENT_FREQUENCY}
    shared: Counter = Counter()
    for shingle in found:
        for owner in dev.shingle_owners.get(shingle, ()):
            shared[owner] += 1
    for owner, count in shared.items():
        smaller = min(len(found), dev.shingle_counts[owner])
        if count >= SHINGLE_MIN_SHARED and smaller and count / smaller >= SHINGLE_CONTAINMENT:
            reasons.append("dev_shingle_containment")
            break
    return sorted(set(reasons))


def mentions_added_token(text: str, added_tokens: Iterable[str]) -> bool:
    return any(token in text for token in added_tokens)


def family_balanced(rows: Sequence[Mapping[str, Any]], budget_tokens: int, seed: int,
                    salt: str = "calibration") -> list[Mapping[str, Any]]:
    """Pick about ``budget_tokens / families`` tokens per family, in a seeded order.

    Within a family, rows are sorted by id and shuffled with
    ``random.Random(f"{seed}:{salt}:{family}")``; rows are taken until the
    family's share is reached.  The result is grouped by family.
    """
    if budget_tokens < 1:
        raise ValueError("calibration budget must be positive")
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    if not by_family:
        raise ValueError("no eligible rows")
    share = budget_tokens // len(by_family)
    selected = []
    for family in sorted(by_family):
        pool = sorted(by_family[family], key=lambda r: str(r["id"]))
        random.Random(f"{seed}:{salt}:{family}").shuffle(pool)
        total = 0
        for row in pool:
            if total >= share:
                break
            selected.append(row)
            total += len(row["input_ids"])
    return selected


def seeded_order(rows: Sequence[Mapping[str, Any]], seed: int, salt: str) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda r: str(r["id"]))
    random.Random(f"{seed}:{salt}").shuffle(ordered)
    return ordered


@dataclass(frozen=True)
class Piece:
    """Tokens ``input_ids[start:end]`` of one row; ``whole`` pieces are the full row."""
    row_id: str
    start: int
    end: int
    whole: bool


def pack_chunks(rows: Sequence[Mapping[str, Any]], n_ctx: int,
                fillers: Iterator[Mapping[str, Any]],
                prefix_ok: Callable[[Sequence[int]], bool]) -> tuple[list[list[Piece]], list[str]]:
    """Pack whole rows into chunks of exactly ``n_ctx`` tokens.

    Rows are taken in order with first-fit over the next ``FILLER_LOOKAHEAD``
    queued rows.  A gap that no queued row fits is filled by the prefix of
    the next filler row whose body round-trips (``prefix_ok`` on the prefix
    without its BOS); that piece goes first in the chunk, so every chunk ends
    with a whole row's EOS.  Returns the chunks and the filler row ids used.
    """
    queue = deque(rows)
    chunks: list[list[Piece]] = []
    used_fillers: list[str] = []
    for row in rows:
        if len(row["input_ids"]) > n_ctx:
            raise ValueError(f"row {row['id']} is longer than the calibration context")
    while queue:
        pieces: list[Piece] = []
        used = 0
        while True:
            remaining = n_ctx - used
            pick = next((index for index, row in enumerate(list(queue)[:FILLER_LOOKAHEAD])
                         if len(row["input_ids"]) <= remaining), None)
            if pick is None:
                break
            row = queue[pick]
            del queue[pick]
            pieces.append(Piece(str(row["id"]), 0, len(row["input_ids"]), True))
            used += len(row["input_ids"])
        remaining = n_ctx - used
        if remaining:
            for filler in fillers:
                ids = filler["input_ids"]
                # The prefix must stop before the EOS and leave whole tokens that round-trip.
                if len(ids) - 1 > remaining and prefix_ok(ids[1:remaining]):
                    pieces.insert(0, Piece(str(filler["id"]), 0, remaining, False))
                    used_fillers.append(str(filler["id"]))
                    break
            else:
                raise ValueError("ran out of filler rows")
        chunks.append(pieces)
    return chunks, used_fillers


def render_calibration(chunks: Sequence[Sequence[Piece]], rows_by_id: Mapping[str, Mapping[str, Any]],
                       decode: Callable[[Sequence[int]], str]) -> tuple[str, list[int]]:
    """Return the calibration text and the token stream llama.cpp must read from it."""
    parts: list[str] = []
    expected: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        for piece_index, piece in enumerate(chunk):
            ids = list(rows_by_id[piece.row_id]["input_ids"])[piece.start:piece.end]
            if piece.start != 0 or ids[0] != BOS_ID:
                raise ValueError("every piece starts at its row's BOS")
            if piece.whole and (ids[-1] != EOS_ID or EOS_ID in ids[1:-1]):
                raise ValueError("a whole row ends with its only EOS")
            body = ids[1:-1] if piece.whole else ids[1:]
            if BOS_ID in body or EOS_ID in body:
                raise ValueError("row body holds BOS or EOS")
            lead = "" if chunk_index == 0 and piece_index == 0 else BOS_TEXT
            parts.append(lead + decode(body) + (EOS_TEXT if piece.whole else ""))
            expected.extend(ids)
    return "".join(parts), expected


def kld_windows(rows: Sequence[Mapping[str, Any]], n_ctx: int, count: int, seed: int,
                window_ok: Callable[[Sequence[int]], bool]) -> list[Piece]:
    """Family-balanced windows: the last ``n_ctx - 1`` tokens of a row before its EOS.

    Families take turns in sorted order; inside a family rows follow a seeded
    order.  Rows too short for a full window, or whose window does not
    round-trip, are skipped.
    """
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        if len(row["input_ids"]) - 2 >= n_ctx - 1:
            by_family[str(row["family"])].append(row)
    pools = {family: deque(seeded_order(pool, seed, f"kld:{family}")) for family, pool in sorted(by_family.items())}
    windows: list[Piece] = []
    while len(windows) < count and any(pools.values()):
        for family in sorted(pools):
            pool = pools[family]
            while pool:
                row = pool.popleft()
                ids = row["input_ids"]
                end = len(ids) - 1
                start = end - (n_ctx - 1)
                if window_ok(ids[start:end]):
                    windows.append(Piece(str(row["id"]), start, end, False))
                    break
            if len(windows) >= count:
                break
    if len(windows) < count:
        raise ValueError(f"only {len(windows)} KLD windows available, {count} requested")
    return windows


def render_kld(windows: Sequence[Piece], rows_by_id: Mapping[str, Mapping[str, Any]],
               decode: Callable[[Sequence[int]], str]) -> tuple[str, list[int]]:
    """Return the held-out text and the stream llama-perplexity stores in its base file."""
    parts: list[str] = []
    expected: list[int] = []
    for index, window in enumerate(windows):
        ids = list(rows_by_id[window.row_id]["input_ids"])[window.start:window.end]
        if BOS_ID in ids or EOS_ID in ids:
            raise ValueError("KLD windows hold no BOS or EOS")
        if index == 0:
            parts.append(decode(ids))
            expected.extend([BOS_ID, *ids])
        else:
            parts.append(KLD_CHUNK_MARK_TEXT + decode(ids))
            expected.extend([KLD_CHUNK_MARK_ID, *ids])
    return "".join(parts), expected


def first_mismatch(actual: Sequence[int], expected: Sequence[int]) -> int | None:
    """Index of the first differing token (or the shorter length), None if equal."""
    for index, (a, b) in enumerate(zip(actual, expected, strict=False)):
        if a != b:
            return index
    return None if len(actual) == len(expected) else min(len(actual), len(expected))
