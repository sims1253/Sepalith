"""Near-duplicate filtering and decontamination for SYNTH-v1.

Three measures, all on normalised code text (whitespace collapsed):

- line 4-grams: hashes of 4 consecutive non-trivial lines (>= 12 characters
  after normalisation). A shared 4-gram means four lines copied verbatim.
- shingle containment: the share of a case's 5-token shingles found in a
  reference corpus (TRAIN or DEV), held as a sorted ``uint64`` array.
- MinHash/LSH (64 permutations, 16 bands of 4) over 5-token shingles for
  near duplicates inside the synthetic set (Jaccard >= threshold).

Reference corpora are only hashed. Nothing from DEV is stored or printed;
callers report aggregate counts.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import hashlib
import json
import re

import numpy as np

WORD_RE = re.compile(r"[A-Za-z0-9_.]+|[^\sA-Za-z0-9_.]")
SPACE_RE = re.compile(r"\s+")
MASK = (1 << 61) - 1
PERMUTATIONS = 64
BANDS = 16


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def norm_line(line: str) -> str:
    return SPACE_RE.sub(" ", line.strip())


def line_4grams(text: str) -> set[int]:
    lines = [n for n in (norm_line(line) for line in text.split("\n")) if len(n) >= 12]
    return {h64("\n".join(lines[i:i + 4])) for i in range(len(lines) - 3)}


def shingles(text: str, k: int = 5) -> set[int]:
    words = WORD_RE.findall(text)
    if len(words) < k:
        return {h64(" ".join(words))} if words else set()
    return {h64(" ".join(words[i:i + k])) for i in range(len(words) - k + 1)}


def strings_in(value: object) -> Iterable[str]:
    """Every string inside a JSON value (keys excluded)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


def texts_of_file(path: str) -> Iterable[str]:
    if path.endswith((".jsonl", ".json")):
        with open(path, encoding="utf-8") as stream:
            if path.endswith(".json"):
                yield from (s for s in strings_in(json.load(stream)) if len(s) >= 40)
                return
            for line in stream:
                if line.strip():
                    yield from (s for s in strings_in(json.loads(line)) if len(s) >= 40)
    else:
        with open(path, encoding="utf-8", errors="replace") as stream:
            yield stream.read()


class ReferenceIndex:
    """Hashed reference corpus: line 4-grams (set) and shingles (sorted array)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.grams: set[int] = set()
        self._shingles: list[np.ndarray] = []
        self.shingle_array = np.zeros(0, dtype=np.uint64)
        self.documents = 0

    def add(self, text: str, *, with_shingles: bool = True) -> None:
        self.documents += 1
        self.grams |= line_4grams(text)
        if with_shingles:
            values = shingles(text)
            if values:
                self._shingles.append(np.fromiter(values, dtype=np.uint64, count=len(values)))

    def freeze(self) -> None:
        if self._shingles:
            self.shingle_array = np.unique(np.concatenate([self.shingle_array, *self._shingles]))
            self._shingles = []

    def containment(self, values: set[int]) -> float:
        if not values or self.shingle_array.size == 0:
            return 0.0
        query = np.fromiter(values, dtype=np.uint64, count=len(values))
        position = np.searchsorted(self.shingle_array, query)
        position[position >= self.shingle_array.size] = 0
        return float(np.mean(self.shingle_array[position] == query))

    def gram_hits(self, values: set[int]) -> int:
        return len(values & self.grams)


def _signature(values: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Universal hashing modulo 2**64 (wrapping uint64 arithmetic)."""
    x = np.fromiter(values, dtype=np.uint64, count=len(values))
    with np.errstate(over="ignore"):
        return (x[None, :] * a[:, None] + b[:, None]).min(axis=1)


def near_duplicates(keys: Sequence[set[int]], threshold: float, seed: int = 3407) -> list[int | None]:
    """For each item, the index of an earlier kept item it duplicates, else None."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MASK, size=PERMUTATIONS, dtype=np.int64).astype(np.uint64) | np.uint64(1)
    b = rng.integers(0, MASK, size=PERMUTATIONS, dtype=np.int64).astype(np.uint64)
    rows = PERMUTATIONS // BANDS
    buckets: dict[tuple[int, tuple], list[int]] = defaultdict(list)
    out: list[int | None] = []
    for index, values in enumerate(keys):
        if not values:
            out.append(None)
            continue
        sig = _signature(values, a, b)
        candidates: set[int] = set()
        bands = [(band, tuple(sig[band * rows:(band + 1) * rows].tolist())) for band in range(BANDS)]
        for band in bands:
            candidates.update(buckets.get(band, ()))
        match = None
        for other in sorted(candidates):
            union = len(values | keys[other])
            if union and len(values & keys[other]) / union >= threshold:
                match = other
                break
        out.append(match)
        if match is None:
            for band in bands:
                buckets[band].append(index)
    return out
