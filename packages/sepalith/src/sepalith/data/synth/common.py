"""Shared helpers: deterministic RNG, R token sets and novelty."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import random
import re

TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[A-Za-z.][A-Za-z0-9._]*|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?L?')
KEYWORDS = frozenset("""if else for while repeat function return next break TRUE FALSE NULL NA NA_integer_
NA_real_ NA_character_ Inf NaN in T F""".split())
# Ubiquitous base functions a reader expects without seeing them in context.
COMMON = frozenset("""c list length paste paste0 sprintf stop warning message is.null invisible seq_along
seq_len vapply lapply sapply names nrow ncol data.frame character numeric integer logical vector unlist
do.call identical any all sum mean max min round abs which rep cat print format stopifnot inherits structure
class is.na nchar unique sort order setdiff union match.arg missing on.exit tryCatch is.numeric is.character
is.logical is.function is.list as.character as.numeric as.integer x i j n out res result value self private
super""".split())


def rng_for(*parts: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def strip_comment(line: str) -> str:
    """Drop a trailing comment outside string literals (single-line heuristic)."""
    quote = None
    escaped = False
    for index, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "#":
            return line[:index]
    return line


def tokens(text: str) -> set[str]:
    out = set()
    for token in TOKEN_RE.findall(text):
        if token in KEYWORDS:
            continue
        if token[0] not in "\"'" and not token[0].isdigit() and len(token) < 2:
            continue
        out.add(token)
    return out


def novelty(target_lines: Iterable[str], prompt_text: str) -> list[str]:
    """Tokens of the target (code only, comments dropped) absent from the prompt."""
    code = "\n".join(strip_comment(line) for line in target_lines)
    seen = tokens(prompt_text)
    return sorted(t for t in tokens(code) if t not in seen and t not in COMMON
                  and t not in ("0", "1", "2", "0L", "1L", "2L"))


def word_in(name: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9._])" + re.escape(name) + r"(?![A-Za-z0-9._])", text) is not None
