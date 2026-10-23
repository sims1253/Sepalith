"""Shared helpers: deterministic RNG, R lexing, novelty and structural similarity."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import random
import re

STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
IDENT_RE = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9._])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?L?")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# Lexer for structural comparison: strings, numbers, identifiers, operators.
LEX_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?L?|'
                    r"[A-Za-z.][A-Za-z0-9._]*|%[^%\s]*%|<-|<<-|->|\|>|==|!=|<=|>=|&&|\|\||::|[^\sA-Za-z0-9]")
KEYWORDS = frozenset("""if else for while repeat function return next break TRUE FALSE NULL NA NA_integer_
NA_real_ NA_character_ Inf NaN in T F""".split())
# Ubiquitous base functions a reader expects without seeing them in context.
COMMON = frozenset("""c list length paste paste0 sprintf stop warning message is.null invisible seq_along
seq_len vapply lapply sapply names nrow ncol data.frame character numeric integer logical vector unlist
do.call identical any all sum mean max min round abs which rep cat print format stopifnot inherits structure
class is.na nchar unique sort order setdiff union match.arg missing on.exit tryCatch is.numeric is.character
is.logical is.function is.list as.character as.numeric as.integer x i j n out res result value self private
super call. na.rm drop stringsAsFactors envir USE.NAMES FUN.VALUE SIMPLIFY collapse sep fixed perl""".split())
TRIVIAL_NUMBERS = frozenset({"0", "1", "2", "0L", "1L", "2L", "1.0", "0.0"})


def rng_for(*parts: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def split_comment(line: str) -> tuple[str, str]:
    """(code, comment) of one line; ``#`` inside string literals is code."""
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
            return line[:index], line[index:]
    return line, ""


def strip_comment(line: str) -> str:
    return split_comment(line)[0]


def lexical_items(lines: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    """(identifiers, string/comment words, numbers) of the given lines.

    Comments (including roxygen ``#'``) are split off first, so a ``'`` in a
    comment never opens a string. String literals contribute their words.
    """
    idents: set[str] = set()
    words: set[str] = set()
    numbers: set[str] = set()
    for line in lines:
        code, comment = split_comment(line)
        for literal in STRING_RE.findall(code):
            words.update(w.lower() for w in WORD_RE.findall(literal[1:-1]))
        bare = STRING_RE.sub(" ", code)
        numbers.update(NUMBER_RE.findall(bare))
        idents.update(t for t in IDENT_RE.findall(bare) if t not in KEYWORDS)
        words.update(w.lower() for w in WORD_RE.findall(comment))
    return idents, words, numbers


def novelty(target_lines: Sequence[str], prompt_lines: Sequence[str]) -> list[str]:
    """Items of the target's code absent from the prompt.

    Counted: identifiers of two or more characters not in the prompt (nor in
    ``COMMON``); words inside string literals not seen anywhere in the prompt
    (code, strings or comments, case-insensitive); numbers other than 0-2 not
    in the prompt. Target comments are not counted (they are free prose).
    """
    p_idents, p_words, p_numbers = lexical_items(prompt_lines)
    seen_words = p_words | {i.lower() for i in p_idents}
    out = []
    for line in target_lines:
        code, _comment = split_comment(line)
        for literal in STRING_RE.findall(code):
            for word in WORD_RE.findall(literal[1:-1]):
                if word.lower() not in seen_words:
                    out.append("str:" + word)
        bare = STRING_RE.sub(" ", code)
        for number in NUMBER_RE.findall(bare):
            if number not in TRIVIAL_NUMBERS and number not in p_numbers:
                out.append("num:" + number)
        for ident in IDENT_RE.findall(bare):
            if len(ident) >= 2 and ident not in KEYWORDS and ident not in COMMON and ident not in p_idents:
                out.append(ident)
    return sorted(set(out))


def masked_tokens(lines: Iterable[str], *, keep_calls: bool = False) -> list[str]:
    """Identifier/string/number-masked token stream of the code (comments dropped).

    With ``keep_calls``, called function names (an identifier followed by
    ``(``), namespaces and ``COMMON`` names stay visible; only local names,
    literals and numbers are masked.
    """
    tokens = []
    for line in lines:
        tokens.extend(LEX_RE.findall(strip_comment(line)))
    out = []
    for index, token in enumerate(tokens):
        if token[0] in "\"'":
            out.append("S")
        elif token[0].isdigit():
            out.append("N")
        elif IDENT_RE.fullmatch(token) and token not in KEYWORDS:
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            keep = keep_calls and (nxt in ("(", "::") or token in COMMON)
            out.append(token if keep else "I")
        else:
            out.append(token)
    return out


def code_tokens(lines: Iterable[str]) -> list[str]:
    """Formatting-invariant token stream of the code (comments and whitespace dropped)."""
    out = []
    for line in lines:
        out.extend(LEX_RE.findall(strip_comment(line)))
    return out


def structural_support(target_lines: Sequence[str], prompt_lines: Sequence[str], k: int = 8) -> float:
    """Share of the target's masked k-grams that also occur in the prompt's code.

    High values mean the target's structure is already shown in the prompt
    (a sibling function, a mirrored branch, a repeated pattern).
    """
    target = masked_tokens(target_lines, keep_calls=True)
    prompt = masked_tokens(prompt_lines, keep_calls=True)
    grams = [tuple(target[i:i + k]) for i in range(len(target) - k + 1)]
    if not grams:
        return 0.0
    seen = {tuple(prompt[i:i + k]) for i in range(len(prompt) - k + 1)}
    return sum(g in seen for g in grams) / len(grams)


def word_in(name: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9._])" + re.escape(name) + r"(?![A-Za-z0-9._])", text) is not None


CALL_NAME_RE = re.compile(r"([A-Za-z.][A-Za-z0-9._]*)\s*\(")


def assess(target_lines: Sequence[str], prompt_lines: Sequence[str],
           package_functions: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """v3 determinacy evidence: (novel items, hard rejections).

    Novel items (budgeted by the determinacy class): code identifiers of two or
    more characters absent from the prompt (``COMMON`` names excepted) and
    numbers other than 0-2 absent from the prompt.

    Hard rejections (the target needs content the prompt does not specify):
    - ``unspecified_comment:<word>``: a target comment uses a word that occurs
      nowhere in the prompt (comments verbatim in the prompt, or recombining
      prompt words, are allowed);
    - ``unspecified_literal:<word>``: a string literal (including glue, sprintf
      and SQL text, whose embedded identifiers are words here) uses a word that
      occurs nowhere in the prompt;
    - ``invisible_helper:<name>``: the target calls a function defined in the
      seed package that the prompt never mentions.
    """
    p_idents, p_words, p_numbers = lexical_items(prompt_lines)
    seen_words = p_words | {i.lower() for i in p_idents}
    prompt_text = "\n".join(prompt_lines)
    helpers = set(package_functions)
    novel: list[str] = []
    hard: list[str] = []
    for line in target_lines:
        code, comment = split_comment(line)
        if comment and comment.strip()[1:].strip() not in prompt_text:
            for word in WORD_RE.findall(comment):
                if word.lower() not in seen_words:
                    hard.append("unspecified_comment:" + word)
        for literal in STRING_RE.findall(code):
            if literal in prompt_text:
                continue
            for word in WORD_RE.findall(literal[1:-1]):
                if word.lower() not in seen_words:
                    hard.append("unspecified_literal:" + word)
        bare = STRING_RE.sub(" ", code)
        for number in NUMBER_RE.findall(bare):
            if number not in TRIVIAL_NUMBERS and number not in p_numbers:
                novel.append("num:" + number)
        for ident in IDENT_RE.findall(bare):
            # A name the prompt shows as an identifier, or as a word in a string or comment
            # (e.g. the choices of match.arg(), column names, list element names), is visible.
            if (len(ident) >= 2 and ident not in KEYWORDS and ident not in COMMON and ident not in p_idents
                    and ident.lower() not in p_words):
                novel.append(ident)
        for call in CALL_NAME_RE.findall(bare):
            if call in helpers and call not in p_idents:
                hard.append("invisible_helper:" + call)
    return sorted(set(novel)), sorted(set(hard))
