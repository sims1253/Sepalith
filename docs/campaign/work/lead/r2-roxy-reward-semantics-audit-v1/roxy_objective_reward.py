"""TRAIN-only, target-free objective checks for roxygen candidate diagnostics.

The checker sees a candidate and a function signature extracted from the
visible source context.  It never sees the reference documentation.  Passing
means only that the candidate is structurally plausible and covers the visible
formal parameters; it does not establish descriptive or behavioral accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


SAFE_TAGS = {
    "param", "return", "returns", "description", "details", "title",
    "noRd", "export", "keywords",
}


@dataclass(frozen=True)
class SignatureEvidence:
    function_name: str
    parameters: tuple[str, ...]
    source_line_offset: int


def _balanced_payload(text: str, open_index: int) -> str:
    depth = 0
    quote = None
    escaped = False
    for i in range(open_index, len(text)):
        char = text[i]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char == "#":
            newline = text.find("\n", i)
            if newline < 0:
                break
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
    raise ValueError("visible function signature is incomplete")


def _split_formals(payload: str) -> tuple[str, ...]:
    pieces, start, depth, quote, escaped = [], 0, 0, None, False
    for i, char in enumerate(payload):
        if quote:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == quote: quote = None
            continue
        if char in "'\"`": quote = char
        elif char in "([{": depth += 1
        elif char in ")]}": depth -= 1
        elif char == "," and depth == 0:
            pieces.append(payload[start:i]); start = i + 1
    pieces.append(payload[start:])
    names = []
    for raw in pieces:
        value = raw.strip()
        if not value: continue
        name = value.split("=", 1)[0].strip()
        if name.startswith("`") and name.endswith("`"): name = name[1:-1]
        if not re.fullmatch(r"(?:\.\.\.|[A-Za-z.][A-Za-z0-9._]*)", name):
            raise ValueError(f"unsupported visible formal: {name}")
        if name in names: raise ValueError(f"duplicate visible formal: {name}")
        names.append(name)
    return tuple(names)


def extract_visible_signature(suffix_lines: Sequence[str]) -> SignatureEvidence:
    text = "\n".join(suffix_lines)
    match = re.search(r"(?m)^\s*(`[^`]+`|[A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=)\s*function\s*\(", text)
    if not match:
        raise ValueError("no unambiguous visible function assignment")
    open_index = text.find("(", match.start())
    payload = _balanced_payload(text, open_index)
    name = match.group(1).strip("`")
    return SignatureEvidence(name, _split_formals(payload), text[:match.start()].count("\n"))


def check_roxygen_candidate(candidate: str, signature: SignatureEvidence) -> dict:
    lines = candidate.splitlines()
    errors, unsupported = [], []
    if not lines or not any(line.strip() for line in lines): errors.append("empty_documentation")
    content = []
    for line in lines:
        if line.strip() and not re.match(r"^\s*#'", line):
            errors.append("non_roxygen_line")
            continue
        content.append(re.sub(r"^\s*#'\s?", "", line))
    tags, prose = [], []
    for body in content:
        stripped = body.strip()
        if not stripped: continue
        if stripped.startswith("@"):
            match = re.match(r"@([A-Za-z][A-Za-z0-9]*)\b\s*(.*)$", stripped)
            if not match:
                errors.append("malformed_tag"); continue
            tag, value = match.groups(); tags.append((tag, value.strip()))
            if tag not in SAFE_TAGS: unsupported.append(tag)
        else:
            prose.append(stripped)
    if not prose and not any(tag in {"title", "description"} and value for tag, value in tags):
        errors.append("missing_nonempty_description")
    seen = []
    for tag, value in tags:
        if tag != "param": continue
        match = re.match(r"([^\s]+)\s+(.+)$", value)
        if not match:
            errors.append("empty_param_description"); continue
        raw_names = match.group(1).split(",")
        for raw in raw_names:
            name = raw.strip().strip("`")
            if not name: errors.append("malformed_param_name"); continue
            if name in seen: errors.append(f"duplicate_param:{name}")
            seen.append(name)
            if name not in signature.parameters: errors.append(f"invented_param:{name}")
    missing = [name for name in signature.parameters if name not in seen]
    errors.extend(f"missing_param:{name}" for name in missing)
    if unsupported: errors.append("unsupported_tags_require_source_resolution")
    result = {
        "schema": "sepalith.roxy-objective-check.v1",
        "signature": asdict(signature), "candidate_parameters": seen,
        "unsupported_tags": sorted(set(unsupported)), "errors": sorted(set(errors)),
    }
    result["objective_pass"] = not result["errors"]
    result["semantic_correctness"] = "not_measured"
    result["proposed_reward_effect"] = "no_positive_credit" if result["objective_pass"] else "eligible_for_narrow_penalty_after_root_review"
    return result
