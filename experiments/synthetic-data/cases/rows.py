"""Registered target constructors: turn (corpus item, model target, meta)
into a scenario-shaped row. Registered by target_construction.kind.

  comment_prefix    comment families: the model's comment becomes the last
                    prefix line above the real (corpus) code block
  cursor_completion completion families: the model's completion becomes
                    region_new at the mid-line cursor; the corpus original
                    is kept in `corpus_target` for provenance

Also the target normalizer registry (spec field target_normalizer):
"comment" strips stray '#'/quotes (comment_to_code.normalize_comment),
"code" trims whitespace, "raw" is the identity.

generate.py merges provenance/backend fields on top of what these return.
"""
from __future__ import annotations

import re

REGISTRY: dict = {}
NORMALIZERS: dict = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def build_row(spec, item: dict, target: str, meta: dict) -> dict:
    kind = spec.target_construction.get("kind")
    if kind not in REGISTRY:
        raise KeyError(f"unknown target_construction {kind!r}; "
                       f"registered: {sorted(REGISTRY)}")
    params = dict(spec.target_construction.get("params") or {})
    row = REGISTRY[kind](spec, item, target, params, meta)
    row.setdefault("family", spec.family)
    row.setdefault("package", item.get("package", "?"))
    row.setdefault("path", item.get("path", "?"))
    row.setdefault("cursor_idx", 0)
    row.setdefault("event_diff", "")
    row.setdefault("note", "")
    row.setdefault("suffix", list(item.get("suffix") or []))
    if item.get("corpus_target") is not None and "corpus_target" not in row:
        row["corpus_target"] = item["corpus_target"]
    return row


@register("comment_prefix")
def rc_comment(spec, item: dict, target: str, params: dict, meta: dict) -> dict:
    indent = params.get("comment_indent", "  # ")
    comment = target.strip().strip()
    prefix = list(item["prefix"]) + [f"{indent}{comment}"]
    note = (f"{meta.get('model', '?')} comment for a "
            f"{len(item['block'])}-line block (case {spec.name})")
    return {
        "prefix": prefix,
        "region_old": [""],
        "region_new": list(item["block"]),
        "note": note,
    }


@register("cursor_completion")
def rc_completion(spec, item: dict, target: str, params: dict, meta: dict) -> dict:
    lines = [l for l in target.split("\n")]
    note = (f"{meta.get('model', '?')} completion of the elided corpus "
            f"expression (case {spec.name}; original kept in corpus_target)")
    return {
        "prefix": list(item["prefix"]),
        "region_old": [""],
        "region_new": lines,
        "note": note,
    }


# ---------------------------------------------------------------------------
# target normalizers
# ---------------------------------------------------------------------------

def register_norm(name: str):
    def deco(fn):
        NORMALIZERS[name] = fn
        return fn
    return deco


def normalize_target(name: str, text: str) -> str:
    fn = NORMALIZERS.get(name or "raw")
    if fn is None:
        raise KeyError(f"unknown target_normalizer {name!r}; "
                       f"registered: {sorted(NORMALIZERS)}")
    return fn(text)


@register_norm("raw")
def n_raw(text: str) -> str:
    return text


@register_norm("code")
def n_code(text: str) -> str:
    return "\n".join(l.rstrip() for l in text.strip().splitlines())


@register_norm("comment")
def n_comment(text: str) -> str:
    """comment_to_code.normalize_comment: models sometimes include the '#',
    quote the string, or wrap the wording across lines."""
    if not isinstance(text, str):
        return ""
    c = text.strip()
    c = c.lstrip("#").strip()
    c = re.sub(r"\s+", " ", c)
    if c.startswith('"') and c.endswith('"') and len(c) > 1:
        c = c[1:-1].strip()
    return c
