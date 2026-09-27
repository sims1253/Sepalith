#!/usr/bin/env python3
"""Select source spans that support a roxygen target.

The selector is intentionally source-only and deterministic.  It keeps the
complete top-level definition containing the target function (including its
formal arguments and body), then follows same-file definitions and
expressions referenced by that definition.  It also records formal-name and
roxygen-tag evidence.  The result is an evidence packet for later admission;
preserving an AST anchor is never treated as proof that the natural-language
documentation is semantically correct.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable


POLICY_ID = "dat10-roxy-source-supported-context-v1"
POLICY_REVISION = "2026-09-14"

_ASSIGNMENT_OPERATORS = {"<-", "=", "<<-", "->", "->>", ":="}
_PARAM_TAG = re.compile(r"^\s*#'\s*@param\s+([^\s]+(?:\s*,\s*[^\s]+)*)")
_TAG = re.compile(r"^\s*#'\s*@([A-Za-z][A-Za-z0-9_.-]*)\b")
_BACKTICK_NAME = re.compile(r"`([A-Za-z.][A-Za-z0-9_.]*)`")

# This is only a name-resolution filter.  It does not assert that a function
# call is correct or that a package's documentation is supported.
_COMMON_R_NAMES = {
    "if", "else", "for", "while", "repeat", "in", "next", "break",
    "function", "return", "switch", "try", "tryCatch", "withVisible",
    "local", "on.exit", "missing", " substitute", "quote", "eval", "parse",
    "TRUE", "FALSE", "NULL", "NA", "NaN", "Inf", "T", "F", "...",
    "c", "list", "length", "names", "dim", "nrow", "ncol", "seq", "seq_len",
    "seq_along", "rep", "matrix", "array", "data.frame", "as.data.frame",
    "is.null", "is.na", "isTRUE", "isFALSE", "identical", "all", "any",
    "which", "match", "paste", "paste0", "sprintf", "format", "print", "cat",
    "message", "warning", "stop", "stopifnot", "as.character", "as.numeric",
    "as.integer", "as.logical", "as.list", "as.vector", "unlist", "do.call",
    "apply", "lapply", "vapply", "sapply", "Map", "Reduce", "Filter", "head",
    "tail", "sort", "order", "unique", "duplicated", "subset", "transform",
    "rbind", "cbind", "drop", "sweep", "split", "aggregate", "sum", "mean",
    "sd", "var", "min", "max", "abs", "round", "floor", "ceiling", "sqrt",
    "log", "exp", "lengths", "readLines", "writeLines", "file.exists", "file.path",
    "Sys.getenv", "Sys.time", "requireNamespace", "get", "assign", "environment",
    "parent.frame", "body", "formals", "environment", "class", "inherits",
}


@dataclass(frozen=True)
class Definition:
    """One named top-level R assignment."""

    name: str
    kind: str
    span: tuple[int, int]
    value_span: tuple[int, int]
    identifier_names: frozenset[str]
    local_names: frozenset[str]
    formal_names: frozenset[str]


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _line_span(node: Any, line_count: int, line_starts: list[int] | None = None) -> tuple[int, int]:
    """Return inclusive physical line indexes without retaining point objects.

    The system tree-sitter-r binding can invalidate point objects when a parser
    is reused.  Byte offsets plus immutable line starts are stable and also
    handle CRLF source snapshots without universal-newline rewriting.
    """
    if line_starts is None:
        # This fallback is used only by small unit callers.
        line_starts = [0]
    start_byte = int(node.start_byte)
    end_byte = int(node.end_byte)
    start = bisect_right(line_starts, start_byte) - 1
    end = bisect_right(line_starts, max(start_byte, end_byte - 1)) - 1
    start = max(0, min(start, max(0, line_count - 1)))
    end = max(start, min(end, max(0, line_count - 1)))
    return start, end


def _identifier_name(source: bytes, node: Any) -> str | None:
    if node is None:
        return None
    if node.type not in {"identifier", "namespace_name", "dots"}:
        return None
    text = _node_text(source, node).strip().strip("`")
    return text or None


def _operator(node: Any, source: bytes) -> str | None:
    for child in node.children:
        if child.type in _ASSIGNMENT_OPERATORS:
            return child.type
        text = _node_text(source, child).strip()
        if text in _ASSIGNMENT_OPERATORS:
            return text
    return None


def _walk(node: Any) -> Iterable[Any]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _formal_names(function_node: Any, source: bytes) -> set[str]:
    parameters = function_node.child_by_field_name("parameters")
    if parameters is None:
        return set()
    result: set[str] = set()
    for parameter in parameters.named_children:
        name_node = parameter.child_by_field_name("name")
        if name_node is None and parameter.type == "dots":
            name_node = parameter
        name = _identifier_name(source, name_node)
        if name:
            result.add(name)
    return result


def _local_names(function_node: Any, source: bytes) -> set[str]:
    """Find formal and assigned names without treating binary expressions as assignments."""
    result = _formal_names(function_node, source)
    for node in _walk(function_node):
        if node.type == "binary_operator" and _operator(node, source) in _ASSIGNMENT_OPERATORS:
            lhs = node.child_by_field_name("lhs")
            name = _identifier_name(source, lhs)
            if name:
                result.add(name)
        if node.type == "for_statement":
            # tree-sitter-r exposes the loop binding as the first named child
            # in versions where the `name` field is absent.
            binding = node.child_by_field_name("name") or node.child_by_field_name("left")
            name = _identifier_name(source, binding)
            if name:
                result.add(name)
    return result


def _identifier_names(node: Any, source: bytes) -> set[str]:
    result: set[str] = set()
    for item in _walk(node):
        if item.type == "identifier":
            name = _identifier_name(source, item)
            if name:
                result.add(name)
    return result


def top_level_definitions(tree: Any, source: bytes, line_count: int) -> list[Definition]:
    line_starts = [0]
    for index, value in enumerate(source):
        if value == 10:  # LF; CRLF is one physical line separator.
            line_starts.append(index + 1)
    definitions: list[Definition] = []
    for node in tree.root_node.named_children:
        if node.type != "binary_operator" or _operator(node, source) not in _ASSIGNMENT_OPERATORS:
            continue
        lhs = node.child_by_field_name("lhs")
        rhs = node.child_by_field_name("rhs")
        name = _identifier_name(source, lhs)
        if not name or rhs is None:
            continue
        definitions.append(Definition(
            name=name,
            kind="function" if rhs.type == "function_definition" else "expression",
            span=_line_span(node, line_count, line_starts),
            value_span=_line_span(rhs, line_count, line_starts),
            identifier_names=frozenset(_identifier_names(rhs, source)),
            local_names=frozenset(_local_names(rhs, source)),
            formal_names=frozenset(_formal_names(rhs, source)),
        ))
    return sorted(definitions, key=lambda item: (item.span, item.name, item.kind))


def _target_definition(
    definitions: list[Definition],
    *,
    anchor_line: int,
    source_lines: list[str],
    suffix_lines: list[str],
) -> tuple[Definition | None, str]:
    functions = [item for item in definitions if item.kind == "function"]
    candidates = [item for item in functions if item.span[0] > anchor_line]
    if not candidates:
        return None, "target_function_not_found_after_anchor"
    first_suffix = next((line for line in suffix_lines if line.strip()), None)
    if first_suffix is not None:
        exact = [item for item in candidates if source_lines[item.span[0]].strip() == first_suffix.strip()]
        if exact:
            candidates = exact
    candidates.sort(key=lambda item: (item.span[0] - anchor_line, item.span[1] - item.span[0], item.name))
    chosen = candidates[0]
    relation = "target_function_starts_at_anchor_plus_one" if chosen.span[0] == anchor_line + 1 else "target_function_nearest_after_anchor"
    return chosen, relation


def _tag_evidence(target_lines: list[str], formals: set[str]) -> dict[str, Any]:
    tags: list[dict[str, Any]] = []
    documented: set[str] = set()
    backticked: set[str] = set()
    for line_no, line in enumerate(target_lines):
        tag_match = _TAG.match(line)
        if tag_match:
            tag = tag_match.group(1)
            tags.append({"line_offset": line_no, "tag": tag})
        param_match = _PARAM_TAG.match(line)
        if param_match:
            for raw_name in re.split(r"\s*,\s*", param_match.group(1)):
                name = raw_name.strip().strip("`")
                if name in {r"\ldots", r"\dots"}:
                    name = "..."
                if name:
                    documented.add(name)
        backticked.update(_BACKTICK_NAME.findall(line))
    return {
        "tag_names": sorted({item["tag"] for item in tags}),
        "tag_count": len(tags),
        "param_names": sorted(documented),
        "formal_names": sorted(formals),
        "documented_params_missing_from_formals": sorted(documented - formals),
        "formals_without_param_tags": sorted(formals - documented),
        "backticked_names": sorted(backticked),
        "semantic_support_claim": False,
        "scope": "tag/formal and source-reference evidence only; prose meaning remains pending review",
    }


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted({(int(start), int(end)) for start, end in spans})
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _line_runs(indices: set[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _render_runs(lines: list[str], runs: list[tuple[int, int]]) -> list[str]:
    rendered: list[str] = []
    for index, (start, end) in enumerate(runs):
        if index and start > runs[index - 1][1] + 1:
            # Keep AST nodes visibly separate without smuggling omitted source
            # into the prompt.  This blank is a synthetic layout separator.
            rendered.append("")
        rendered.extend(lines[start:end + 1])
    return rendered


def _digest_ints(values: Iterable[int]) -> str:
    payload = ",".join(str(value) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def select_source_spans(
    *,
    source: bytes,
    source_lines: list[str],
    anchor_line: int,
    target_lines: list[str],
    suffix_lines: list[str],
    parser: Any,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return ``prefix``, ``suffix`` and a source-only evidence record."""
    tree = parser.parse(source)
    if tree.root_node.has_error:
        return [], [], {
            "status": "ast_parse_error",
            "semantic_support_claim": False,
            "selected_line_runs": [],
            "omitted_line_runs": _line_runs(set(range(len(source_lines))) - {anchor_line}),
        }
    definitions = top_level_definitions(tree, source, len(source_lines))
    target, target_relation = _target_definition(
        definitions, anchor_line=anchor_line, source_lines=source_lines, suffix_lines=suffix_lines,
    )
    if target is None:
        return [], [], {
            "status": "target_function_not_found",
            "target_relation": target_relation,
            "definition_count": len(definitions),
            "semantic_support_claim": False,
            "selected_line_runs": [],
            "omitted_line_runs": _line_runs(set(range(len(source_lines))) - {anchor_line}),
        }

    definitions_by_name: dict[str, list[Definition]] = defaultdict(list)
    for definition in definitions:
        definitions_by_name[definition.name].append(definition)

    # The closure includes same-file functions and named expressions needed by
    # the target.  It has no row/family/length quota and is stable under input
    # ordering because names and spans are sorted before serialization.
    selected: dict[tuple[int, int, str], Definition] = {}
    queue: list[Definition] = [target]
    resolved_names: set[str] = set()
    unresolved_names: set[str] = set()
    while queue:
        current = queue.pop(0)
        key = (current.span[0], current.span[1], current.name)
        if key in selected:
            continue
        selected[key] = current
        references = sorted(current.identifier_names - current.local_names - {current.name})
        for name in references:
            matches = definitions_by_name.get(name, [])
            if matches:
                resolved_names.add(name)
                queue.extend(matches)
            elif name not in _COMMON_R_NAMES and not name.startswith("."):
                unresolved_names.add(name)

    tag_evidence = _tag_evidence(target_lines, set(target.formal_names))
    docs_reference_names = sorted(
        name for name in tag_evidence["backticked_names"] if name in definitions_by_name
    )
    for name in docs_reference_names:
        resolved_names.add(name)
        for definition in definitions_by_name[name]:
            if (definition.span[0], definition.span[1], definition.name) not in selected:
                queue.append(definition)
    # Follow documentation references through the same closure.
    while queue:
        current = queue.pop(0)
        key = (current.span[0], current.span[1], current.name)
        if key in selected:
            continue
        selected[key] = current
        for name in sorted(current.identifier_names - current.local_names - {current.name}):
            matches = definitions_by_name.get(name, [])
            if matches:
                resolved_names.add(name)
                queue.extend(matches)
            elif name not in _COMMON_R_NAMES and not name.startswith("."):
                unresolved_names.add(name)

    selected_definitions = sorted(selected.values(), key=lambda item: (item.span, item.name, item.kind))
    selected_spans = _merge_spans(item.span for item in selected_definitions)
    selected_lines = set()
    for start, end in selected_spans:
        selected_lines.update(range(start, end + 1))
    selected_lines.discard(anchor_line)
    prefix_runs = _line_runs({line for line in selected_lines if line < anchor_line})
    suffix_runs = _line_runs({line for line in selected_lines if line > anchor_line})
    omitted_lines = set(range(len(source_lines))) - selected_lines - {anchor_line}
    target_formals = set(target.formal_names)
    missing_docs = set(tag_evidence["documented_params_missing_from_formals"])
    status = "source_spans_selected_pending_semantic_review"
    if missing_docs:
        status = "source_spans_selected_docs_formal_revalidation_required"
    elif unresolved_names:
        status = "source_spans_selected_unresolved_reference_revalidation_required"
    evidence = {
        "status": status,
        "semantic_support_claim": False,
        "anchor_preservation_is_not_semantic_support": True,
        "target_relation": target_relation,
        "target_definition_name": target.name,
        "target_definition_kind": target.kind,
        "target_definition_span": list(target.span),
        "target_function_span": list(target.value_span),
        "target_formal_names": sorted(target_formals),
        "selected_definition_count": len(selected_definitions),
        "selected_definitions": [
            {"name": item.name, "kind": item.kind, "span": list(item.span),
             "is_target": item is target}
            for item in selected_definitions
        ],
        "resolved_reference_names": sorted(resolved_names),
        "unresolved_reference_names": sorted(unresolved_names),
        "docs_reference_names": docs_reference_names,
        "docs_evidence": tag_evidence,
        "selected_source_line_runs": [list(run) for run in _line_runs(selected_lines)],
        "selected_line_runs": [list(run) for run in _line_runs(selected_lines)],
        "selected_line_count": len(selected_lines),
        "omitted_line_runs": [list(run) for run in _line_runs(omitted_lines)],
        "omitted_line_count": len(omitted_lines),
        "selected_line_index_digest": _digest_ints(sorted(selected_lines)),
        "source_line_count": len(source_lines),
        "parser": "tree-sitter-r",
        "policy_id": POLICY_ID,
        "policy_revision": POLICY_REVISION,
    }
    return _render_runs(source_lines, prefix_runs), _render_runs(source_lines, suffix_runs), evidence


def policy_document() -> dict[str, Any]:
    return {
        "schema": "sepalith.dat10.roxy_supported_context_policy.v1",
        "policy_id": POLICY_ID,
        "revision": POLICY_REVISION,
        "purpose": "shared source-span identity for serving and training candidate contexts",
        "selection": [
            "Reconstruct the normalized TRAIN before-state and parse it with tree-sitter-r.",
            "Select the complete top-level definition containing the target function; this includes formals and body.",
            "Resolve same-file named function and expression references to a fixed point, retaining complete definitions.",
            "Record roxygen tags, documented formals, and same-file names referenced by documentation.",
            "Concatenate only selected complete spans with synthetic blank separators; no source bytes are truncated.",
            "Keep every target; contexts exceeding model budgets go to measured queues for a later bounded-context decision.",
        ],
        "guards": {
            "global_split": "inherited frozen global TRAIN source-review join; root admission must recheck",
            "license": "revalidate pinned license path/hash per source",
            "heldout": "TRAIN source scenario and normalized TRAIN paths only; no DEV/final payload reads",
            "target": "complete region_new passed unchanged; no target truncation or rewrite",
            "semantic_support": "anchor and AST reproduction are evidence only, never an admission claim",
        },
        "budget_policy": {
            "context_selection_budget": "unbounded by selector",
            "long_context_queue": [4096, 8192, 16384, 32768, 131072],
            "target_queue_threshold": 1024,
            "overflow_action": "retain row and label measured queue; never silently crop",
        },
    }
