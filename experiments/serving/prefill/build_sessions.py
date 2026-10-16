#!/usr/bin/env python3
"""Freeze PRM03 edit-session traces from TRAIN contexts for prefill benching.

Builds the frozen session set used by ``bench.py``: realistic per-file event
sequences (open, typing bursts, an accepted suggestion, a new diagnostic, a
same-file cursor move, a file switch and switch back) rendered through
``sepalith.campaign_protocol.render_prompt`` (zeta2-prm03-v1).  Everything is
derived from the DAT10 TRAIN context sidecar via seeded, deterministic
mutations; the manifest pins source hashes, row ids, the seed and the sha256
of every rendered prompt so the bench can prove it replays the same workload.

Data rules (brief §5): TRAIN rows only; the sealed final set and pre-split
pools are never read.  Idempotent: re-running with the same manifest seed
reproduces byte-identical sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))

from sepalith.campaign_protocol import (  # noqa: E402
    EOS_ID,
    TERMINAL,
    PromptContext,
    ProtocolError,
    render_prompt,
    utf16_length,
)

DEFAULT_SOURCE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT10-finish-source-repair-v4-braceindent/context-sidecar.jsonl"
)
DEFAULT_TOKEN_ROWS = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl"
)
DEFAULT_TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)
SEED = 20261017
N_SESSIONS = 20

# Realistic typing fragments (R code an analyst would type inside a function
# body).  Each fragment is consumed word-by-word across consecutive pauses.
TYPING_FRAGMENTS = (
    "x <- x[order(group, decreasing = TRUE)]",
    "if (anyNA(x)) x <- safena(x)",
    "out <- list(a = x, b = group)",
    "attr(out, \"sorted\") <- TRUE",
    "res <- tryCatch(mean(x), error = function(e) NA_real_)",
)
DIAGNOSTIC_TEMPLATES = (
    "warning: no visible binding for global variable '{name}'",
    "info: variable '{name}' is used before assignment",
    "style: put spaces around the assignment operator in '{name}'",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rng(row_id: str, seed: int) -> random.Random:
    return random.Random(hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest())


def _clone(context: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(context)))


def _cursor_line(context: Mapping[str, Any]) -> tuple[list[str], int]:
    region = list(context["region_old"])
    index = context["cursor"]["region_line_index"]
    if index < 0:
        index = 0
    return region, index


def _set_cursor(context: dict[str, Any], index: int, column: int) -> None:
    line = context["region_old"][index]
    context["cursor"] = {
        "region_line_index": index,
        "code_point_column": column,
        "utf16_column": utf16_length(line[:column]),
    }


def _bump_range(context: dict[str, Any], region: Sequence[str]) -> None:
    """Recompute replacement_range geometry for a new region_old."""
    span = context["replacement_range"]
    start = span["start"]
    if not region:
        end = dict(start)
    elif len(region) == 1:
        end = {"line": start["line"], "character": start["character"] + utf16_length(region[0])}
    else:
        end = {"line": start["line"] + len(region) - 1, "character": utf16_length(region[-1])}
    context["replacement_range"] = {
        **span,
        "end": end,
        "document_version": span["document_version"] + 1,
        "content_sha256": hashlib.sha256("\n".join(region).encode()).hexdigest(),
    }


def _apply_typing(context: dict[str, Any], rng: random.Random, chunk: str) -> None:
    """Insert a typing burst at the cursor; the cursor advances past it."""
    region = list(context["region_old"])
    if not region:
        region = [chunk]
        context["region_old"] = region
        _set_cursor(context, 0, len(chunk))
        _bump_range(context, region)
        return
    index = context["cursor"]["region_line_index"]
    if index < 0:
        index = 0
    column = context["cursor"]["code_point_column"] or 0
    line = region[index]
    region[index] = line[:column] + chunk + line[column:]
    context["region_old"] = region
    _set_cursor(context, index, column + len(chunk))
    _bump_range(context, region)


def _history_event(
    context: Mapping[str, Any], old_text: str, new_text: str, rng: random.Random
) -> dict[str, Any]:
    """Build an observed-edit history event in the sidecar's wire format."""
    path = context["path"]
    span = context["replacement_range"]
    line = span["start"]["line"] + 1
    event_id = hashlib.sha256(
        f"{path}:{old_text}:{new_text}:{rng.random()}".encode()
    ).hexdigest()[:24]
    diff = (
        f'User edited "{path}":\n\n'
        f"```diff\n@@ -{line} +{line} @@\n-{old_text}\n+{new_text}\n```"
    )
    return {
        "event_id": event_id,
        "kind": "observed_edit_diff",
        "path": path,
        "workspace_revision_before": span["content_sha256"],
        "old_text": old_text,
        "new_text": new_text,
        "range_utf16": {
            "uri": span["uri"],
            "document_version": span["document_version"],
            "content_sha256": span["content_sha256"],
            "start": dict(span["start"]),
            "end": dict(span["end"]),
        },
        "event_diff": diff,
    }


def _target_body(target_ids: Sequence[int], decode) -> tuple[str, bool]:  # noqa: ANN001
    """Decode a training target into (body_text, is_no_op)."""
    ids = list(target_ids)
    while ids and ids[-1] in (EOS_ID,):
        ids.pop()
    text = decode(ids)
    if text.endswith("\n" + TERMINAL):
        text = text[: -len("\n" + TERMINAL)]
    elif text.endswith(TERMINAL):
        text = text[: -len(TERMINAL)]
    return text, text.strip() == "[NO_EDIT]"


def _apply_accept(context: dict[str, Any], body: str, rng: random.Random) -> None:
    """Accept a suggestion: region becomes the body, history grows."""
    region = body.split("\n") if body else []
    old_text = "\n".join(context["region_old"])
    history = list(context["history"])
    history.append(_history_event(context, old_text, body, rng))
    context["history"] = history
    context["region_old"] = region
    if region:
        _set_cursor(context, len(region) - 1, len(region[-1]))
    else:
        context["cursor"] = {
            "region_line_index": -1,
            "code_point_column": None,
            "utf16_column": None,
        }
    _bump_range(context, region)


def _apply_diagnostic(context: dict[str, Any], rng: random.Random) -> None:
    name = "value"
    region = context["region_old"]
    if region:
        for token in reversed(region[0].replace("(", " ").split()):
            if token.replace(".", "").replace("_", "").isalpha():
                name = token
                break
    template = DIAGNOSTIC_TEMPLATES[rng.randrange(len(DIAGNOSTIC_TEMPLATES))]
    line_no = context["replacement_range"]["start"]["line"] + 1
    content = f"line {line_no}: {template.format(name=name)}"
    context["diagnostics"] = [
        *list(context["diagnostics"]),
        {"content": content, "path": context["path"]},
    ]


def _merge_window(base: Mapping[str, Any], window: Mapping[str, Any]) -> dict[str, Any]:
    """Same-file cursor move: take the partner's window, keep base's history."""
    merged = _clone(base)
    for key in (
        "prefix",
        "suffix_lines",
        "scope_lines",
        "region_old",
        "cursor",
        "replacement_range",
    ):
        merged[key] = json.loads(json.dumps(window["context"][key]))
    return merged


def build_event_sequence(
    base: Mapping[str, Any],
    move: Mapping[str, Any],
    switch: Mapping[str, Any],
    target_ids: Sequence[int],
    decode,  # noqa: ANN001
    seed: int,
) -> list[dict[str, Any]]:
    """One session: open → typing → accept → typing → diag → move → typing → switch → back."""
    rng = _stable_rng(base["row_id"], seed)
    events: list[dict[str, Any]] = []

    def emit(event_type: str, context: Mapping[str, Any]) -> dict[str, Any]:
        event = {"type": event_type, "context": _clone(context)}
        events.append(event)
        return event

    current = _clone(base["context"])

    emit("open", current)

    fragment = TYPING_FRAGMENTS[rng.randrange(len(TYPING_FRAGMENTS))]
    words = fragment.split(" ")
    bursts = max(2, min(4, len(words) // 3))
    per_burst = max(1, len(words) // bursts)
    for i in range(bursts):
        chunk = " ".join(words[i * per_burst : (i + 1) * per_burst])
        if i == bursts - 1:
            chunk = " ".join(words[i * per_burst :])
        _apply_typing(current, rng, chunk)
        emit("type", current)

    body, is_no_op = _target_body(target_ids, decode)
    if is_no_op or not body:
        # No change accepted: the analyst edits manually instead.
        _apply_typing(current, rng, "  # TODO: verify inputs")
        body = "\n".join(current["region_old"])
    _apply_accept(current, body, rng)
    emit("accept", current)

    fragment2 = TYPING_FRAGMENTS[(TYPING_FRAGMENTS.index(fragment) + 1) % len(TYPING_FRAGMENTS)]
    for chunk in (" ".join(fragment2.split(" ")[:3]), fragment2.split(" ", 1)[1] if " " in fragment2 else fragment2):
        _apply_typing(current, rng, chunk)
        emit("type", current)

    _apply_diagnostic(current, rng)
    emit("diag", current)

    before_move = _clone(current)
    current = _merge_window(current, move)
    emit("cursor_move", current)

    fragment3 = TYPING_FRAGMENTS[(TYPING_FRAGMENTS.index(fragment) + 2) % len(TYPING_FRAGMENTS)]
    _apply_typing(current, rng, " ".join(fragment3.split(" ")[:4]))
    emit("type", current)

    emit("file_switch", switch["context"])
    emit("switch_back", before_move)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--token-rows", type=Path, default=DEFAULT_TOKEN_ROWS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sessions", type=int, default=N_SESSIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    import tokenizers  # local import: heavy dependency only needed at freeze time

    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    rows: dict[str, dict[str, Any]] = {}
    with args.token_rows.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["id"]] = row

    sides: list[dict[str, Any]] = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            sides.append(json.loads(line))

    # groups of same (package, path): cursor-move partners
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for side in sides:
        key = (str(side.get("package_id", "?")), side["context"]["path"])
        groups.setdefault(key, []).append(side)

    rng = random.Random(args.seed)
    # length-stratified, family-mix sampling over rows that have a partner
    candidates = [
        side
        for side in sides
        if len(groups[(str(side.get("package_id", "?")), side["context"]["path"])]) >= 2
        and side["row_id"] in rows
    ]
    candidates.sort(key=lambda side: rows[side["row_id"]]["target_start"])
    strata = 4
    per_stratum = args.sessions // strata + 1
    picked: list[dict[str, Any]] = []
    for i in range(strata):
        lo = i * len(candidates) // strata
        hi = (i + 1) * len(candidates) // strata
        stratum = candidates[lo:hi]
        rng.shuffle(stratum)
        picked.extend(stratum[:per_stratum])
    picked = picked[: args.sessions]

    sessions: list[dict[str, Any]] = []
    used_switch: set[str] = set()
    for base in picked:
        key = (str(base.get("package_id", "?")), base["context"]["path"])
        partner = rng.choice([s for s in groups[key] if s["row_id"] != base["row_id"]])
        switch_pool = [s for s in sides if s.get("package_id") != base.get("package_id")]
        switch = rng.choice(switch_pool)
        while switch["row_id"] in used_switch and len(used_switch) < len(sides):
            switch = rng.choice(switch_pool)
        used_switch.add(switch["row_id"])
        row = rows[base["row_id"]]
        target_ids = row["input_ids"][row["target_start"] :]
        events = build_event_sequence(base, partner, switch, target_ids, decode, args.seed)
        sessions.append(
            {
                "session_id": base["row_id"],
                "package_id": base.get("package_id"),
                "family": base.get("family"),
                "base_row_id": base["row_id"],
                "move_row_id": partner["row_id"],
                "switch_row_id": switch["row_id"],
                "events": events,
            }
        )

    # verify + freeze: every event must render and match its pinned sha
    manifest_events: list[dict[str, Any]] = []
    out_path = args.out_dir / "sessions.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for session in sessions:
            for i, event in enumerate(session["events"]):
                context = PromptContext.from_mapping(event["context"])
                text = render_prompt(context)
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                digest = hashlib.sha256(text.encode()).hexdigest()
                manifest_events.append(
                    {
                        "session_id": session["session_id"],
                        "index": i,
                        "type": event["type"],
                        "prompt_sha256": digest,
                        "prompt_tokens": len(ids),
                    }
                )
            out.write(json.dumps(session) + "\n")

    manifest = {
        "schema": "sepalith.prefill.sessions.v1",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "seed": args.seed,
        "n_sessions": len(sessions),
        "source": {
            "path": str(args.source),
            "sha256": _sha256_file(args.source),
            "token_rows_path": str(args.token_rows),
            "token_rows_sha256": _sha256_file(args.token_rows),
            "tokenizer_sha256": _sha256_file(args.tokenizer),
        },
        "sessions_sha256": _sha256_file(out_path),
        "events": manifest_events,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    by_type: dict[str, int] = {}
    for event in manifest_events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1
    lens = [event["prompt_tokens"] for event in manifest_events]
    print(
        json.dumps(
            {
                "sessions": len(sessions),
                "events": len(manifest_events),
                "by_type": by_type,
                "prompt_tokens_p50": sorted(lens)[len(lens) // 2],
                "prompt_tokens_max": max(lens),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as error:
        print(f"context validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
