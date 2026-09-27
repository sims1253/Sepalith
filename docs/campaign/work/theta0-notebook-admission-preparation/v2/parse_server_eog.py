#!/usr/bin/env python3
"""Require both native EOG IDs in an already-retained server log.

This parser never launches a process or infers EOG from a model/config file.
It accepts only log lines that explicitly carry an EOG/EOS observation marker
and an integer ID.  The output retains line numbers and line hashes, while the
root worker retains the original server log for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "sepalith.run01.theta0-server-eog-observation.v1"
REQUIRED_IDS = (1, 130073)
ID_RE = re.compile(r"(?<!\d)(130073|1)(?!\d)")
MARKER_RE = re.compile(r"\b(?:native[ _-]*)?(?:eog|eos|end[ _-]*of[ _-]*generation)\b", re.I)
OBSERVED_RE = re.compile(r"\b(?:observed|registered|emitted|sampled|terminal|stop|ids?)\b", re.I)


def has_marker(pattern: re.Pattern[str], value: str) -> bool:
    """Treat JSON snake_case event names like their spaced log form."""

    return pattern.search(value.replace("_", " ").replace("-", " ")) is not None


def walk_json(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield child_path, child
            yield from walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_json(child, f"{path}[{index}]")


def ids_from_json(line: str) -> set[int]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return set()
    if not isinstance(value, dict):
        return set()
    labels = " ".join(str(value.get(key, "")) for key in ("event", "type", "name", "message", "status"))
    if not has_marker(MARKER_RE, labels) or not has_marker(OBSERVED_RE, labels):
        # An explicit eog/eos key is itself the observation label, but still
        # requires a status/observed field to avoid counting unrelated config.
        if not any(has_marker(MARKER_RE, path) for path, _ in walk_json(value)):
            return set()
        if not has_marker(OBSERVED_RE, line):
            return set()
    found: set[int] = set()
    for path, child in walk_json(value):
        if not has_marker(MARKER_RE, path) and not re.search(r"(?:token|id)", path, re.I):
            continue
        if isinstance(child, int) and child in REQUIRED_IDS:
            found.add(child)
        elif isinstance(child, list):
            found.update(item for item in child if isinstance(item, int) and item in REQUIRED_IDS)
    if not found:
        found.update(int(match.group(1)) for match in ID_RE.finditer(line))
    return found


def ids_from_text(line: str) -> set[int]:
    if not has_marker(MARKER_RE, line) or not has_marker(OBSERVED_RE, line):
        return set()
    return {int(match.group(1)) for match in ID_RE.finditer(line)}


def inspect_log(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    observed: dict[int, list[dict[str, Any]]] = {item: [] for item in REQUIRED_IDS}
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.decode("utf-8", errors="replace")
        ids = ids_from_json(line) or ids_from_text(line)
        for token_id in sorted(ids):
            if token_id in observed:
                observed[token_id].append({
                    "line": line_number,
                    "line_sha256": hashlib.sha256(raw_line).hexdigest(),
                })
    observed_ids = [token_id for token_id in REQUIRED_IDS if observed[token_id]]
    missing = [token_id for token_id in REQUIRED_IDS if not observed[token_id]]
    return {
        "schema": SCHEMA,
        "log_path": str(path),
        "log_bytes": len(raw),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "required_native_eog_ids": list(REQUIRED_IDS),
        "observed_native_eog_ids": observed_ids,
        "missing_native_eog_ids": missing,
        "status": "pass_required_eog_ids_observed" if not missing else "fail_missing_required_eog_ids",
        "evidence": {str(token_id): rows for token_id, rows in observed.items()},
        "interpretation": "This proves only that explicit EOG/EOS observation lines were retained in the supplied log; it does not prove canonical PRM-03 completion acceptance.",
    }


def self_test() -> dict[str, Any]:
    synthetic = (
        '{"event":"native_eog_ids_observed","observed":true,"eog_ids":[1,130073]}\n'
        'native EOG token id 1 observed by synthetic server\n'
        'native EOG token id 130073 observed by synthetic server\n'
    ).encode()
    path = Path("/tmp/run01-synthetic-server-eog.log")
    path.write_bytes(synthetic)
    result = inspect_log(path)
    if result["status"] != "pass_required_eog_ids_observed":
        raise AssertionError(result)
    result["synthetic_only"] = True
    result["synthetic_log_sha256"] = hashlib.sha256(synthetic).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--log", type=Path, help="retained root-owned server log")
    parser.add_argument("--out", type=Path, help="create-only JSON output")
    args = parser.parse_args()
    result = self_test() if args.self_test else inspect_log(args.log)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass_required_eog_ids_observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
