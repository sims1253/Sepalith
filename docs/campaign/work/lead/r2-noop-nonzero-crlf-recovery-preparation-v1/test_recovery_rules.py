#!/usr/bin/env python3
"""Small pure controls for the v4 no-op geometry relaxations."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_recovery_audit as audit  # noqa: E402


def main() -> None:
    # Existing nonzero selection: an unchanged closing brace is valid.
    raw, changed = audit.normalized_source(b"x() {\n}\n", "exact_bytes")
    assert raw == b"x() {\n}\n" and changed is False
    assert audit.range_text("prefix\n}\n", {"line": 1, "character": 0}, {"line": 1, "character": 1}) == "}"

    # Uniform CRLF is normalized for checks, while the raw bytes remain intact.
    normalized, changed = audit.normalized_source(b"x() {\r\n}\r\n", "uniform_crlf_to_lf")
    assert normalized == b"x() {\n}\n" and changed is True

    # Mixed EOL is intentionally fail-closed.
    try:
        audit.normalized_source(b"x() {\r\n}\n", "uniform_crlf_to_lf")
    except audit.RecoveryError as error:
        assert str(error) == "mixed_eol_not_supported"
    else:
        raise AssertionError("mixed EOL was accepted")

    print("v4 geometry controls: PASS")


if __name__ == "__main__":
    main()
