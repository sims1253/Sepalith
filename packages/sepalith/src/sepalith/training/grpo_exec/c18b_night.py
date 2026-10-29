#!/usr/bin/env python3
"""C18b entry for ``grpo_exec.night``: apply ``c18b.install()`` (C18b recipe module and entry modules), then run the C18 code."""
from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from sepalith.training.grpo_exec import night, c18b

    c18b.install()
    return night.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
