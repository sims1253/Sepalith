"""Convert captured edit records and render prompts without training dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .protocol import EditContext, render_context, serialize_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("normalize", "render"))
    parser.add_argument("input", type=Path, help="One JSON edit-context record")
    parser.add_argument("--legacy", action="store_true", help="Explicitly migrate an unversioned eval record")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        context = EditContext.from_legacy(record) if args.legacy else EditContext.from_dict(record)
        output = serialize_snapshot(context) + "\n" if args.operation == "normalize" else render_context(context)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"sepalith-context: {exc}\n")
    # Rendering emits exact prompt bytes, without an extra terminal newline.
    sys.stdout.buffer.write(output.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
