#!/usr/bin/env python3
"""Stream one approved input until its named row, then write only that row."""

import json
import re
import sys
from pathlib import Path


if len(sys.argv) != 4:
    raise SystemExit("usage: extract_target_row.py INPUT_JSONL TARGET_ROW_ID OUTPUT_JSONL")

input_path = Path(sys.argv[1])
target = sys.argv[2]
output_path = Path(sys.argv[3])
row_pattern = re.compile(r'"row_id"\s*:\s*"([^"]+)"')

for row_number, line in enumerate(input_path.open(encoding="utf-8"), start=1):
    match = row_pattern.search(line)
    if not match or match.group(1) != target:
        continue
    row = json.loads(line)
    if row.get("row_id") != target:
        continue
    output_path.write_text(line, encoding="utf-8")
    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "row_id": target,
        "source_line": row_number,
        "output_bytes": len(line.encode("utf-8")),
    }, separators=(",", ":")))
    break
else:
    raise SystemExit(f"target row not found before EOF: {target}")
