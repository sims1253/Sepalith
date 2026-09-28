#!/usr/bin/env python3
"""Small read-only assertions for the frozen audit result."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / "audit-result.json").read_text())
capacity = json.loads((ROOT / "capacity.json").read_text())

assert audit["audit"]["status"] == "pass"
assert audit["audit"]["file_count"] == 13
assert all(entry["match"] for entry in audit["audit"]["pair"].values())
assert audit["audit"]["manifest_sha256"] == "f7bd7b819584a8abeadc37c677914bd8775f350bd097479d326425292ba8e943"
assert capacity["projection_with_declared_future"]["over_cap_bytes"] > 0
assert capacity["projection_after_conditional_ordinary_c_release"]["margin_bytes"] > 0
assert capacity["free_space_observation"]["floor_pass"] is True
assert audit["named_paths"]["future_450_e"]["exists"] is False
print("read-only audit assertions: PASS")
