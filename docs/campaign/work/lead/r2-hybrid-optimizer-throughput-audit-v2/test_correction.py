#!/usr/bin/env python3
"""Small model-free assertions for the live header-only correction."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
audit = json.loads((HERE / "live-dtype-analysis.json").read_text(encoding="utf-8"))
live = audit["live_inventory"]
counts = audit["corrected_step_census"]
headers = audit["checkpoint_header_pins"]["model_headers"]

assert audit["read_policy"]["safetensors_payload_read"] is False
assert audit["read_policy"]["checkpoint_loaded"] is False
assert live["parameter_tensors"] == 381
assert live["parameters"] == 2_516_756_480
assert live["dtype_rows"] == {"BF16": 296, "F32": 85}
assert live["dtype_elements"] == {"BF16": 2_516_582_400, "F32": 174_080}
assert counts["bf16_stochastic_round_chunks"] == 2_442
assert counts["bf16_rand_like_invocations"] == 2_442
assert counts["bf16_apply_fp32_update_invocations"] == 804
assert counts["bf16_pos_neg_inf_scalar_tensor_allocations"] == 1_608
assert counts["f32_direct_apply_fp32_update_invocations"] == 85
assert counts["f32_rand_like_invocations"] == 0
assert counts["f32_pos_neg_inf_scalar_tensor_allocations"] == 0
assert headers[0]["header_sha256"] == headers[1]["header_sha256"]
assert all(item["file_bytes"] == 5_033_904_496 for item in headers)
print("live dtype correction: PASS (381 tensors, 296 BF16, 85 F32; header-only)")
