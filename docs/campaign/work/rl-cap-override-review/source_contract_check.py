"""Static red-capable check of the frozen RL entry cap seam."""
from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path("/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source/experiments/training/campaign_rl_entry.py")
text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
assert "CUDA_MEMORY_FRACTION_LIMIT = 0.80" in text
assert "recipe.cuda_memory_fraction must equal identity.policy.cuda_memory_fraction" in text
run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
run_text = ast.get_source_segment(text, run)
assert 'admission["runtime"]["cuda_memory_fraction"]' in run_text
assert "_load_live_model" in run_text
load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_live_model")
load_text = ast.get_source_segment(text, load)
assert "_configure_cuda_allocator(torch, validated_fraction)" in load_text
print("frozen entry contract: cap equality, run propagation, and allocator setup present")
