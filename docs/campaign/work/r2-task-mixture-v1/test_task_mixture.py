#!/usr/bin/env python3
"""Bounded CPU checks for the DAT-10 candidate registry and schedule."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign")
HERE = PLAN / "work/r2-task-mixture-v1"
ROWS = HERE / "candidate-token-rows.jsonl"
SCHEDULE = HERE / "proposed-draw-manifest.json"
PREP = HERE / "prepare_task_mixture.py"
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

# Importing the preparation module is CPU-only; its main function is guarded.
sys.path.insert(0, str(EXEC / "packages/sepalith/src"))
from sepalith.campaign_protocol import validate_training_row  # noqa: E402

# Exercise split fail-closed decisions without opening any source files.
spec = importlib.util.spec_from_file_location("r2_mixture_prep_test", PREP)
assert spec and spec.loader
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)
_fake_groups = {
    "train": {"split": "train_group"},
    "val": {"split": "train_group"},
    "other": {"split": "dev_group"},
}
_fake_partition = {"train": "cpt_train", "val": "cpt_validation", "other": "cpt_train"}
assert prep.binding_status("train", _fake_groups, _fake_partition) == "bound_cpt_train"
assert prep.binding_status("val", _fake_groups, _fake_partition) == "cpt_validation_reserved"
assert prep.binding_status("other", _fake_groups, _fake_partition) == "dat02_split_dev_group"
assert prep.binding_status("absent", _fake_groups, _fake_partition) == "unknown_dat02_group"
assert prep.binding_status("missing", {**_fake_groups, "missing": {"split": "train_group"}}, _fake_partition) == "group_not_in_cpt_partition"
assert prep.binding_status(None, _fake_groups, _fake_partition) == "missing_group_id"

rows = 0
ids = set()
prompts = set()
families = Counter()
operations = Counter()
noops = 0
target_labels = 0
with ROWS.open("rb") as f:
    for line_no, raw in enumerate(f, 1):
        row = json.loads(raw)
        validate_training_row(row)
        ident = row["id"]
        assert ident not in ids, (line_no, ident)
        ids.add(ident)
        prompt = hashlib.sha256(row["prompt_text"].encode()).hexdigest()
        assert prompt not in prompts, (line_no, ident)
        prompts.add(prompt)
        assert row["split"] == "train" and row["renderer_id"] == "zeta2-prm03-v1"
        assert len(row["input_ids"]) <= 4096
        assert row["target_token_count"] + 1 <= 192
        assert row["target_start"] == row["prompt_token_count"] + 1
        families[row["family"]] += 1
        operations[row["target_operation"]] += 1
        if row["target_operation"] == "no_op":
            noops += 1
        target_labels += row["target_token_count"] + 1
        rows += 1
assert rows == 11515
assert noops == 1643
assert operations == Counter({"replace": 9872, "no_op": 1643})

schedule = json.loads(SCHEDULE.read_text())
assert schedule["status"] == "complete"
assert schedule["max_steps"] == 1000 and schedule["effective_batch"] == 16
assert len(schedule["row_ids"]) == 16000
assert schedule["token_rows_sha256"] == digest(ROWS)
# Re-run the reviewed metadata-only validator, which checks manifest/schedule
# hashes, finite presentations, family/no-op caps, and row metadata identity.
sampling_spec = importlib.util.spec_from_file_location("r2_mixture_sampling_test", EXEC / "experiments/training/campaign_sampling.py")
assert sampling_spec and sampling_spec.loader
sampling = importlib.util.module_from_spec(sampling_spec)
sys.modules[sampling_spec.name] = sampling
sampling_spec.loader.exec_module(sampling)
sampling.validate_draw_manifest(schedule)
print(json.dumps({"status": "PASS", "rows": rows, "families": dict(sorted(families.items())),
                  "operations": dict(sorted(operations.items())), "noops": noops,
                  "target_label_tokens": target_labels, "draws": len(schedule["row_ids"]),
                  "schedule_status": schedule["status"]}, sort_keys=True))
