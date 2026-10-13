#!/usr/bin/env python3
"""Bind root's later scientific/resource decision into the prepared trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "bound recipe output must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def bind(template_path, admission_path, output):
    template_path, admission_path = Path(template_path), Path(admission_path)
    template = json.loads(template_path.read_text())
    admission = json.loads(admission_path.read_text())
    require(template.get("schema") == "sepalith.sft11.full-weight-cpt-lr-pilot-template.v4", "template schema differs")
    require(admission.get("schema") == "sepalith.sft11.full-weight-cpt-lr-pilot-root-admission.v4", "admission schema differs")
    require(admission.get("status") == "admitted" and admission.get("launch_authorized") is True, "root has not admitted this configuration")
    require(admission.get("template_sha256") == sha(template_path), "admission refers to another template")
    selected = admission.get("selected", {})
    require(set(selected) == {"parent_candidate_id", "optimizer", "micro_batch", "gradient_accumulation", "learning_rate", "scheduler", "warmup_ratio", "checkpoint_every", "evaluation_steps", "selected_milestones", "telemetry_every"}, "admission selection fields differ")
    require(selected["parent_candidate_id"] in template["parent_candidates"], "parent candidate is not prepared")
    micro = selected["micro_batch"]
    require(micro in (1, 2), "micro batch must be 1 or 2")
    require(type(selected["gradient_accumulation"]) is int and micro * selected["gradient_accumulation"] == 16, "effective batch must be 16")
    dispatch = json.loads(Path(template["optimizer_dispatch"]["path"]).read_text())
    require(selected["optimizer"]["arm"] == dispatch.get("arm"), "optimizer arm lacks this template's actual runtime dispatch")
    require(type(selected["learning_rate"]) is float and 0 < selected["learning_rate"] <= 1e-3, "learning rate is outside preparation bound")
    require(selected["optimizer"].get("hidden_lr") == selected["learning_rate"], "scheduler/base hidden learning rate differs from optimizer")
    require(selected["scheduler"] in ("cosine", "constant_with_warmup"), "scheduler differs")
    require(type(selected["warmup_ratio"]) is float and 0 <= selected["warmup_ratio"] < 1, "warmup ratio differs")
    cadence = selected["checkpoint_every"]
    max_steps = template["cohort"]["updates"]
    require(type(cadence) is int and cadence > 0 and max_steps % cadence == 0, "checkpoint cadence must divide the cohort horizon")
    evaluations = selected["evaluation_steps"]
    milestones = selected["selected_milestones"]
    require(isinstance(evaluations, list) and all(type(x) is int and 1 <= x <= max_steps for x in evaluations), "evaluation steps invalid")
    require(all(step % cadence == 0 for step in evaluations), "evaluation step lacks a scheduled full checkpoint")
    require(isinstance(milestones, list) and set(evaluations) <= set(milestones) and max_steps in milestones, "selected milestones must preserve evaluations and terminal")
    require(type(selected["telemetry_every"]) is int and selected["telemetry_every"] > 0, "telemetry cadence invalid")
    require(max_steps == 24 and micro == 1 and selected["gradient_accumulation"] == 16, "LR pilot horizon/batching differs")
    require(selected["scheduler"] == "constant_with_warmup" and int(max_steps * selected["warmup_ratio"]) == 2, "LR pilot warmup must be exactly two updates")
    require(cadence == 24 and evaluations == [24] and milestones == [24], "LR pilot permits only terminal checkpoint/evaluation")
    pair = (selected["optimizer"].get("hidden_lr"), selected["optimizer"].get("side_lr"))
    require(selected["optimizer"].get("arm") == "aurora_mix" and pair in ((3e-5, 3e-6), (1e-5, 1e-6)) and selected["learning_rate"] == pair[0], "LR pilot pair differs")
    parent = template["parent_candidates"][selected["parent_candidate_id"]]
    bound = dict(template)
    bound["schema"] = "sepalith.sft11.full-weight-cpt-lr-pilot-bound.v4"
    bound["status"] = "root_admitted_not_launched"
    bound["root_admission"] = {"path": str(admission_path.resolve()), "sha256": sha(admission_path)}
    bound["parent"] = parent
    bound["runtime"] = {**selected, "effective_batch": 16, "max_steps": max_steps}
    del bound["parent_candidates"]
    write_new(output, bound)
    return bound


def main():
    p = argparse.ArgumentParser(); p.add_argument("--template", type=Path, required=True); p.add_argument("--admission", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); value = bind(a.template, a.admission, a.output)
    print(json.dumps({"status": value["status"], "output": str(a.output), "sha256": sha(a.output)}, sort_keys=True))


if __name__ == "__main__": main()
