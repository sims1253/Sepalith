#!/usr/bin/env python3
"""Recompute the bounded independent review claims without model payload reads."""
import hashlib
import json
from pathlib import Path

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-full-weight-cpt-full-corpus-trainer-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = json.loads((PACKET / "source-manifest.json").read_text())
    artifacts = json.loads((PACKET / "artifact-manifest.json").read_text())
    source_errors = []
    for item in source["files"]:
        path = PACKET / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            source_errors.append(item["path"])
    artifact_errors = []
    for item in artifacts["files"]:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            artifact_errors.append(str(path))

    recipe_sha = sha(PACKET / "recipe.template.json")
    admission = json.loads((PACKET / "root-admission.template.json").read_text())
    binder = (PACKET / "source/experiments/training/bind_full_corpus_cpt.py").read_text()
    trainer = (PACKET / "source/experiments/training/full_weight_cpt_trainer.py").read_text()
    cache = (PACKET / "source/experiments/training/cpt_streaming_cache.py").read_text()
    sft = (PLAN / "docs/campaign/work/lead/r2-full-weight-edit-sft-eval-gate-v3/source/experiments/training/full_weight_edit_sft.py").read_text()
    model = json.loads(Path("/mnt/e/sepalith/campaign-20260915/models/SFT11-CPT-cloud-1902-merged-candidate-v1/config.json").read_text())
    result = {
        "status": "BLOCKED_TEMPLATE_REFERENCE",
        "source_manifest_errors": source_errors,
        "artifact_manifest_errors": artifact_errors,
        "source_identity": source["identity_sha256"],
        "recipe_sha256": recipe_sha,
        "admission_template_recipe_sha256": admission["template_sha256"],
        "admission_template_matches_recipe": admission["template_sha256"] == recipe_sha,
        "guards": {
            "cpt_binder_32k_allowlist": "(2048,4096,8192,16384,32768)" in binder,
            "cpt_runtime_32k_allowlist": "(2048, 4096, 8192, 16384, 32768)" in trainer,
            "sft_runtime_32k_allowlist": "(2048, 4096, 8192, 16384, 32768)" in sft,
            "cache_tail_literal_32k": "[-32768:]" in cache,
            "model_max_position_embeddings": model.get("max_position_embeddings"),
        },
        "evaluator_sha256": sha(PACKET / "source/experiments/training/campaign_cpt_eval.py"),
    }
    assert not source_errors and not artifact_errors
    assert result["recipe_sha256"] == "09349f90ebe19bead8390c91f0b6c513a739730c20f6185fda3916db28f3917f"
    assert result["admission_template_matches_recipe"] is False
    assert all(result["guards"][key] for key in ("cpt_binder_32k_allowlist", "cpt_runtime_32k_allowlist", "sft_runtime_32k_allowlist", "cache_tail_literal_32k"))
    assert result["guards"]["model_max_position_embeddings"] == 131072
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
