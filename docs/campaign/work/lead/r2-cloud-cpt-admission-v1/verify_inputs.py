#!/usr/bin/env python3
"""CPU-only verification of the immutable inputs proposed for cloud CPT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
RECIPE = PLAN / "docs/campaign/work/lead/r2-cpt-remaining-v1/recipe.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    recipe = json.loads(RECIPE.read_text())
    selected = {item["path"]: item["sha256"] for item in recipe["inputs"]}
    checks = []
    for raw_path, expected in sorted(selected.items()):
        path = Path(raw_path)
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": raw_path,
                "bytes": path.stat().st_size if path.is_file() else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "ok": actual == expected,
            }
        )
    manifest = Path("/mnt/e/sepalith/campaign-20260915/data-work/CPT-remaining-v1/manifest.json")
    materialized = json.loads(manifest.read_text())
    artifact_checks = {
        "train_manifest_matches_recipe": (
            materialized["artifacts"]["cpt_train.jsonl"]["sha256"]
            == recipe["train_rows"]["sha256"]
        ),
        "schedule_manifest_matches_recipe": (
            materialized["artifacts"]["draws-1902.json"]["sha256"]
            == recipe["draw_schedule"]["sha256"]
        ),
        "schedule_manifest_bytes_match_file": (
            materialized["artifacts"]["draws-1902.json"]["bytes"]
            == Path(recipe["draw_schedule"]["path"]).stat().st_size
        ),
        "coverage_matches_recipe": (
            materialized["schedule"]["unique_rows"]
            == recipe["coverage_contract"]["scheduled_unique_rows"]
            and materialized["schedule"]["replay_rows"]
            == recipe["coverage_contract"]["scheduled_replay_rows"]
            and materialized["schedule"]["draws"]
            == recipe["coverage_contract"]["terminal_draw_cursor"]
        ),
    }
    passed = all(row["ok"] for row in checks) and all(artifact_checks.values())
    output = {
        "schema": "sepalith.cloud-cpt.input-verification.v1",
        "recipe": str(RECIPE),
        "recipe_sha256": sha256(RECIPE),
        "status": "pass" if passed else "fail",
        "checks": checks,
        "artifact_cross_checks": artifact_checks,
        "transport_bytes": sum(row["bytes"] or 0 for row in checks),
        "coverage": materialized["selection"],
        "schedule": materialized["schedule"],
    }
    (HERE / "input-verification.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
