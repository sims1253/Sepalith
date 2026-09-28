#!/usr/bin/env python3
"""Create a root-reviewable signal-driver config from already accepted pins."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from signal_pilot_harness import canonical, sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--expected-merged-weights-sha256", required=True)
    ap.add_argument("--expected-tokenizer-json-sha256", required=True)
    ap.add_argument("--pilot-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if not a.model_path.is_dir() or not a.model_manifest.is_file():
        raise SystemExit("accepted model path/manifest is absent")
    if sha256(a.model_manifest) != a.expected_manifest_sha256:
        raise SystemExit("accepted model manifest hash differs")
    names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
    artifacts = []
    for path in sorted(a.model_path.rglob("*")):
        rel = path.relative_to(a.model_path)
        if path.is_file() and not any(part.startswith(".") for part in rel.parts):
            artifacts.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not names.issubset({row["path"] for row in artifacts}):
        raise SystemExit("required model config/tokenizer files are absent")
    weights = [row for row in artifacts if row["path"].endswith((".safetensors", ".bin"))]
    aggregate = hashlib.sha256(canonical(weights)).hexdigest()
    if aggregate != a.expected_merged_weights_sha256:
        raise SystemExit("accepted merged weight aggregate differs")
    tokenizer_hash = next(row["sha256"] for row in artifacts if row["path"] == "tokenizer.json")
    if tokenizer_hash != a.expected_tokenizer_json_sha256:
        raise SystemExit("accepted tokenizer hash differs")
    config = json.loads(a.template.read_text(encoding="utf-8"))
    if config.get("model") is not None or config.get("output_path") is not None or config.get("binding_config") is not None or config.get("root_admission") is not None:
        raise SystemExit("binding template is not blank")
    config["model"] = {
        "manifest_path": str(a.model_manifest.resolve()),
        "manifest_sha256": a.expected_manifest_sha256,
        "model_path": str(a.model_path.resolve()), "artifacts": artifacts,
        "merged_weights_sha256": aggregate,
        "tokenizer_json_sha256": tokenizer_hash,
    }
    config["output_path"] = str(a.pilot_output.resolve())
    config["status"] = "root_binding_prepared_admission_required"
    if a.output.exists():
        raise SystemExit("bound config output already exists")
    a.output.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "config": str(a.output), "sha256": sha256(a.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
