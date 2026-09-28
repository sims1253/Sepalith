#!/usr/bin/env python3
"""Static serving contract check; it never opens a model or weight payload."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


DSPARK_CHECKPOINT = (
    "/home/m0hawk/.local/state/sepalith/dp-b/checkpoints/sepalith-r2/"
    "dspark_minicpm5_2b_step500_profile/step_8"
)
TARGET_Q8 = (
    "/home/m0hawk/.local/state/sepalith/campaign-20260915/models/"
    "SFT11-task-global-b-500-quant/model-Q8_0.gguf"
)
TARGET_Q8_SHA256 = "d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db"

SOURCE_PINS = {
    "convert_hf_to_gguf.py": (
        "e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f",
        ("--target-model-dir", "--dspark is only supported for DeepseekV4ForCausalLM"),
    ),
    "conversion/__init__.py": (
        "7f5d0adaa7f932420418351994f65c14e7b05de951edf388b3d78ee4e7004997",
        ("Qwen3DSparkModel",),
    ),
    "conversion/qwen.py": (
        "65c61155458078232dd3f9d23284710fa39c29cf7ecc341194c825cf5334f43f",
        ("class DSparkModel", "MODEL_ARCH.DFLASH", "target_layer_ids", "i + 1"),
    ),
    "gguf-py/gguf/tensor_mapping.py": (
        "2f40f1596e77478d4884c5a94a6799737d36a6eee34a09af6008692635195b7c",
        ("model.markov_head.markov_w1", "model.markov_head.markov_w2"),
    ),
    "common/speculative.cpp": (
        "81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3",
        ("draft-dspark", "dflash.block_size", "n_draft_max = is_dspark ? block_size", "markov_w1.weight"),
    ),
    "common/arg.cpp": (
        "566a8122afc02bbfcca572b7ac8d7d7d985b2b087b780434be75f588184f458d",
        ("--spec-draft-n-max", "--spec-ngram-mod-n-match"),
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = args.plan.resolve()
    packet = plan / "docs/campaign/work/lead/r2-trained-draft-serving-preparation-v1"
    execution = plan / "docs/campaign/work/lead/r2-draft-local-execution-v1"
    source = plan / "docs/campaign/work/lead/r2-draft-local-execution-v1/source"
    results: dict[str, dict[str, Any]] = {}

    def check(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        try:
            detail = fn()
            results[name] = {"status": "PASS", **detail}
        except Exception as exc:  # static gate should return a useful receipt
            results[name] = {"status": "FAIL", "error": str(exc)}

    def target_manifest() -> dict[str, Any]:
        value = load(execution / "target-manifest.json")
        assert value["status"].startswith("accepted_step500")
        assert value["accepted_parent_wrapper"]["status"] == "accepted"
        assert value["accepted_parent_wrapper"]["sha256"] == "05eac983926eacc45b78f59d281eeaeeecfc8c58ed03647b4a8cbe794160afa5"
        assert value["merged_weights_sha256"] == "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c"
        assert value["profile_binding"] == {
            "block_size": 7,
            "max_context_tokens": 4096,
            "num_anchors": 32,
            "num_draft_layers": 5,
            "precision": "bf16",
            "resume": False,
            "rows": 8,
            "steps": 8,
            "target_layer_ids": [1, 10, 20, 30, 39],
        }
        assert value["tokenizer_contract"] == {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "native_eog_ids": [1, 130073],
            "pad_token_id": 1,
            "vocab_size": 130560,
        }
        return {"wrapper_sha256": value["accepted_parent_wrapper"]["sha256"], "profile": value["profile_binding"]}

    def checkpoint_binding() -> dict[str, Any]:
        spec = load(packet / "benchmark-spec.json")
        draft = spec["trained_draft"]
        assert draft["checkpoint"] == DSPARK_CHECKPOINT
        assert draft["root_checkpoint_readback"]["model_safetensors_sha256"] == "6eed8dfafd7a2b419950d9bafe3865fbe93885061bbebae1da70041f321875b8"
        assert draft["root_checkpoint_readback"]["config_sha256"] == "dd328e3ea202509d032d2b9de020c84724082ee41e41c5cd41516fb5df5b2435"
        assert draft["root_checkpoint_readback"]["optimizer_state_count"] == 8
        assert draft["root_checkpoint_readback"]["changed_public_tensor_count"] == 58
        assert draft["root_checkpoint_readback"]["target_matrix_count"] == 2
        assert spec["target"]["gguf_q8_path"] == TARGET_Q8
        assert spec["target"]["gguf_q8_sha256"] == TARGET_Q8_SHA256
        return {"checkpoint": DSPARK_CHECKPOINT, "target_q8_sha256": TARGET_Q8_SHA256}

    def profile_source() -> dict[str, Any]:
        path = source / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/profile_config.py"
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        for marker in ("max_train_steps=8", "checkpointing_steps=8", "block_size=7", "target_layer_ids=[1, 10, 20, 30, 39]", "precision=\"bf16\"", "allow_resume=False"):
            assert marker in text, marker
        trainer = source / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/warmstart_trainer.py"
        trainer_text = trainer.read_text(encoding="utf-8")
        ast.parse(trainer_text, filename=str(trainer))
        for marker in ("sepalith-warmstart.json", "FROZEN_TARGET_KEYS", "target matrices"):
            assert marker in trainer_text, marker
        ckpt = source / "docs/campaign/work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec/deepspec/trainer/ckpt_manager.py"
        ckpt_text = ckpt.read_text(encoding="utf-8")
        assert "f\"step_{global_step}\"" in ckpt_text
        assert "step_latest" in ckpt_text
        assert "save_pretrained" in ckpt_text
        return {"profile_config": str(path), "checkpoint_writer": str(ckpt)}

    def converter_sources() -> dict[str, Any]:
        root = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453")
        hashes: dict[str, str] = {}
        for relative, (expected, markers) in SOURCE_PINS.items():
            path = root / relative
            assert path.is_file(), path
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                ast.parse(text, filename=str(path))
            actual = digest(path)
            hashes[relative] = actual
            assert actual == expected, f"{relative}: {actual} != {expected}"
            for marker in markers:
                assert marker in text, f"{relative}: missing marker {marker!r}"
        conversion = (packet / "conversion-command.txt").read_text(encoding="utf-8")
        executable_lines = "\n".join(line for line in conversion.splitlines() if not line.lstrip().startswith("#"))
        assert "--target-model-dir \"$TARGET_HF\"" in executable_lines
        assert "--dspark" not in executable_lines
        assert "--outtype f16" in executable_lines and "Q8_0 2" in executable_lines
        return {"source_hashes": hashes, "command_has_no_dspark_switch": True}

    def fixture() -> dict[str, Any]:
        manifest = load(plan / "docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json")
        assert manifest["fixture_row_count"] == 4
        assert manifest["constraints"]["accepted_split"] == "train"
        rows = manifest["selected_rows"]
        assert len(rows) == 4
        assert {row["split"] for row in rows} == {"train"}
        ids = [row["row_id"] for row in rows]
        expected = ["0003ea6fc6a4d0b3b354efba", "000a6aaa48aee4291dbcb0cb", "000d2bf789b999f887d12308", "00ef45e53aea030d3465f860"]
        assert ids == expected
        probe = (plan / "docs/campaign/work/serving-readiness/runtime_native_probe.py").read_text(encoding="utf-8")
        assert 'choices=("baseline", "ngram-mod")' in probe
        assert "n_predict" in probe and "returned_token_ids" in probe
        return {"rows": ids, "probe_has_only_existing_arms": True}

    def import_gate() -> dict[str, Any]:
        value = load(execution / "root-import-check.json")
        assert value["status"] == "PASS"
        assert value["actual_clean_environment_used"] is True
        assert value["cuda_visible_devices"] == ""
        assert value["profile_limit"] == 1800
        assert value["stage_limits"] == {
            "teacher-cache-profile": 420,
            "native-cuda-smoke": 240,
            "warmstart-trainer-profile": 720,
        }
        return {"actual_clean_environment_used": True, "cuda_visible_devices": ""}

    check("target_manifest", target_manifest)
    check("checkpoint_binding", checkpoint_binding)
    check("profile_and_checkpoint_writer", profile_source)
    check("converter_and_runtime_sources", converter_sources)
    check("train_fixture_and_probe", fixture)
    check("execution_import_gate", import_gate)
    status = "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL"
    receipt = {
        "schema_version": "sepalith.r2.trained-dspark.serving-contract-check.v1",
        "status": status,
        "no_model_payload_read": True,
        "no_weight_hash_by_checker": True,
        "plan": str(plan),
        "packet": str(packet),
        "checks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out": str(args.out)}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
