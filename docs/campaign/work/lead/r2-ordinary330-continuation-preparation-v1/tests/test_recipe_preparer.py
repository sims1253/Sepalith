#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("prepare_runtime_recipe", PACKET / "prepare_runtime_recipe.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)

trainer_spec = importlib.util.spec_from_file_location(
    "ordinary330_trainer", PACKET / "source/experiments/training/full_weight_cpt_trainer.py"
)
trainer = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer)


with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    migration = json.loads((PACKET / "runtime-source-migration-admission.template.json").read_text())
    migration["status"] = "admitted"
    migration["launch_authorized"] = True
    migration_path = temporary / "migration.json"
    migration_path.write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n")
    output = temporary / "runtime-recipe.json"
    result = prepare_module.prepare(
        PACKET / "source-manifest.json",
        migration_path,
        "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/test-ordinary330-preparer/runtime",
        "/mnt/e/sepalith/campaign-20260915/checkpoints/test-ordinary330-preparer",
        output,
    )
    original = json.loads(prepare_module.ORIGINAL.read_text())
    prepared = json.loads(output.read_text())
    assert trainer.identity(original) == trainer.identity(prepared)
    assert result["production_identity_sha256"] == "730c2aba916ae9ab8a97a1584207e7c1d7095e23ce1988cdd968df7fd17844b6"
    assert result["source_manifest_sha256"] == sha(PACKET / "source-manifest.json")
    assert prepared["runtime_source"]["manifest_sha256"] == sha(PACKET / "source-manifest.json")
    assert prepared["runtime_source_migration"]["admission_sha256"] == sha(migration_path)

    bad = dict(migration)
    bad["scientific_source_manifest_sha256"] = "0" * 64
    bad_path = temporary / "bad-migration.json"
    bad_path.write_text(json.dumps(bad, indent=2, sort_keys=True) + "\n")
    try:
        prepare_module.prepare(
            PACKET / "source-manifest.json",
            bad_path,
            "/home/m0hawk/.local/state/sepalith/campaign-20260915/training/test-bad/runtime",
            "/mnt/e/sepalith/campaign-20260915/checkpoints/test-bad",
            temporary / "bad-recipe.json",
        )
    except ValueError as error:
        assert "binding differs" in str(error)
    else:
        raise AssertionError("a mismatched scientific source binding was accepted")

print(json.dumps({"status": "PASS", "tests": 2}, sort_keys=True))
