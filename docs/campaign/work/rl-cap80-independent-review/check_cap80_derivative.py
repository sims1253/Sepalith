"""Independent CPU-only verification of the explicit cap80 checkpoint derivative."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json

SOURCE = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c/archive/full/checkpoint-100")
DERIVED = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/RL-primary-full100-cap80/full/checkpoint-100")
OLD_IDENTITY = "48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2"
NEW_IDENTITY = "ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818"
SOURCE_MANIFEST = "2fa678487c5a760a2d04b798fc29787cd300752e7b6bd3445fdb8a9b09399ebd"
SOURCE_STATE = "930385c7715c91d3be3e9a570ee8de752d12f04ef1e6db55b10aab05900b9ed8"
DERIVED_MANIFEST = "2fbd0fad2b2fc38dab9bf306fdfb5dac0be88bfd21889462954328e39bed4a8b"
DERIVED_STATE = "68ac4013e530cd210470b4184b031c6b4b26ac8926f0ec001337e6888ddf156e"
PROVENANCE = "db25d3322bac3a98920c408087f35c112e9d964b46a41aedff2fe6a8841ba609"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_json(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def unexpected_identity_diffs(source: object, derived: object, path: str = "") -> list[str]:
    """Return all identity changes except the one permitted resource field."""
    if path == "policy.cuda_memory_fraction":
        if source == 0.75 and derived == 0.80:
            return []
        return [f"{path}: {source!r}->{derived!r}"]
    if isinstance(source, Mapping) and isinstance(derived, Mapping):
        if set(source) != set(derived):
            return [f"{path}.keys differ"]
        out: list[str] = []
        for key in sorted(source):
            out.extend(unexpected_identity_diffs(
                source[key], derived[key], f"{path}.{key}".strip("."),
            ))
        return out
    if isinstance(source, list) and isinstance(derived, list):
        if len(source) != len(derived):
            return [f"{path}.length differs"]
        out: list[str] = []
        for index, (before, after) in enumerate(zip(source, derived)):
            out.extend(unexpected_identity_diffs(before, after, f"{path}[{index}]"))
        return out
    return [] if source == derived else [f"{path}: values differ"]


def state_nonidentity_digest(path: Path) -> tuple[str, str, str, dict[str, object]]:
    state = read_json(path / "campaign-state.json")
    assert isinstance(state, Mapping)
    sampler = state["sampler"]
    assert isinstance(sampler, Mapping)
    cursor = {
        key: sampler[key]
        for key in ("source_draw_cursor", "selected_id_index", "consumed_rows", "consumed_prompt_copies")
    }
    return (
        sha_json({key: value for key, value in state.items() if key != "identity"}),
        sha_json(sampler),
        sha_json(state["identity"]),
        cursor,
    )


def main() -> None:
    source_manifest = read_json(SOURCE / "campaign-manifest.json")
    derived_manifest = read_json(DERIVED / "campaign-manifest.json")
    assert isinstance(source_manifest, Mapping) and isinstance(derived_manifest, Mapping)
    source_identity = source_manifest["identity"]
    derived_identity = derived_manifest["identity"]
    assert sha_json(source_identity) == OLD_IDENTITY
    assert sha_json(derived_identity) == NEW_IDENTITY
    assert unexpected_identity_diffs(source_identity, derived_identity) == []
    assert source_identity["policy"]["cuda_memory_fraction"] == 0.75
    assert derived_identity["policy"]["cuda_memory_fraction"] == 0.80
    assert source_manifest["full"] is True and derived_manifest["full"] is True
    assert source_manifest["step"] == derived_manifest["step"] == 100

    source_files = source_manifest["files"]
    derived_files = derived_manifest["files"]
    assert set(derived_files) - set(source_files) == {"resource-cap-migration.json"}
    assert set(source_files) - set(derived_files) == set()
    assert all(source_files[name] == derived_files[name]
               for name in source_files if name != "campaign-state.json")
    assert derived_files["resource-cap-migration.json"] == {
        "bytes": 2766, "sha256": PROVENANCE,
    }
    provenance = read_json(DERIVED / "resource-cap-migration.json")
    assert isinstance(provenance, Mapping)
    assert provenance["source_identity_sha256"] == OLD_IDENTITY
    assert provenance["derived_identity_sha256"] == NEW_IDENTITY
    assert provenance["identity_change"] == {
        "path": "policy.cuda_memory_fraction", "before": 0.75, "after": 0.8,
    }

    source_state = state_nonidentity_digest(SOURCE)
    derived_state = state_nonidentity_digest(DERIVED)
    assert source_state[0] == derived_state[0]
    assert source_state[1] == derived_state[1]
    assert source_state[2] == OLD_IDENTITY and derived_state[2] == NEW_IDENTITY
    assert source_state[3] == derived_state[3] == {
        "source_draw_cursor": 800, "selected_id_index": 425,
        "consumed_rows": 25600, "consumed_prompt_copies": 6400,
    }

    assert sha_file(SOURCE / "campaign-manifest.json") == SOURCE_MANIFEST
    assert sha_file(SOURCE / "campaign-state.json") == SOURCE_STATE
    assert sha_file(DERIVED / "campaign-manifest.json") == DERIVED_MANIFEST
    assert sha_file(DERIVED / "campaign-state.json") == DERIVED_STATE

    # The frozen production verifier inventories each checkpoint and compares
    # the full identity. This imports no framework and performs no writes.
    import sys
    sys.path.insert(0, "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190b9/source/experiments/training")
    from campaign_checkpoint import verify_checkpoint
    assert verify_checkpoint(SOURCE, source_identity, require_full=True)["step"] == 100
    assert verify_checkpoint(DERIVED, derived_identity, require_full=True)["step"] == 100

    print("identity delta: policy.cuda_memory_fraction 0.75 -> 0.80 only")
    print("inventory: source preserved; derivative adds only resource-cap-migration.json")
    print("state residual and sampler: equal; cursor=800 index=425 rows=25600 copies=6400")
    print("strict production verify: source and derivative full checkpoint step 100 PASS")
    print("INDEPENDENT_CAP80_DERIVATIVE_PASS")


if __name__ == "__main__":
    main()
