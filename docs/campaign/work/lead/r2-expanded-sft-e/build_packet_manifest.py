#!/usr/bin/env python3
"""Hash the small E preparation closure; large model/checkpoint files stay referenced."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = (
    "README.md",
    "build_packet_manifest.py",
    "commands.json",
    "cpu-metadata-tests.log",
    "preflight.stdout",
    "preflight.time",
    "prepare_packet.py",
    "recipe.json",
    "run_owned.py",
    "runner-recipe.json",
    "tests/test_resume500_e_packet.py",
    "tests/test_runner_snapshot_replay.py",
)


def sha(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


rows = []
for name in NAMES:
    path = HERE / name
    rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
manifest = {
    "schema": "sepalith.sft11.expanded-e-packet-manifest.v1",
    "status": "prepared_not_admitted_not_launched",
    "source_identity": "bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384",
    "d_full500_manifest_sha256": "6a4fe603566af95a09c3fd3bdadea37dac2717db55b08063a0bc9a1d93a1223b",
    "files": rows,
}
(HERE / "packet-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"files": len(rows), "packet_manifest": sha(HERE / "packet-manifest.json")}, sort_keys=True))
