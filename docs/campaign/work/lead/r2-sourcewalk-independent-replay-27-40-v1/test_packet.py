#!/usr/bin/env python3
"""Small metadata-only checks for the 27–40 replay continuation packet."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PIN = json.loads((PACKET / "input-pin.json").read_text())

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

driver = Path(PIN["driver"]["path"])
index = Path(PIN["replay_index"]["path"])
assert driver.is_file() and sha(driver) == PIN["driver"]["sha256"]
assert index.is_file() and sha(index) == PIN["replay_index"]["sha256"]

value = json.loads(index.read_text())
assert value["schema"] == "sepalith.dat10.sourcewalk_raw_index.v3"
assert value["status"] == "complete"
assert value["requested_shards"] == list(range(41))
entries = {item["shard"]: item for item in value["index_files"]}
selected = PIN["selected_range"]["shards"]
assert selected == list(range(27, 41))
assert [item["shard"] for item in PIN["selected_range"]["index_shards"]] == selected
assert sum(entries[shard]["rows"] for shard in selected) == PIN["selected_range"]["index_rows"]
assert sum(entries[shard]["bytes"] for shard in selected) == PIN["selected_range"]["index_bytes"]
for expected in PIN["selected_range"]["index_shards"]:
    observed = entries[expected["shard"]]
    assert {key: observed[key] for key in ("shard", "rows", "bytes", "sha256")} == expected

output = Path(PIN["output"]["path"])
assert sorted(int(path.name.split("-")[1]) for path in (output / "shards").glob("shard-[0-9][0-9][0-9][0-9]") if path.is_dir()) == list(range(27))
print("packet_metadata_checks=pass")
