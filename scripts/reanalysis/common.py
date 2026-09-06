"""Read verified snapshots; no live-source fallback."""

import dataclasses
import hashlib
import json
import types
from pathlib import Path


class Evidence:
    def __init__(self, inventory, snapshot):
        self.manifest = json.loads(Path(inventory).read_text())
        self.snapshot = Path(snapshot)
        self.records = self.manifest["inputs"]

    def record(self, suffix):
        matches = [r for r in self.records if r["path"].endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one input for {suffix}: {len(matches)}")
        return matches[0]

    def read(self, suffix):
        r = self.record(suffix)
        if not r["exists"]:
            raise FileNotFoundError(r["path"])
        data = (self.snapshot / r["sha256"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError(f"Corrupt snapshot: {suffix}")
        return data.decode()

    def json(self, suffix):
        return json.loads(self.read(suffix))

    def rows(self, suffix):
        return [
            json.loads(line) for line in self.read(suffix).splitlines() if line.strip()
        ]

    def module(self, suffix):
        # Explicitly requested frozen Python sources are trusted local inputs.
        mod = types.ModuleType("frozen_source")
        mod.__file__ = self.record(suffix)["path"]
        exec(compile(self.read(suffix), mod.__file__, "exec"), mod.__dict__)  # noqa: S102 - verified trusted local source
        return mod


def keyed(rows, key="id"):
    out = {}
    for r in rows:
        if r[key] in out:
            raise ValueError(f"Duplicate {key}: {r[key]}")
        out[r[key]] = r
    return out


def paired(a, b, fields=("family", "package", "path")):
    a, b = keyed(a), keyed(b)
    if a.keys() != b.keys():
        raise ValueError("Unmatched row IDs")
    for k in sorted(a):
        if any(a[k].get(f) != b[k].get(f) for f in fields):
            raise ValueError(f"Row metadata mismatch: {k}")
        yield a[k], b[k]


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=dataclasses.asdict) + "\n"
    )
