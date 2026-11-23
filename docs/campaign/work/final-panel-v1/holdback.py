#!/usr/bin/env python3
"""Write or check the pre-registered holdback split of one DAT-02 split. Group IDs only.

It parses exactly the pinned DAT-02 bytes and reads each group's `group_id`,
`split` and `flags`. It opens no parent, source or tarball, and it prints
counts and the file's sha256 only.

    python docs/campaign/work/final-panel-v1/holdback.py write --split final_candidate_group --out FILE
    python docs/campaign/work/final-panel-v1/holdback.py check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_panel as F  # noqa: E402


def split_for(split: str, recipe: F.Json) -> F.Json:
    pin = recipe["inputs"]["dat02"]
    data = Path(pin["path"]).read_bytes()
    if F.sha256_bytes(data) != pin["sha256"]:
        raise F.PanelError("dat02_pin_mismatch")
    return F.holdback_split(json.loads(data), split, recipe["holdback"]["build_groups"], pin["sha256"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--split", required=True, choices=tuple(F.AUTHORIZED_SPLITS))
    write.add_argument("--out", type=Path, required=True)
    sub.add_parser("check", help="recompute the final split and compare it with the committed file and the pin")
    args = parser.parse_args(argv)
    recipe = F.load_recipe()
    if args.command == "write":
        doc = split_for(args.split, recipe)
        data = F.holdback_bytes(doc)
        F._write_readonly(args.out.resolve(), data)
    else:
        doc = split_for(F.FINAL_SPLIT, recipe)
        data = F.holdback_bytes(doc)
        committed = (F.ROOT / recipe["holdback"]["split_file"]).read_bytes()
        if data != committed or F.sha256_bytes(data) != recipe["holdback"]["final_split_sha256"]:
            print("holdback split differs from the committed file or the recipe pin", file=sys.stderr)
            return 1
    print(json.dumps({"split": doc["split"], "eligible_groups": doc["eligible_groups"],
                      "build_groups": len(doc["build_group_ids"]), "holdback_groups": len(doc["holdback_group_ids"]),
                      "sha256": F.sha256_bytes(data)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
