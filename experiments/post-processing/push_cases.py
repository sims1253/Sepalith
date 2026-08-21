#!/usr/bin/env python3
"""Push the vetted synthetic-family data to the private HF hub repo,
incrementally.

  push_cases.py            # push changed/new files under cases_v1/ (+ the
                           # assembled sft_vX datasets on first sight)
  push_cases.py --full     # re-push everything (ignore state)

State: experiments/post-processing/.push_cases_state.json maps
repo-path -> {size, mtime}; only changed files upload. Family files grow
in place (the waves append), so a changed file is re-uploaded whole —
fine at these sizes. .done/.stats sidecars ride along (provenance).

Repo: the main scholzmx/sepalith dataset (private) — CRAN shards live
under datasets/, the synthetic families under synthetic/. HF_TOKEN env.
"""
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "scholzmx/sepalith"  # one dataset: packages under datasets/, synthetic under synthetic/
ROOT = Path("/mnt/h/sepalith/datasets")
STATE = Path(__file__).parent / ".push_cases_state.json"
COVERED = [
    ROOT / "cases_v1",            # the wave programs (rewrite/compound/doc_sync/...)
    ROOT / "synthetic_analyst_v1",
    ROOT / "scenarios_v1",        # established families (grew this session)
]
# assembled mixtures are large; push them but only when they change
COVERED += [ROOT / f"sft_v{i}" for i in (5, 6)]


def main() -> int:
    full = "--full" in sys.argv
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)

    state = {} if full else json.loads(STATE.read_text()) if STATE.exists() else {}
    pushed = skipped = 0
    t0 = time.time()
    for base in COVERED:
        if not base.exists():
            continue
        rel_base = base.relative_to(ROOT)
        for f in sorted(base.glob("*.jsonl")) + sorted(base.glob("*.json")):
            if f.name.endswith(".contaminated.bak"):
                continue
            rel = f"synthetic/{rel_base}/{f.name}"
            st = f.stat()
            sig = {"size": st.st_size, "mtime": st.st_mtime}
            if not full and state.get(rel) == sig:
                skipped += 1
                continue
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rel,
                            repo_id=REPO, repo_type="dataset")
            state[rel] = sig
            pushed += 1
            print(f"pushed {rel} ({st.st_size/1e6:.1f} MB)", flush=True)
    STATE.write_text(json.dumps(state, indent=1))
    print(f"done: pushed={pushed} skipped-unchanged={skipped} "
          f"elapsed={time.time()-t0:.0f}s -> https://huggingface.co/datasets/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
