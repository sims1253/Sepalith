#!/usr/bin/env python3
"""Unified HF projection for the sepalith dataset (v2 layout).

The NAS layout grew organically (waves write wherever); this script
PROJECTS it into one clean structure without moving local files:

  corpus/                      acquired, license-tracked
    cran/packages/...          per-package shards (+ manifest, licenses)
    provenance/<pkg>.json      per-package license/provenance
    hidden-r/...               harvested general-code R
    edit-pairs/...             mined git edit pairs
  families/<family>/<source>.jsonl[.stats.json|.done.jsonl]
                               ALL synthetic families, family-first,
                               one file per author-source (purge-safe)
  mixtures/sft_vX/...          assembled train/eval (derived data)

Source labels: glm-5.3 (zai), gemini-3.7-flash (agy), muse-spark-1.2
(opencode GO/free), x-preview (zen), model-id (openrouter), corpus
(deterministic/mined — no LLM authorship).

Also deletes legacy repo paths (the era-1/2 scattered layout) so the
repo converges to exactly the projection. State-file incremental;
HF_TOKEN from env.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "scholzmx/sepalith"
NAS = Path("/mnt/h/sepalith/datasets")
LOCAL_CORPUS = Path("/home/m0hawk/Documents/Sepalith")  # hidden-r/edit-pairs live repo-side
STATE = Path(__file__).parent / ".push_cases_state.json"

# backend/model -> canonical source label
SOURCE = {
    "zai": "glm-5.3", "agy": "gemini-3.7-flash",
    "opencode-spark": "muse-spark-1.2", "opencode-spark-free": "muse-spark-1.2",
    "xpreview-free": "x-preview", "sparkfree": "muse-spark-1.2",
    "spark": "muse-spark-1.2",
}
OPENROUTER_MODEL_RE = re.compile(r"orfree_(.+)\.jsonl$")

# exact filenames whose author-source the suffix rules can't see
EXPLICIT_SOURCE = {
    "rewrite_lint_fix.jsonl": "glm-5.3",        # zai quota-burn wave
    "fix_issue_inject.jsonl": "glm-5.3",
    "analyst_direct.jsonl": "glm-5.3", "analyst_gemini.jsonl": "gemini-3.7-flash",
    "comment_to_code_gemini.jsonl": "gemini-3.7-flash",
}


def source_label(fname: str) -> str:
    if fname in EXPLICIT_SOURCE:
        return EXPLICIT_SOURCE[fname]
    m = OPENROUTER_MODEL_RE.search(fname)
    if m:
        return m.group(1).replace("--", "/")
    for key, lab in SOURCE.items():
        if fname.endswith(f"_{key}.jsonl") or f"_{key}." in fname:
            return lab
    return "corpus"


def fam_of(fname: str) -> str:
    """cases_v1 wave names embed the author; strip it for the family."""
    stem = fname
    for key in ["_sparkfree", "_xpreview-free"] + [f"_{k}" for k in SOURCE]:
        stem = stem.replace(key, "")
    m = OPENROUTER_MODEL_RE.search(stem)
    if m:
        stem = stem[: m.start()] + stem[m.end():]
    return stem.replace(".jsonl", "").rstrip("_-") or "misc"


CARD = Path("/mnt/h/sepalith/datasets/hf_dataset_card.md")


def projection() -> list[tuple[Path, str]]:
    """(local file, repo path) pairs — the whole intended repo content."""
    out: list[tuple[Path, str]] = [(CARD, "README.md")]

    # -- corpus (from the era-1 pushes + NAS) --
    out.append((NAS / "license_texts.jsonl", "corpus/cran/licenses.jsonl"))
    out.append((NAS / "manifest.jsonl", "corpus/cran/manifest.jsonl"))
    packages = sorted((NAS / "packages").glob("*/*.jsonl")) if (NAS / "packages").exists() else []
    for p in packages:
        out.append((p, f"corpus/cran/packages/{p.parent.name}/{p.name}"))
    prov = sorted(NAS.glob("provenance/*.json"))
    for p in prov:
        out.append((p, f"corpus/provenance/{p.name}"))
    for name in ("codex_r_strict.jsonl", "ling_coder_r.jsonl"):
        f = LOCAL_CORPUS / "experiments" / "corpus" / name
        if not f.exists():
            f = NAS / "hidden_r" / name
        if f.exists():
            out.append((f, f"corpus/hidden-r/{name}"))
    ep = LOCAL_CORPUS / "experiments" / "post-processing" / "edit_pairs"
    for rel in ("train.jsonl", "eval.jsonl", "pr_instructed.jsonl"):
        f = ep / rel
        if not f.exists():
            f = NAS / "edit_pairs" / rel
        if f.exists():
            out.append((f, f"corpus/edit-pairs/{rel}"))

    # -- families: scenarios_v1 + cases_v1, family-first --
    # main rows: families/<family>/<source>.jsonl
    # sidecars:  families/<family>/<source>.done.jsonl|.stats.json
    for base in (NAS / "scenarios_v1", NAS / "cases_v1"):
        if not base.exists():
            continue
        for f in sorted(base.iterdir()):
            if ("contaminated" in f.name or ".partial." in f.name
                    or f.name.endswith((".done.jsonl", ".stats.json"))
                    or f.name in ("stats.json", "suffix_scenarios.stats.json")):
                continue  # sidecars ride via their .jsonl twin; partials are transient
            if f.suffix != ".jsonl":
                continue
            fam, src = fam_of(f.name), source_label(f.name)
            out.append((f, f"families/{fam}/{src}.jsonl"))
            for suf in (".done.jsonl", ".stats.json"):
                side = base / (f.name + suf)
                if side.exists():
                    out.append((side, f"families/{fam}/{src}{suf}"))

    # -- analyst families --
    sa = NAS / "synthetic_analyst_v1"
    if sa.exists():
        for f in sorted(sa.glob("*.jsonl")):
            out.append((f, f"families/synthetic_analyst/{f.name}"))

    # -- mixtures --
    for v in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7",
              "sft_v8_1", "sft_v8_2"):
        d = NAS / v
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")) + sorted(d.glob("stats.json")):
            out.append((f, f"mixtures/{v}/{f.name}"))

    # -- corpus ledgers/manifests for EVERY acquired corpus (the public
    # licensing story is auditable from this repo; heavy raw content
    # stays on the source NAS and is documented in the dataset card).
    # CRAN's full corpus/ was already here (era-1); the newer corpora
    # project at ledger+manifest level.
    for src_p, dst in [
        ("/mnt/h/sepalith/bioc_staging/bioc_license_ledger.jsonl",
         "corpus/bioc/license_ledger.jsonl"),
        ("/mnt/h/sepalith/bioc_staging/manifest.jsonl",
         "corpus/bioc/manifest.jsonl"),
        ("/mnt/h/sepalith/stack_staging/license_ledger.jsonl",
         "corpus/stack_v2/license_ledger.jsonl"),
        ("/mnt/h/sepalith/stack_staging/measure_report.json",
         "corpus/stack_v2/measure_report.json"),
        ("/mnt/h/sepalith/stack_staging/logs/phase_c_report.json",
         "corpus/stack_v2/dedup_report.json"),
        ("/mnt/h/sepalith/stack_staging/logs/phase_e_report.json",
         "corpus/stack_v3/acceptance_report.json"),
        ("/mnt/h/sepalith/pwc_staging/pwc_license_ledger.jsonl",
         "corpus/pwc/license_ledger.jsonl"),
        ("/mnt/h/sepalith/pwc_staging/measure_report.json",
         "corpus/pwc/measure_report.json"),
        ("/mnt/h/sepalith/pwc_staging/manifest.jsonl",
         "corpus/pwc/manifest.jsonl"),
    ]:
        if Path(src_p).exists():
            out.append((Path(src_p), dst))

    # -- pretraining/: THE A2 TRAINING PACKAGE (2026-08-26). Everything a
    #    rented instance needs, laid out exactly as run.py's data-root, so:
    #      hf download scholzmx/sepalith --repo-type dataset \
    #        --include "pretraining/*" --local-dir /data
    #      python3 run.py all --data-root /data
    #    Blocks are int32 1025-overlap-1 (tokenizer a2_tokenizer_v1);
    #    manifest/stats/contamination make the draw plan + gates portable.
    a2r = Path("/mnt/h/sepalith/a2/r")
    if a2r.exists():
        for npy in sorted(a2r.glob("*.npy")):
            out.append((npy, f"pretraining/a2/r/{npy.name}"))
        for j in ("stats.json", "contamination.json"):
            f = a2r / j
            if f.exists():
                out.append((f, f"pretraining/a2/r/{j}"))
    tokf = NAS / "a2_tokenizer_v1" / "tokenizer.json"
    if tokf.exists():
        out.append((tokf,
                    "pretraining/datasets/a2_tokenizer_v1/tokenizer.json"))
    transfers = Path("/mnt/h/sepalith/a2_transfers")
    if transfers.exists():
        for d in sorted(transfers.iterdir()):
            b = d / "blocks.npy"
            if not b.exists():
                continue
            out.append((b, f"pretraining/a2_transfers/{d.name}/blocks.npy"))
            s2 = d / "stats.json"
            if s2.exists():
                out.append((s2,
                            f"pretraining/a2_transfers/{d.name}/stats.json"))
    man = Path("/mnt/h/sepalith/a2_mixture_manifest.json")
    if man.exists():
        out.append((man, "pretraining/a2_mixture_manifest.json"))

    return out


def legacy_paths() -> list[str]:
    return ["datasets/scenarios_v1", "datasets/synthetic_analyst_v1", "datasets/sft_v3",
            "synthetic/scenarios_v1", "synthetic/synthetic_analyst_v1",
            "synthetic/cases_v1", "synthetic/sft_v5", "synthetic/sft_v6",
            "synthetic/analyst_direct.jsonl", "synthetic/analyst_scripts.jsonl",
            "synthetic/na_rm_contexts.jsonl", "synthetic/paper_to_r.jsonl",
            "scenarios", "edit-pairs", "finish-block", "hidden-r",
            "license_texts.jsonl", "provenance_manifest.jsonl"]


def main() -> int:
    full = "--full" in sys.argv
    api = HfApi(token=os.environ["HF_TOKEN"])
    pairs = projection()
    print(f"projection: {len(pairs)} files", flush=True)

    # 0. the map FIRST — it must land even if later steps rate-limit
    #    (the real card lives on the NAS: datasets/hf_dataset_card.md)
    if CARD.exists():
        api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                        repo_id=REPO, repo_type="dataset")
    else:
        card = """# Sepalith dataset

Open, R-specialized next-edit-suggestion training data. Private.

## Layout
- `corpus/` — acquired, license-tracked sources (CRAN shards + provenance,
  bioc/stack/pwc ledgers; heavy raw content NAS-resident, see card).
- `families/<family>/<source>.jsonl` — synthetic case families, one file
  per author-source, purge-safe, `base_sample_id` provenance.
- `mixtures/sft_vX/` — assembled train/eval splits (derived).
- `pretraining/` — the A2 training package (blocks + tokenizer +
  manifest + gate artifacts; consumed by run.py on the instance).
"""
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=REPO, repo_type="dataset")


    # 1. delete legacy layout (idempotent; missing folders are skipped)
    for path in legacy_paths():
        try:
            api.delete_folder(repo_id=REPO, repo_type="dataset", path_in_repo=path)
            print(f"deleted legacy {path}/", flush=True)
        except Exception as e:
            if "not found" not in str(e).lower() and "404" not in str(e):
                print(f"(skip {path}: {str(e)[:80]})", flush=True)

    # 2. push the projection: staging dir + per-top-dir upload_folder
    #    (3-6 COMMITS per run instead of one per file — the HF hourly
    #    commit budget is 128 and per-file uploads exhausted it)
    import shutil
    stage = Path("/tmp/hf_projection")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    state = {} if full else json.loads(STATE.read_text()) if STATE.exists() else {}
    staged = 0
    for f, rel in pairs:
        if not f.exists():
            continue
        sig = {"size": f.stat().st_size, "mtime": f.stat().st_mtime}
        if not full and state.get(rel) == sig:
            continue
        tgt = stage / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, tgt)
        state[rel] = sig
        staged += 1
    pushed = 0
    for top in sorted({r.split("/", 1)[0] for _, r in pairs}):
        d = stage / top
        if d.is_file():
            api.upload_file(path_or_fileobj=str(d), path_in_repo=top,
                            repo_id=REPO, repo_type="dataset")
            pushed += 1
            print(f"pushed {top} (file)", flush=True)
            continue
        if not d.exists() or not any(d.iterdir()):
            continue
        api.upload_folder(folder_path=str(d), path_in_repo=top,
                          repo_id=REPO, repo_type="dataset")
        pushed += 1
        print(f"pushed {top}/ ({sum(1 for _ in d.rglob(chr(42)))} files)", flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    STATE.write_text(json.dumps(state, indent=1))
    skipped = len(pairs) - staged

    print(f"done: pushed={pushed} skipped={skipped} -> "
          f"https://huggingface.co/datasets/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
