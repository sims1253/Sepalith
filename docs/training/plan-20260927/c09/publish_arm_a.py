#!/usr/bin/env python3
"""C09 step 3 (R8): publish arm A (checkpoint 536) to scholzmx/sepalith-2b-edit-sft/arm-a/.

    # check inputs, render the model cards, list the commit (writes nothing remote):
    .venv/bin/python docs/training/plan-20260927/c09/publish_arm_a.py --preview-dir /tmp/arm-a-card
    # the night's single model upload (one commit), then sha256 verification:
    HF_TOKEN=... .venv/bin/python docs/training/plan-20260927/c09/publish_arm_a.py --upload \\
        --receipt docs/training/plan-20260927/receipts/C09-publish-arm-a.json

Inputs, all checked before anything is uploaded:

- the checkpoint (``model.safetensors`` must match its campaign manifest);
- ``quantize_and_gate.py``'s work dir: ``run.json`` (GGUF paths and sha256)
  and ``gate-receipt.json``.  The card requirement gates publication: Q8_0
  DEV75 valid/cap/edit/nofp within 2 of the bf16 GGUF (``dev75_parity``).
  Q4_K_M drift is recorded, never gated.  The proposed quant-gate verdicts go
  into the card as information (their thresholds await the user's sign-off);
- ``cpt_forgetting.py``'s result (parent plus ``arm-a-536``).

Uploaded, in one commit: ``arm-a/`` with config, generation config,
tokenizer, chat template, ``model.safetensors``, ``arm-a-Q8_0.gguf``,
``arm-a-Q4_K_M.gguf`` (imatrix, q8_0 output head and embeddings) and
``README.md``; plus the repository ``README.md``.  Never optimizer, RNG,
scheduler or trainer state, never another checkpoint.  The script refuses if
``arm-a/model.safetensors`` already exists remotely.

Exit codes: 0 done (or preview written); 3 stop-for-user (the message says
why); anything else is an error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ID = "scholzmx/sepalith-2b-edit-sft"
FOLDER = "arm-a"
CHECKPOINT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/full/checkpoint-536")
WORK = Path("/mnt/e/sepalith/campaign-20260915/quant-work/arm-a")
FORGETTING = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/SFT-v2-r4-lr1e-5/cpt-forgetting-536.json")
CHECKPOINT_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                    "tokenizer_config.json", "chat_template.jinja")
GGUF_LABELS = {"Q8_0": "arm-a-Q8_0.gguf", "Q4_K_M": "arm-a-Q4_K_M.gguf"}
FORBIDDEN = ("optimizer", "rng_state", "scheduler", "trainer_state", "training_args", "dat-07-final", "sealed-final")

RECIPE = "docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json"
RECIPE_SHA256 = "de4bcb0a6da1f3481874aeb82a19d80eae0ec6389b87ba18d1e4a9ca25bf2a4e"
PARENT_WEIGHTS_SHA256 = "4bf617f9ea66a31e547397b93f0998372686e076d03ba22961780abd285a3c51"
# PC bf16 gate results for the arm (status/C09.json): DEV75 of 75 (43 edit, 32 no-op), DEV250 of 250 (200 edit, 50 no-op).
PC_GATES = {
    268: {"valid": 73, "cap": 2, "edit": 26, "nofp": 11, "score250": 110, "edit250": 76, "noop250": 34},
    536: {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117, "edit250": 79, "noop250": 38},
    803: {"valid": 75, "cap": 0, "edit": 28, "nofp": 9, "score250": 109, "edit250": 76, "noop250": 33},
    1071: {"valid": 74, "cap": 1, "edit": 28, "nofp": 9, "score250": 105, "edit250": 76, "noop250": 29},
}


class StopForUser(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise StopForUser(f"missing input {path}")
    return json.loads(path.read_text())


def r1_counts(dev75: dict, dev250: dict | None) -> dict:
    c75 = dev75["summary"]["counts"]
    out = {"valid": c75["protocol_valid"], "cap": c75["cap_hit"], "edit": c75["edit_exact"],
           "nofp": c75["strict_noop_false_suggestions"]}
    if dev250:
        c250 = dev250["summary"]["counts"]
        out |= {"score250": c250["edit_exact"] + c250["strict_noop_correct"], "edit250": c250["edit_exact"],
                "noop250": c250["strict_noop_correct"]}
    return out


def collect(work: Path, checkpoint: Path, forgetting_path: Path) -> dict:
    manifest = load(checkpoint / "campaign-manifest.json")
    files = {}
    for name in CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file():
            raise StopForUser(f"missing checkpoint file {path}")
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        if name in manifest["files"] and manifest["files"][name]["sha256"] != files[name]["sha256"]:
            raise StopForUser(f"{path} differs from the campaign manifest")
    if manifest["identity"]["parent"]["weights_sha256"] != PARENT_WEIGHTS_SHA256:
        raise StopForUser("checkpoint parent is not CPT 11,586")

    run = load(work / "run.json")
    receipt = load(work / "gate-receipt.json")
    if receipt.get("gate_panel") != "dev250":
        raise StopForUser("gate receipt is not a full DEV250 run")
    tiers = {tier["label"]: tier for tier in receipt["tiers"]}
    ggufs = {}
    for label, remote in GGUF_LABELS.items():
        record = run["ggufs"].get(label)
        if not record or label not in tiers:
            raise StopForUser(f"no {label} GGUF in the quant run")
        path = Path(record["path"])
        if sha256(path) != record["sha256"] or tiers[label]["gguf"]["sha256"] != record["sha256"]:
            raise StopForUser(f"{path} differs from run.json or the gate receipt")
        ggufs[label] = {"path": str(path), "remote": remote, "bytes": record["bytes"], "sha256": record["sha256"]}
    parity = tiers["Q8_0"].get("dev75_parity") or {}
    if parity.get("outcome") != "pass":
        raise StopForUser(f"card C09: Q8_0 DEV75 counts not within 2 of bf16: {parity}")

    results = {}
    for label in ("bf16", "Q8_0", "Q4_K_M"):
        dev = {panel: load(work / "eval" / f"{label}-{panel}.json") for panel in ("dev75", "dev250")}
        if any(value["status"] != "complete" or value.get("smoke") for value in dev.values()):
            raise StopForUser(f"{label} DEV results are incomplete or smoke runs")
        results[label] = r1_counts(dev["dev75"], dev["dev250"])
        if label != "bf16":
            results[label]["kld"] = tiers[label].get("kld")
            results[label]["gate_verdict"] = tiers[label]["verdict"]
            results[label]["shippable"] = tiers[label]["shippable"]
            results[label]["token_identity_pct"] = tiers[label]["paired"]["token_identity"]["pct"]
    runtime = receipt["reference"].get("runtime") or {}

    forgetting = load(forgetting_path)
    if forgetting.get("status") != "complete":
        raise StopForUser("CPT forgetting result is not complete")
    comparison = next((c for c in forgetting["comparisons"] if c["label"] == "arm-a-536"), None)
    model = next((m for m in forgetting["models"] if m["label"] == "arm-a-536"), None)
    if not comparison or not model or model["model_safetensors_sha256"] != files["model.safetensors"]["sha256"]:
        raise StopForUser("CPT forgetting result does not cover this checkpoint")
    return {"manifest": manifest, "files": files, "ggufs": ggufs, "results": results, "runtime": runtime,
            "gate_receipt": {"path": str(work / "gate-receipt.json"), "sha256": sha256(work / "gate-receipt.json"),
                             "smallest_shippable": receipt.get("smallest_shippable")},
            "forgetting": {"path": str(forgetting_path), "sha256": sha256(forgetting_path),
                           "nll": model["mean_causal_nll"], "parent_reproduction": forgetting.get("parent_reproduction"),
                           "comparison": comparison}}


def fmt_kld(result: dict) -> str:
    kld = result.get("kld") or {}
    if "mean_kld" not in kld:
        return "n/a"
    return f"{kld['mean_kld']:.4f} / {kld['same_top_p_pct']:.1f}%"


def arm_card(info: dict) -> str:
    ident = info["manifest"]["identity"]
    f = info["forgetting"]
    cmp = f["comparison"]
    repro = f["parent_reproduction"] or {}
    res = info["results"]
    gate_rows = "\n".join(
        f"| {step}{' (published)' if step == 536 else ''} | {g['valid']} | {g['cap']} | {g['edit']} | {g['nofp']} | "
        f"{g['edit250']} | {g['noop250']} | {g['score250']} |" for step, g in PC_GATES.items())
    gguf_rows = "\n".join(
        f"| {label} | {r['valid']} | {r['cap']} | {r['edit']} | {r['nofp']} | {r['edit250']} | {r['noop250']} | "
        f"{r['score250']} | {fmt_kld(r) if label != 'bf16' else '-'} |" for label, r in res.items())
    sizes = {label: f"{g['bytes'] / 1e9:.2f} GB" for label, g in info["ggufs"].items()}
    lo, hi = cmp["package_bootstrap95_delta"]
    return f"""# Arm A: editing SFT, checkpoint 536

Full-weight editing SFT of the Sepalith 2B CPT checkpoint 11,586
([scholzmx/sepalith-2b-cpt](https://huggingface.co/scholzmx/sepalith-2b-cpt),
folder `checkpoint-11586/`) for R next-edit suggestion under the
`zeta2-prm03-v1` prompt and output contract (a no-op answer is `[NO_EDIT]`).
Feed pre-tokenized prompt ids; end-of-generation ids are 1 and 130073.

## Lineage

| Item | Value |
| --- | --- |
| Parent | CPT step 11,586, `model.safetensors` sha256 `{PARENT_WEIGHTS_SHA256}` |
| Recipe | `{RECIPE}` (sha256 `{RECIPE_SHA256[:16]}...`) in [sims1253/Sepalith](https://github.com/sims1253/Sepalith) |
| SFT schedule v2 | draw schedule sha256 `{ident['data']['draw_schedule_sha256']}` (18.75% no-op, each no-op row at most 3 times) |
| TRAIN rows | `{ident['data']['cohort_id']}`, rows sha256 `{ident['data']['rows_sha256'][:16]}...` |
| Learning rate | 1e-5 (selected by the pre-registered LR pilot over 3e-6, 1e-5, 3e-5) |
| Step | 536 of 1,071, chosen by the user: most exact edits over both DEV panels (106 of 243) and less drift from the parent than 803 and 1,071 |
| Weights | `model.safetensors` sha256 `{info['files']['model.safetensors']['sha256']}` |

Checkpoints 803 and 1,071 of the same run are not published. No optimizer
state is included.

## Development results (PC, bf16, greedy, cap 1,024)

DEV75 is a restraint-stress panel (43 edit, 32 no-op cases); DEV250 follows
the TRAIN family mix (200 edit, 50 no-op). `nofp` counts no-op cases where the
model suggested an edit anyway (lower is better).

| Step | valid/75 | cap | edit/43 | nofp/32 | DEV250 edit/200 | DEV250 no-op/50 | score250 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{gate_rows}

The CPT parent scores 0 on both panels: it never produces the PRM03 format.

## GGUF (llama.cpp b10453, {info['runtime'].get('build', '?')} build, greedy, cap 1,024, one request per case)

| File | valid/75 | cap | edit/43 | nofp/32 | DEV250 edit | DEV250 no-op | score250 | mean KLD / top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{gguf_rows}

`arm-a-Q8_0.gguf` ({sizes['Q8_0']}) is the reference serving file.
`arm-a-Q4_K_M.gguf` ({sizes['Q4_K_M']}) uses an importance matrix from 132
TRAIN-only calibration chunks, with the output head and token embeddings at
q8_0. KLD is measured against the bf16 GGUF on a held-out TRAIN slice.
llama.cpp generation is batch 1 and is not token-identical to the PC's
batched generation, so the GGUF bf16 row is the reference for the quantized
files.

## Forgetting check (CPT 2K held-out causal loss, 499 rows, 50 packages)

| Model | Mean causal NLL |
| --- | ---: |
| CPT parent 11,586 (recorded during CPT) | 0.94197 |
| CPT parent 11,586 (this harness) | {repro.get('this_harness', float('nan')):.5f} |
| Arm A, step 536 | {f['nll']:.5f} |

Change against the parent in the same harness: {cmp['delta_vs_reference']:+.5f}
({cmp['relative_change_percent_vs_reference']:+.2f}%; package bootstrap 95% interval
{lo:+.5f} to {hi:+.5f}); lower on {cmp['rows_lower_than_reference']} of {cmp['rows']} rows.

## Limits

Development panels only. The sealed final evaluation set has not been opened.
Exact match against one reference edit undercounts valid alternatives.

Released under Apache-2.0, like the base model
[openbmb/MiniCPM5-2B-Midtrain](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain).
"""


REPO_CARD = """---
license: apache-2.0
base_model: scholzmx/sepalith-2b-cpt
library_name: transformers
tags:
  - r
  - code
  - next-edit-suggestion
  - sepalith
  - gguf
---

# Sepalith 2B editing SFT

Editing-SFT models for [Sepalith](https://github.com/sims1253/Sepalith), a
local-first R next-edit-suggestion model served as GGUF through llama.cpp.
Each folder is one adopted model with weights, config, tokenizer, GGUF files
and its own model card. No optimizer state.

| Folder | Model |
| --- | --- |
| [`arm-a/`](arm-a/README.md) | SFT arm A: CPT 11,586, schedule v2, LR 1e-5, step 536 |
"""


def operations(info: dict, arm_readme: str):
    ops = [(f"{FOLDER}/{name}", Path(record["path"])) for name, record in info["files"].items()]
    ops += [(f"{FOLDER}/{g['remote']}", Path(g["path"])) for g in info["ggufs"].values()]
    for remote, path in ops:
        if any(token in remote.lower() or token in str(path).lower() for token in FORBIDDEN):
            raise SystemExit(f"refusing forbidden file {path}")
    ops += [(f"{FOLDER}/README.md", arm_readme.encode()), ("README.md", REPO_CARD.encode())]
    return ops


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--forgetting", type=Path, default=FORGETTING)
    ap.add_argument("--preview-dir", type=Path, help="write the rendered cards and the commit plan here")
    ap.add_argument("--upload", action="store_true", help="create the public repo and make the single commit")
    ap.add_argument("--receipt", type=Path, help="upload receipt JSON")
    a = ap.parse_args(argv)
    try:
        info = collect(a.work_dir, a.checkpoint, a.forgetting)
    except StopForUser as stop:
        print(f"STOP FOR USER: {stop}", file=sys.stderr)
        return 3
    arm_readme = arm_card(info)
    ops = operations(info, arm_readme)
    plan = [{"path_in_repo": remote, "source": str(src) if isinstance(src, Path) else "<rendered>"} for remote, src in ops]
    for item in plan:
        print(f"{item['path_in_repo']} <- {item['source']}")
    if a.preview_dir:
        a.preview_dir.mkdir(parents=True, exist_ok=True)
        (a.preview_dir / "arm-a-README.md").write_text(arm_readme)
        (a.preview_dir / "README.md").write_text(REPO_CARD)
        (a.preview_dir / "commit-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not a.upload:
        print("preview only; pass --upload for the single model commit")
        return 0
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    if api.file_exists(REPO_ID, f"{FOLDER}/model.safetensors", repo_type="model"):
        print(f"STOP FOR USER: {REPO_ID}/{FOLDER}/model.safetensors already exists", file=sys.stderr)
        return 3
    commit = api.create_commit(
        repo_id=REPO_ID, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(src) if isinstance(src, Path) else src)
                    for remote, src in ops],
        commit_message="Add arm-a: editing SFT checkpoint 536 (weights, config, tokenizer, GGUF Q8_0/Q4_K_M, card; no optimizer state)")
    verified = {}
    large = {f"{FOLDER}/model.safetensors": info["files"]["model.safetensors"]["sha256"],
             **{f"{FOLDER}/{g['remote']}": g["sha256"] for g in info["ggufs"].values()}}
    for remote_info in api.get_paths_info(REPO_ID, list(large), repo_type="model", revision=commit.oid):
        verified[remote_info.path] = bool(remote_info.lfs) and remote_info.lfs.sha256 == large[remote_info.path]
    receipt = {"schema": "sepalith.c09.publish-arm-a.v1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "repo": REPO_ID, "commit": commit.oid, "commit_url": commit.commit_url, "files": plan,
               "checkpoint_files": info["files"], "ggufs": info["ggufs"], "results": info["results"],
               "gate_receipt": info["gate_receipt"], "forgetting": info["forgetting"], "sha256_verified": verified,
               "optimizer_state_uploaded": False, "final_set_access": False}
    if a.receipt:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        a.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"commit": commit.commit_url, "sha256_verified": verified}, indent=2))
    return 0 if len(verified) == len(large) and all(verified.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
