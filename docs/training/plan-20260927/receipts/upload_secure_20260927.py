"""Secure the post-CPT production artifacts on public Hugging Face repos.

Allow-list only. The sealed final evaluation set (DAT-07-final-*, sealed-final/)
and upstream pools that may contain final rows are deliberately NOT uploaded.
Optimizer state and pickled trainer files are not uploaded.
"""

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
MODEL_REPO = "scholzmx/sepalith-2b-cpt"
DATA_REPO = "scholzmx/sepalith"
PREFIX = "campaign-20260915"
CKPT_ROOT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full")
DATA_WORK = Path("/mnt/e/sepalith/campaign-20260915/data-work")
LEAD = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead")
WORK = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work")
STAGE = Path("/mnt/e/sepalith/hf-staging")
CKPT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "campaign-manifest.json",
    "campaign-state.json",
    "trainer_state.json",
]
FORBIDDEN = ("dat-07-final", "sealed-final", "sealed_final", "final-evaluator", "final_eval")


def log(msg):
    print(msg, flush=True)


def guard(path):
    low = str(path).lower()
    if any(token in low for token in FORBIDDEN):
        sys.exit(f"refusing forbidden path: {path}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


MODEL_CARD = """---
license: apache-2.0
base_model: openbmb/MiniCPM5-2B-Midtrain
library_name: transformers
tags:
  - r
  - code
  - continued-pretraining
  - sepalith
---

# Sepalith 2B CPT checkpoints

Full-weight continued-pretraining (CPT) checkpoints of
[openbmb/MiniCPM5-2B-Midtrain](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain)
on about 472M tokens of permissively licensed R code (10,046 packages), trained
for [Sepalith](https://github.com/sims1253/Sepalith), a local-first R
next-edit-suggestion model.

These are base checkpoints. They are **not** tuned for editing or chat.

| Folder | Step | Role |
| --- | ---: | --- |
| `checkpoint-11586/` | 11,586 | Selected parent for editing SFT |
| `checkpoint-11649/` | 11,649 | Final CPT step, preserved alternative |

Architecture: `LlamaForCausalLM`, 42 layers, hidden size 2048, 16 attention
heads with 2 key-value heads, vocabulary 130,560, untied embeddings,
end-of-generation ids 1 and 130073, 2,516,756,480 parameters, bfloat16.

Held-out causal loss (lower is better), identical cases for both checkpoints:

| Context | 11,586 | 11,649 |
| --- | ---: | ---: |
| 2K (499 cases, 50 packages) | 0.94197 | 0.94435 |
| 8K (20 cases, 8 packages) | 0.71110 | 0.71169 |
| 16K (6 cases, 3 packages) | 0.76017 | 0.76103 |

Optimizer state is not published. Training data and provenance live in the
[scholzmx/sepalith](https://huggingface.co/datasets/scholzmx/sepalith) dataset
under `campaign-20260915/`. Released under Apache-2.0, like the base model.
"""


def upload_model():
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    for step in ("11586", "11649"):
        src = CKPT_ROOT / f"checkpoint-{step}"
        for name in CKPT_FILES:
            guard(src / name)
            if not (src / name).is_file():
                sys.exit(f"missing {src / name}")
        log(f"model: uploading checkpoint-{step}")
        api.upload_folder(
            repo_id=MODEL_REPO,
            repo_type="model",
            folder_path=str(src),
            path_in_repo=f"checkpoint-{step}",
            allow_patterns=CKPT_FILES,
            commit_message=f"Add CPT checkpoint {step} (weights, config, tokenizer; no optimizer state)",
        )
    api.upload_file(
        repo_id=MODEL_REPO,
        repo_type="model",
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        commit_message="Add model card",
    )


def upload_data():
    folders = [
        (DATA_WORK / "DAT10-finish-source-repair-v3", f"{PREFIX}/sft/DAT10-finish-source-repair-v3"),
        (DATA_WORK / "DAT10-novel-v1/roxygen-supported-context-v1", f"{PREFIX}/sft/roxygen-supported-context-v1"),
        (LEAD / "corrected-dev75-v1", f"{PREFIX}/dev/corrected-dev75-v1"),
    ]
    files = [
        (LEAD / "r2-full-weight-edit-sft-preparation-v1/repaired15006-draw-schedule.json", f"{PREFIX}/sft/schedules/repaired15006-draw-schedule.json"),
        (LEAD / "r2-full-weight-edit-sft-eval-gate-v1/one-pass-15008-draw-schedule.alternative.json", f"{PREFIX}/sft/schedules/one-pass-15008-draw-schedule.alternative.json"),
        (WORK / "r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl", f"{PREFIX}/cpt/validation/cpt_validation-2k.jsonl"),
        (DATA_WORK / "CPT-prefix-extension-v1/cohort-manifest.json", f"{PREFIX}/cpt/CPT-prefix-extension-v1/cohort-manifest.json"),
        (DATA_WORK / "CPT-prefix-extension-v1/combined-draw-schedule-final-v1.json", f"{PREFIX}/cpt/CPT-prefix-extension-v1/combined-draw-schedule-final-v1.json"),
        (DATA_WORK / "CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl", f"{PREFIX}/cpt/CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl"),
        (STAGE / "campaign-docs-snapshot-20260927.tar.gz", f"{PREFIX}/provenance/campaign-docs-snapshot-20260927.tar.gz"),
        (STAGE / "campaign-docs-snapshot-20260927.files.txt", f"{PREFIX}/provenance/campaign-docs-snapshot-20260927.files.txt"),
    ]
    for src, dst in folders:
        guard(src)
        log(f"data: uploading folder {src} -> {dst}")
        api.upload_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(src), path_in_repo=dst,
                          commit_message=f"Add {dst}")
    for src, dst in files:
        guard(src)
        log(f"data: uploading file {src} -> {dst}")
        api.upload_file(repo_id=DATA_REPO, repo_type="dataset", path_or_fileobj=str(src), path_in_repo=dst,
                        commit_message=f"Add {dst}")


def verify():
    ok = True
    for step in ("11586", "11649"):
        src = CKPT_ROOT / f"checkpoint-{step}" / "model.safetensors"
        info = api.get_paths_info(MODEL_REPO, [f"checkpoint-{step}/model.safetensors"], repo_type="model")[0]
        remote = info.lfs.sha256 if info.lfs else None
        local = sha256(src)
        log(f"verify checkpoint-{step}: local {local[:16]} remote {str(remote)[:16]} size {info.size}")
        ok &= remote == local
    receipt = {"model_repo": MODEL_REPO, "data_repo": DATA_REPO, "prefix": PREFIX, "weights_sha256_verified": ok}
    (STAGE / "upload-receipt.json").write_text(json.dumps(receipt, indent=2))
    log(json.dumps(receipt))
    return ok


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "model"):
        upload_model()
    if stage in ("all", "data"):
        upload_data()
    if stage in ("all", "verify"):
        sys.exit(0 if verify() else 1)
