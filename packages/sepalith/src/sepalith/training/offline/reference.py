#!/usr/bin/env python3
"""Reference log-probabilities of every DPO pair under the starting model (card C13 step 4; night only).

The model is loaded exactly as the DPO trainer loads it (unsloth, bf16, full
weights, training mode with gradient checkpointing), and each completion's
summed log-probability comes from the same ``offline.dpo.completion_logp``
under ``torch.no_grad``, so at update 0 the policy reproduces the stored values
(the trainer's reference-alignment gate checks this). Before the first pair, a
parity check compares ``completion_logp`` with the fused cross-entropy loss of
the labelled forward pass on the first pairs.

Output: ``reference.logps`` (JSONL in pair-file order: pair id, chosen and
rejected log-probabilities and token counts), written in flushed blocks. The
job is resumable: a rerun keeps the verified prefix and continues. At the
deadline (``SEPALITH_DEADLINE_EPOCH``), on SIGTERM or on the stop-request file
it flushes and exits 0 without a manifest (``cuda_host_guard`` reports any
nonzero child exit as a failure, so the night driver tells "deferred" from
"complete" by the manifest). When every pair is done it writes
``reference.manifest`` (parent weights, pair file and log-probability file
hashes) and exits 0.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import signal
import time
from types import FrameType
from typing import cast

from sepalith.training.offline.recipe import REFERENCE_SCHEMA, load_offline
from sepalith.training.sft.recipe_v2 import Json, require, sha256

EXIT_DONE = 0
FLUSH_EVERY = 32
PARITY_PAIRS = 2


def verified_prefix(path: Path, pair_ids: Sequence[str]) -> list[Json]:
    """Complete, in-order rows already written; a torn last line is dropped."""
    if not path.exists():
        return []
    rows: list[Json] = []
    keep = 0
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                break
            row = cast(Json, json.loads(raw))
            index = len(rows)
            require(index < len(pair_ids) and row.get("pair_id") == pair_ids[index], "reference rows are out of pair order")
            require(all(math.isfinite(float(cast(float, row[k]))) for k in ("chosen_logp", "rejected_logp")), "non-finite stored value")
            rows.append(row)
            keep += len(raw)
    if keep != path.stat().st_size:
        with path.open("r+b") as stream:
            stream.truncate(keep)
    return rows


def run(recipe_path: Path) -> int:
    recipe = load_offline(recipe_path, require_reference=False)
    require(recipe["kind"] == "dpo", "reference log-probabilities belong to a DPO recipe")
    record = cast(Json, recipe["reference"])
    logps_path, manifest_path = Path(cast(str, record["logps"])), Path(cast(str, record["manifest"]))
    if manifest_path.exists():
        from sepalith.training.offline.recipe import reference_manifest

        reference_manifest(recipe)
        print(json.dumps({"status": "already_complete", "manifest": str(manifest_path)}))
        return EXIT_DONE
    cohort = cast(Json, recipe["cohort"])
    pairs_record = cast(Json, cohort["rows"])
    pairs_path = Path(cast(str, pairs_record["path"]))
    require(sha256(pairs_path) == pairs_record["sha256"], "DPO pair file differs")
    with pairs_path.open() as stream:
        pairs = [cast(Json, json.loads(line)) for line in stream if line.strip()]
    pair_ids = [cast(str, p["pair_id"]) for p in pairs]
    logps_path.parent.mkdir(parents=True, exist_ok=True)
    done = verified_prefix(logps_path, pair_ids)

    stop = {"requested": False}

    def on_signal(_number: int, _frame: FrameType | None) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGTERM, on_signal)
    from sepalith.training.sft.night_controls import deadline_from_env

    deadline = deadline_from_env()
    stop_file = os.environ.get("SEPALITH_STOP_REQUEST_FILE")

    def should_stop() -> bool:
        return stop["requested"] or (deadline is not None and time.time() >= deadline) or bool(stop_file and Path(stop_file).exists())

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from unsloth import FastLanguageModel
    import torch

    from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
    from sepalith.training.contracts.trainer_tokenizer_alignment import (
        assert_runtime_tokenizer,
        embedding_identity,
        restore_trainer_eog_alignment,
    )
    from sepalith.training.offline.dpo import completion_logp, validate_pair

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one root-owned CUDA device required")
    parent = cast(Json, recipe["parent"])
    parent_path = cast(str, parent["path"])
    started = time.monotonic()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=parent_path, max_seq_length=cast(int, cohort["max_sequence_tokens"]),
                                                         dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=True,
                                                         float32_mixed_precision=False, fast_inference=False, trust_remote_code=False,
                                                         use_gradient_checkpointing=True)
    reference_tokenizer = load_pinned_reference_tokenizer(Path(parent_path))
    repair = restore_trainer_eog_alignment(model, tokenizer, reference_tokenizer)
    require(repair.get("repair", {}).get("vocab_mapping_unchanged") is True, "post-load token repair changed vocabulary")
    audit = assert_runtime_tokenizer(model, tokenizer, reference_tokenizer, embedding_identity(model), "reference_load")
    model.config.use_cache = False
    try:
        FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    except TypeError:
        FastLanguageModel.for_training(model)
    model.train()
    device = torch.device("cuda")

    def logp(prompt: list[int], completion: list[int]) -> float:
        ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
        return float(completion_logp(model, ids, len(prompt)))

    parity: list[Json] = []
    with torch.no_grad():
        for pair in pairs[:PARITY_PAIRS]:
            prompt = cast(list[int], pair["prompt_ids"])
            for name in ("chosen_ids", "rejected_ids"):
                completion = cast(list[int], pair[name])
                ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
                labels = ids.clone()
                labels[:, :len(prompt)] = -100
                loss = float(model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss)
                fused = -loss * len(completion)
                direct = logp(prompt, completion)
                parity.append({"pair_id": pair["pair_id"], "part": name, "completion_logp": direct, "fused_ce_logp": fused})
                require(abs(direct - fused) <= max(0.25, 0.01 * abs(fused)),
                        f"completion_logp disagrees with the fused loss: {direct} vs {fused}")
        with logps_path.open("a") as stream:
            for index in range(len(done), len(pairs)):
                if should_stop():
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({"status": "deferred", "done": index, "pairs": len(pairs)}))
                    return EXIT_DONE
                pair = validate_pair(pairs[index])
                prompt = cast(list[int], pair["prompt_ids"])
                chosen = cast(list[int], pair["chosen_ids"])
                rejected = cast(list[int], pair["rejected_ids"])
                row = {"pair_id": pair["pair_id"], "chosen_logp": logp(prompt, chosen), "rejected_logp": logp(prompt, rejected),
                       "chosen_tokens": len(chosen), "rejected_tokens": len(rejected)}
                require(all(math.isfinite(cast(float, row[k])) and cast(float, row[k]) <= 0 for k in ("chosen_logp", "rejected_logp")),
                        f"invalid reference log-probability for {pair['pair_id']}")
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                if (index + 1) % FLUSH_EVERY == 0:
                    stream.flush()
                    os.fsync(stream.fileno())
            stream.flush()
            os.fsync(stream.fileno())
    rows = verified_prefix(logps_path, pair_ids)
    require(len(rows) == len(pairs), "reference rows missing")
    manifest = {"schema": REFERENCE_SCHEMA, "status": "complete", "recipe_sha256": sha256(recipe_path),
                "parent_path": parent_path, "parent_weights_sha256": cast(Json, parent["files"])["model.safetensors"],
                "pairs_sha256": pairs_record["sha256"], "pairs": len(pairs), "logps": str(logps_path), "logps_sha256": sha256(logps_path),
                "dtype": "bfloat16", "mode": "unsloth full-weight training mode, gradient checkpointing, torch.no_grad",
                "parity_check": parity, "tokenizer_contract": audit, "seconds_this_attempt": time.monotonic() - started,
                "final_set_access": False}
    tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest_path)
    print(json.dumps({"status": "complete", "pairs": len(pairs), "logps_sha256": manifest["logps_sha256"]}))
    return EXIT_DONE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.recipe)


if __name__ == "__main__":
    raise SystemExit(main())
