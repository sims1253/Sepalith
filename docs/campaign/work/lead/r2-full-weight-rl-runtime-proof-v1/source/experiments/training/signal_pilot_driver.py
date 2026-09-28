#!/usr/bin/env python3
"""Inference-only RL-11 TRAIN signal pilot driver.

The live path loads one root-bound merged model, generates four candidates for
each of 112 fixed TRAIN prompts, scores them with the pinned reward API, and
publishes only complete groups. It creates no optimizer or adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
PROTOCOL = SOURCE / "packages" / "sepalith" / "src"
if str(PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PROTOCOL))

from campaign_r_parse_probe import ParseProbeError, RParseOnlyProbe
from campaign_rl_buffer import RewardBufferIndex
from campaign_rl_train import CampaignPRM03Reward
from campaign_tokenizer_contract import (
    EXPECTED_TOKENIZER_IDENTITY,
    TokenizerContractError,
    load_pinned_reference_tokenizer,
    restore_pinned_tokenizer_contract,
)
from sepalith.campaign_protocol import BOS_ID, EOS_ID, NATIVE_EOG_IDS, VOCAB_SIZE
from signal_pilot_harness import (
    PilotError, canonical, commit_group, load_spec, record_infrastructure_failure,
    require, sha256, summarize, validate_committed,
)


class DriverError(PilotError):
    pass


def trim_generated_sequence(sequence: Sequence[int], prompt_length: int, *, completion_max_tokens: int) -> dict[str, Any]:
    values = list(sequence)
    require(type(prompt_length) is int and prompt_length >= 1, "prompt length invalid")
    require(len(values) >= prompt_length, "generation is shorter than its exact prompt")
    tail = values[prompt_length:]
    require(len(tail) <= completion_max_tokens, "generation exceeded completion cap")
    terminal = next((index for index, token in enumerate(tail) if token in NATIVE_EOG_IDS), None)
    generated = tail if terminal is None else tail[:terminal + 1]
    require(all(type(token) is int and 0 <= token < VOCAB_SIZE for token in generated), "generation token invalid")
    return {
        "generated_tokens": generated,
        "terminal_reason": "eos" if terminal is not None else "length" if len(tail) == completion_max_tokens else "unknown",
        "padded_after_terminal": 0 if terminal is None else len(tail) - len(generated),
    }


def _ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("ascii")).hexdigest()


def _safe_relative(value: Any) -> Path:
    require(isinstance(value, str) and value, "artifact path missing")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "artifact path escapes model snapshot")
    return path


def verify_source(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash every file in the driver closure against its source manifest."""
    manifest_path = Path(str(binding.get("manifest_path", "")))
    require(manifest_path.is_file(), "source manifest missing")
    require(sha256(manifest_path) == binding.get("manifest_sha256"), "source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "sepalith.rl11.signal-driver-source.v1", "source manifest schema mismatch")
    root = manifest_path.parent
    for entry in manifest.get("files", []):
        rel = _safe_relative(entry.get("path"))
        path = root / rel
        require(path.is_file(), f"source file missing:{rel}")
        require(path.stat().st_size == entry.get("bytes") and sha256(path) == entry.get("sha256"), f"source file differs:{rel}")
    require(len(manifest.get("files", [])) >= 10, "source closure incomplete")
    return manifest


def verify_model(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Bind actual manifest, config, tokenizer and all declared weight bytes."""
    required = {
        "manifest_path", "manifest_sha256", "model_path", "artifacts",
        "merged_weights_sha256", "tokenizer_json_sha256",
    }
    require(set(binding) == required, "model binding fields mismatch")
    manifest_path = Path(str(binding["manifest_path"]))
    model_path = Path(str(binding["model_path"]))
    require(manifest_path.is_file() and model_path.is_dir(), "model snapshot path missing")
    require(sha256(manifest_path) == binding["manifest_sha256"], "model manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") in {"complete", "accepted", "prepared_complete"}, "model manifest is not complete")
    artifacts = binding["artifacts"]
    require(isinstance(artifacts, list) and artifacts, "model artifact inventory missing")
    seen: set[str] = set()
    weights: list[dict[str, Any]] = []
    required_names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
    for entry in artifacts:
        require(isinstance(entry, Mapping), "model artifact entry invalid")
        rel = _safe_relative(entry.get("path"))
        key = rel.as_posix()
        require(key not in seen, "duplicate model artifact")
        seen.add(key)
        path = model_path / rel
        require(path.is_file(), f"model artifact missing:{key}")
        require(path.stat().st_size == entry.get("bytes") and sha256(path) == entry.get("sha256"), f"model artifact differs:{key}")
        if key.endswith((".safetensors", ".bin")):
            weights.append({"path": key, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    actual_files = {path.relative_to(model_path).as_posix() for path in model_path.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(model_path).parts)}
    require(seen == actual_files, "model artifact inventory is not exhaustive")
    require(required_names.issubset(seen), "required model config/tokenizer artifacts absent")
    require(weights, "model weight inventory empty")
    aggregate = hashlib.sha256(canonical(sorted(weights, key=lambda row: row["path"]))).hexdigest()
    require(aggregate == binding["merged_weights_sha256"], "merged weight aggregate mismatch")
    tokenizer = next(entry for entry in artifacts if entry["path"] == "tokenizer.json")
    require(tokenizer["sha256"] == binding["tokenizer_json_sha256"], "tokenizer binding mismatch")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    require(int(config.get("vocab_size", -1)) == VOCAB_SIZE, "model vocabulary differs")
    identity = {
        "manifest_sha256": binding["manifest_sha256"],
        "merged_weights_sha256": aggregate,
        "tokenizer_json_sha256": binding["tokenizer_json_sha256"],
        "model_path": str(model_path.resolve()),
    }
    identity["identity_sha256"] = hashlib.sha256(canonical(identity)).hexdigest()
    return identity


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_pilot_data(spec: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], RewardBufferIndex]:
    """Scan all eligible rows/contexts, then retain the fixed pilot selection."""
    inputs = spec["pool"]["inputs"]
    selected_manifest = json.loads(Path(inputs["selected_ids"]["path"]).read_text(encoding="utf-8"))
    require(selected_manifest.get("split") == "train", "selected ID manifest is not TRAIN")
    ordered_ids = selected_manifest.get("row_ids")
    require(isinstance(ordered_ids, list) and len(ordered_ids) == 15006 and len(set(ordered_ids)) == 15006, "eligible ID coverage differs")
    held = set(spec["pool"]["held_contradictory_ids"])
    require(not held.intersection(ordered_ids), "held contradiction entered eligible IDs")
    selected = {row["row_id"] for row in groups}
    rows: dict[str, tuple[int, dict[str, Any]]] = {}
    all_row_order: list[str] = []
    with Path(inputs["rows"]["path"]).open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            row_id = row.get("id")
            require(isinstance(row_id, str) and row_id not in rows and row_id not in all_row_order, "row ID invalid or duplicate")
            all_row_order.append(row_id)
            if row_id in selected:
                rows[row_id] = (position, row)
    require(all_row_order == ordered_ids, "row file coverage/order differs from eligible IDs")
    contexts: dict[str, dict[str, Any]] = {}
    context_order: list[str] = []
    with Path(inputs["context"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row_id = record.get("row_id")
            require(isinstance(row_id, str) and row_id not in context_order, "context ID invalid or duplicate")
            context_order.append(row_id)
            if row_id in selected:
                contexts[row_id] = record
    require(context_order == ordered_ids, "context coverage/order differs from eligible IDs")
    buffer_index = RewardBufferIndex.load(
        Path(inputs["buffer_manifest"]["path"]), inputs["buffer_manifest"]["sha256"], ordered_ids,
    )
    prepared: list[dict[str, Any]] = []
    repaired_mode_overlays = 0
    for group in groups:
        row_id = group["row_id"]
        require(row_id in rows and row_id in contexts, "pilot row/context missing")
        position, row = rows[row_id]
        context = contexts[row_id]
        require(position == group["pool_position"], "pilot pool position differs")
        require(row.get("family") == group["family"] == context.get("family"), "pilot family differs")
        require(row.get("package_id") == group["package_id"] == context.get("package_id"), "pilot package differs")
        require(str(context.get("split", "")).lower() == "train" and context.get("context_has_target_or_reward_keys") is False, "context is not leakage-free TRAIN")
        input_ids = row.get("input_ids")
        start = row.get("target_start")
        require(isinstance(input_ids, list) and type(start) is int and 1 <= start < len(input_ids), "stored prompt geometry invalid")
        require(all(type(token) is int and 0 <= token < VOCAB_SIZE for token in input_ids), "stored token invalid")
        prompt = input_ids[:start]
        require(prompt[0] == BOS_ID and len(prompt) == group["prompt_tokens"], "stored prompt length/BOS differs")
        require(_ids_hash(prompt) == group["prompt_ids_sha256"], "stored prompt identity differs")
        require(len(prompt) <= 3072 and len(prompt) + 1024 <= 4096, "pilot prompt exceeds context geometry")
        require(len(input_ids) - start == group["target_tokens"] and group["target_tokens"] <= 1024, "pilot full target cap/truncation differs")
        require(input_ids[-1] == EOS_ID and row.get("target_token_count") + 1 == group["target_tokens"], "pilot target terminal EOS differs")
        require(row.get("target_truncated") in (None, False), "pilot target is truncated")
        envelope = buffer_index.envelope_for(row_id)
        if envelope["syntax_evidence_mode"] != group["syntax_evidence_mode"]:
            require(
                group["family"] == "finish_block"
                and group["syntax_evidence_mode"] == "framed_fragment"
                and envelope["syntax_evidence_mode"] == "completion_prefix"
                and spec.get("repair_binding") == {
                    "changed_finish_rows": 3503, "diagnostic_suffix_for_repaired_finish": "",
                    "framed_fragment_rows_for_repaired_finish": 0,
                    "reward_buffer_sha256": "dca94ce5c24e2c3842940b23ab6f4be6f967f8e83f99276b150f107703d0c22d",
                    "rows_sha256": "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e",
                }, "unapproved pilot syntax-mode difference",
            )
            group["selection_syntax_evidence_mode"] = group["syntax_evidence_mode"]
            group["syntax_evidence_mode"] = envelope["syntax_evidence_mode"]
            repaired_mode_overlays += 1
        prepared.append({
            "group": dict(group), "prompt_ids": prompt, "context": context["context"],
            "prompt_text": row["prompt_text"],
            "target_operation": row["target_operation"], "target_body_text": row["target_body_text"],
            "family": row["family"], "package_id": row["package_id"], "reward_buffer": envelope,
        })
    require(len(prepared) == 112, "pilot group materialization differs")
    require(repaired_mode_overlays == 8, "repaired finish pilot overlay denominator differs")
    return prepared, buffer_index


def candidate_seed(selection_seed: str, group_index: int, candidate_index: int) -> int:
    raw = f"{selection_seed}\0{group_index}\0{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def normalize_generation(*, sequence: Sequence[int], prompt: Sequence[int], group: Mapping[str, Any], candidate_index: int, seed: int) -> dict[str, Any]:
    values = list(sequence)
    require(values[:len(prompt)] == list(prompt), "generation changed the exact stored prompt")
    trimmed = trim_generated_sequence(values, len(prompt), completion_max_tokens=1024)
    generated = trimmed["generated_tokens"]
    require(generated, "generator returned an empty candidate")
    record = {
        "row_id": group["row_id"], "group_index": group["group_index"],
        "candidate_index": candidate_index, "seed": seed,
        "prompt_ids_sha256": _ids_hash(prompt), "generated_ids": generated,
        "generated_ids_sha256": _ids_hash(generated), "generated_token_count": len(generated),
        "terminal_reason": trimmed["terminal_reason"],
        "padded_after_terminal": trimmed["padded_after_terminal"],
        "cap_hit": len(generated) == 1024 and generated[-1] not in {EOS_ID, 130073},
    }
    return record


def run_groups(*, spec: Mapping[str, Any], spec_sha: str, groups: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]], output: Path, model_identity: Mapping[str, Any], generator: Callable[[Sequence[int], int], Sequence[int]], rewarder: Callable[[Mapping[str, Any], Sequence[int]], tuple[float, Mapping[str, Any]]]) -> dict[str, Any]:
    committed = validate_committed(output, spec_sha256=spec_sha, model_identity=model_identity, groups=groups)
    for index in range(len(committed), len(groups)):
        item = prepared[index]
        group = item["group"]
        generations: list[dict[str, Any]] = []
        rewards: list[dict[str, Any]] = []
        for candidate_index in range(4):
            seed = candidate_seed(spec["selection"]["seed"], index, candidate_index)
            sequence = generator(item["prompt_ids"], seed)
            generation = normalize_generation(sequence=sequence, prompt=item["prompt_ids"], group=group, candidate_index=candidate_index, seed=seed)
            generations.append(generation)
            try:
                value, detail = rewarder(item, generation["generated_ids"])
            except ParseProbeError as error:
                record_infrastructure_failure(output, spec_sha256=spec_sha, model_identity=model_identity, group=group, candidate_index=candidate_index, error=error)
                raise
            reward = dict(detail)
            reward.update({
                "row_id": group["row_id"], "group_index": index,
                "candidate_index": candidate_index, "reward": float(value),
                "parser_infrastructure_failure": False,
                "protocol_failure": detail.get("failure", detail.get("protocol_failure")),
            })
            rewards.append(reward)
        commit_group(output, spec_sha256=spec_sha, model_identity=model_identity, group=group, generations=generations, rewards=rewards)
    final = validate_committed(output, spec_sha256=spec_sha, model_identity=model_identity, groups=groups)
    return summarize(final, groups)


def import_live_frameworks(importer=__import__):
    """Keep the required Unsloth-before-torch production import order testable."""
    unsloth = importer("unsloth", fromlist=("FastLanguageModel",))
    torch = importer("torch")
    return unsloth.FastLanguageModel, torch


def repair_loaded_tokenizer(model, tokenizer, model_path: Path,
                            prepared: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference = load_pinned_reference_tokenizer(model_path)
    prompt_rows = [{
        "id": item["group"]["row_id"], "prompt_text": item["prompt_text"],
        "input_ids": list(item["prompt_ids"]), "target_start": len(item["prompt_ids"]),
    } for item in prepared]
    try:
        audit = restore_pinned_tokenizer_contract(
            model, tokenizer, reference_tokenizer=reference, prompt_rows=prompt_rows,
        )
    except TokenizerContractError as error:
        raise DriverError(f"loaded tokenizer failed pinned contract: {error}") from error
    identity = (len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    require(identity == EXPECTED_TOKENIZER_IDENTITY, "loaded tokenizer identity differs after repair")
    require(audit.get("vocab_mapping_unchanged") is True and audit.get("added_tokens") is False,
            "tokenizer repair changed vocabulary")
    require(audit.get("prompt_parity", {}).get("checked_rows") == 112,
            "tokenizer prompt parity denominator differs")
    return audit


def build_live_components(config: Mapping[str, Any], model_path: Path,
                          buffer_index: RewardBufferIndex,
                          prepared: Sequence[Mapping[str, Any]]):
    """Defer CUDA/framework imports until all CPU identities pass."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}, "explicit CUDA lease is absent")
    FastLanguageModel, torch = import_live_frameworks()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "live pilot requires exactly one leased CUDA device")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path), max_seq_length=4096,
        dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=False,
    )
    tokenizer_audit = repair_loaded_tokenizer(model, tokenizer, model_path, prepared)
    FastLanguageModel.for_inference(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not any(parameter.requires_grad for parameter in model.parameters()), "pilot model has trainable parameters")

    def generate(prompt: Sequence[int], seed: int) -> list[int]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ids = torch.tensor([list(prompt)], dtype=torch.long, device="cuda:0")
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            result = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=1024,
                do_sample=True, temperature=0.7, top_p=0.95,
                repetition_penalty=1.0, eos_token_id=[EOS_ID, 130073],
                pad_token_id=EOS_ID, use_cache=True,
            )
        return [int(value) for value in result[0].detach().cpu().tolist()]

    parser = config["parser"]
    probe = RParseOnlyProbe(Path(parser["path"]), parser["sha256"], timeout_seconds=float(parser["timeout_seconds"]))
    scorer = CampaignPRM03Reward(tokenizer=tokenizer, max_completion_tokens=1024, reward_buffer_index=buffer_index, parse_probe=probe)

    def reward(item: Mapping[str, Any], generated: Sequence[int]) -> tuple[float, Mapping[str, Any]]:
        return scorer.score_one(
            item["context"], item["target_operation"], item["target_body_text"], generated,
            item["reward_buffer"], row_id=item["group"]["row_id"],
            family=item["family"], package_id=item["package_id"],
        )
    return generate, reward, tokenizer_audit


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(config.get("schema") == "sepalith.rl11.expanded-signal-driver-config.v1", "driver config schema mismatch")
    require(config.get("optimizer_updates") == 0, "signal pilot must not update an optimizer")
    require(config.get("geometry") == {"groups": 112, "candidates_per_group": 4, "prompt_max_tokens": 3072, "max_new_tokens": 1024, "context_max_tokens": 4096}, "driver geometry mismatch")
    require(config.get("training_data_binding") == {
        "eligible_rows": 15006,
        "rows_sha256": "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e",
        "reward_buffer_sha256": "dca94ce5c24e2c3842940b23ab6f4be6f967f8e83f99276b150f107703d0c22d",
        "pilot_groups": 112, "candidates": 448,
    }, "repaired TRAIN/reward binding differs")
    parser = config.get("parser")
    require(isinstance(parser, Mapping) and parser.get("generated_r_executed") is False, "parser policy differs")
    parser_path = Path(str(parser.get("path", "")))
    require(parser_path.is_file() and sha256(parser_path) == parser.get("sha256"), "parser identity mismatch")
    verify_source(config["source"])
    return config


def verify_root_admission(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove this runnable config came from one exact pending binding and root decision."""
    pending_record = config.get("binding_config")
    admission_record = config.get("root_admission")
    require(isinstance(pending_record, Mapping) and isinstance(admission_record, Mapping), "root admission binding absent")
    pending_path = Path(str(pending_record.get("path", "")))
    admission_path = Path(str(admission_record.get("path", "")))
    require(pending_path.is_file() and sha256(pending_path) == pending_record.get("sha256"), "pending model binding differs")
    require(admission_path.is_file() and sha256(admission_path) == admission_record.get("sha256"), "root admission differs")
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    reconstructed = dict(config)
    reconstructed["status"] = "root_binding_prepared_admission_required"
    reconstructed["binding_config"] = None
    reconstructed["root_admission"] = None
    require(canonical(reconstructed) == canonical(pending), "admitted config differs from exact pending binding")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    expected_model = config["model"]
    require(admission.get("schema") == "sepalith.rl11.expanded-signal-driver-root-admission.v1" and admission.get("status") == "admitted" and admission.get("launch_authorized") is True, "root signal admission missing")
    require(admission.get("binding_config_sha256") == pending_record["sha256"], "admission refers to another pending binding")
    require(admission.get("source_manifest_sha256") == config["source"]["manifest_sha256"] and admission.get("pilot_spec_sha256") == config["pilot_spec"]["sha256"], "admission source/spec identity differs")
    require(admission.get("model_manifest_sha256") == expected_model["manifest_sha256"] and admission.get("merged_weights_sha256") == expected_model["merged_weights_sha256"] and admission.get("tokenizer_json_sha256") == expected_model["tokenizer_json_sha256"], "admission model/tokenizer identity differs")
    require(admission.get("rows_sha256") == config["training_data_binding"]["rows_sha256"] and admission.get("reward_buffer_sha256") == config["training_data_binding"]["reward_buffer_sha256"], "admission TRAIN/reward identity differs")
    require(admission.get("output_path") == config["output_path"], "admission output path differs")
    require(admission.get("model_stage") == "selected_full_weight_edit_sft" and admission.get("train_only_rewards") is True and admission.get("dev_or_final_access") is False and admission.get("optimizer_updates") == 0, "admission scope differs")
    selected = admission.get("selected_model_evidence", {})
    selected_path = Path(str(selected.get("path", "")))
    require(selected_path.is_file() and sha256(selected_path) == selected.get("sha256"), "selected full-weight SFT evidence differs")
    return admission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    spec_path = Path(config["pilot_spec"]["path"])
    require(sha256(spec_path) == config["pilot_spec"]["sha256"], "pilot spec binding differs")
    spec, spec_sha, groups = load_spec(spec_path)
    prepared, buffer_index = load_pilot_data(spec, groups)
    if args.command == "preflight":
        admitted = config.get("status") == "root_admitted_signal_only"
        if admitted:
            verify_root_admission(config)
        print(json.dumps({"status": "pass", "eligible_rows_scanned": 15006, "pilot_groups": 112, "rollouts": 448, "model_bound": config.get("model") is not None, "root_admitted": admitted, "runnable": admitted}, sort_keys=True))
        return 0
    require(config.get("model") is not None, "root model binding is absent")
    require(config.get("status") == "root_admitted_signal_only", "root signal admission is absent")
    verify_root_admission(config)
    output = Path(str(config.get("output_path", "")))
    require(str(output) and output.is_absolute(), "root output path must be absolute")
    identity = verify_model(config["model"])
    generator, rewarder, tokenizer_audit = build_live_components(
        config, Path(config["model"]["model_path"]), buffer_index, prepared,
    )
    identity = dict(identity)
    identity["tokenizer_runtime_contract"] = tokenizer_audit["contract"]
    identity["tokenizer_prompt_parity_rows"] = tokenizer_audit["prompt_parity"]["checked_rows"]
    identity["identity_sha256"] = hashlib.sha256(canonical(identity)).hexdigest()
    summary = run_groups(spec=spec, spec_sha=spec_sha, groups=groups, prepared=prepared, output=output, model_identity=identity, generator=generator, rewarder=rewarder)
    summary_path = output / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp = summary_path.with_suffix(".tmp")
    temp.write_bytes(canonical(summary) + b"\n")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, summary_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
