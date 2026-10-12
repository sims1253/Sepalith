"""Frozen development-panel reads inside the single training resource owner.

These measurements separate exact region agreement, protocol validity and
strict no-op behavior. Protocol validity is not R syntax or semantic validity.
The final evaluation set is not an accepted input to this evaluator.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from sepalith.training.checkpoint.campaign_checkpoint import write_json


# bf16 is the authoritative PC dtype; fp16 and fp32 exist for Kaggle T4s,
# which have no bf16.  Only bf16 results may bind a training gate.
GENERATION_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
DEFAULT_GENERATION_DTYPE = "bf16"

# Batched greedy generation (C07 follow-up).  Batch size 1 with the dynamic
# cache is the original per-case path, unchanged.  Anything else batches cases
# in a fixed order that depends only on the panel (prompt token counts and
# case ids), never on the checkpoint, so the baseline and every gate put the
# same cases in the same batches with the same padding.
GENERATION_BATCH_ORDERING = "prompt_tokens_descending_then_case_id_ascending"
GENERATION_PADDING_SIDE = "left"
GENERATION_PAD_TOKEN_ID = 1
GENERATION_BOS_TOKEN_ID = 0
GENERATION_KV_CACHES = ("dynamic", "static")
DEFAULT_GENERATION_KV_CACHE = "dynamic"
MAX_GENERATION_BATCH_SIZE = 256


def generation_batching(batch_size=1, kv_cache=DEFAULT_GENERATION_KV_CACHE):
    """Return the ``generation_runtime`` batching record, or None for the original per-case path."""
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_GENERATION_BATCH_SIZE:
        raise ValueError(f"Development generation batch size must be an integer in [1, {MAX_GENERATION_BATCH_SIZE}]")
    if kv_cache not in GENERATION_KV_CACHES:
        raise ValueError(f"Unsupported development generation KV cache: {kv_cache!r}")
    if batch_size == 1 and kv_cache == DEFAULT_GENERATION_KV_CACHE:
        return None
    return {"batch_size": batch_size, "ordering": GENERATION_BATCH_ORDERING, "padding_side": GENERATION_PADDING_SIDE,
            "pad_token_id": GENERATION_PAD_TOKEN_ID, "kv_cache": kv_cache,
            "teacher_forced_forward": "per_case_unpadded_use_cache_false",
            "generation_seconds": "batch_wall_seconds_split_evenly_across_its_cases"}


def generation_batches(cases, batch_size):
    """Split ``cases`` (``(case_id, prompt_tokens)`` pairs) into batches of indexes.

    Longest prompts first, so the most memory-hungry batch runs first; ties are
    broken by case id.  The result depends only on the panel.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("A positive generation batch size is required")
    ids = [case_id for case_id, _ in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Generation batching needs unique case ids")
    order = sorted(range(len(cases)), key=lambda index: (-cases[index][1], cases[index][0]))
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def truncate_at_stop(generated, stop_ids, cap):
    """Cut one batched row after its first stop token, as batch-1 ``generate`` stops there.

    Rows that finish early are filled with the pad id (1, itself a stop id)
    while the rest of the batch runs on; the cut removes that fill.  A row that
    never stops ran to the cap.
    """
    for index, token in enumerate(generated):
        if token in stop_ids:
            return generated[:index + 1]
    if len(generated) != cap:
        raise ValueError("A batched row ended without a stop token before the generation cap")
    return generated


def greedy_generate_kwargs(cap, stop_ids):
    """The greedy ``generate`` arguments every DEV path shares (RULES.md metrics)."""
    return {"max_new_tokens": cap, "do_sample": False, "num_beams": 1, "eos_token_id": list(stop_ids),
            "pad_token_id": GENERATION_PAD_TOKEN_ID, "bos_token_id": GENERATION_BOS_TOKEN_ID, "repetition_penalty": 1.0,
            "return_dict_in_generate": False, "use_cache": True}


def batched_greedy_generate(model, prompts, *, cap, stop_ids, kv_cache=DEFAULT_GENERATION_KV_CACHE, device=None):
    """Greedy-generate a batch of prompts (lists of token ids); return each row's generated ids.

    Prompts are left-padded with the pad id and masked out; ``generate`` derives
    the position ids from the mask, so each row sees the positions it would see
    alone.  Each row stops at its first stop id; finished rows stay in the batch
    as padding and do not change the other rows' inputs.
    """
    import torch

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Batched generation needs non-empty prompts")
    if kv_cache not in GENERATION_KV_CACHES:
        raise ValueError(f"Unsupported development generation KV cache: {kv_cache!r}")
    device = next(model.parameters()).device if device is None else device
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), width), GENERATION_PAD_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention_mask[row, width - len(prompt):] = 1
    extra = {} if kv_cache == DEFAULT_GENERATION_KV_CACHE else {"cache_implementation": kv_cache}
    sequence = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                              **greedy_generate_kwargs(cap, stop_ids), **extra)
    if sequence.shape[0] != len(prompts) or not 0 < sequence.shape[1] - width <= cap:
        raise ValueError("Batched generation returned an unexpected shape")
    if not torch.equal(sequence[:, :width], input_ids):
        raise ValueError("Batched generation changed the prompts")
    return [truncate_at_stop(row, stop_ids, cap) for row in sequence[:, width:].tolist()]


def generation_dtype(name):
    """Return the torch dtype for a ``GENERATION_DTYPES`` key."""
    if name not in GENERATION_DTYPES:
        raise ValueError(f"Unsupported development generation dtype: {name!r}")
    import torch

    return getattr(torch, GENERATION_DTYPES[name])


def pin_cuda_device(index, environ=None):
    """Expose exactly one CUDA device to this process and return its id.

    ``index`` selects among the devices already visible: the entries of an
    existing ``CUDA_VISIBLE_DEVICES`` lease, or the physical devices when the
    variable is unset (Kaggle).  This must run before CUDA initializes.
    """
    environ = os.environ if environ is None else environ
    if type(index) is not int or index < 0:
        raise ValueError("CUDA device index must be a non-negative integer")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise ValueError("CUDA is already initialized; pin the device before any CUDA use")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        chosen = str(index)
    else:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if not entries:
            raise ValueError("CUDA_VISIBLE_DEVICES hides every device; refusing to pin one")
        if index >= len(entries):
            raise ValueError(f"CUDA device index {index} is outside the visible devices {visible!r}")
        chosen = entries[index]
    environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


def load_development_model(path, *, dtype=DEFAULT_GENERATION_DTYPE, device="cuda"):
    """Load a dense checkpoint for DEV generation on one device.

    With the defaults this is exactly the original PC call: bf16 weights on
    the single leased CUDA device.
    """
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=generation_dtype(dtype), device_map={"": device},
        local_files_only=True, trust_remote_code=False,
    )


def verified_file(record):
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Expected an absolute development input file: {path}")
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != record["sha256"]:
        raise ValueError(f"Development input hash differs: {path}")
    return path


@contextmanager
def case_evaluation_guard(model):
    """Restore RNG and Unsloth training state after one development case.

    ``checkpoint_callback`` still guards the whole panel.  This narrower guard
    is required because pinned Unsloth generation restores itself only when
    ``generate`` starts with ``model.training`` true.  The callback enters the
    panel in eval mode, so each generated case must explicitly call the
    model's supported ``for_training`` hook before the next teacher-forced
    forward.  The model is returned to eval mode for the remainder of panel
    scoring.  CPU fixture models without that hook use their ordinary train
    mode as a compatibility fallback.
    """
    import random
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    was_training = getattr(model, "training", None)
    modules = getattr(model, "modules", None)
    checkpointing = []
    if callable(modules):
        checkpointing = [module.gradient_checkpointing for module in modules()
                         if hasattr(module, "gradient_checkpointing")]
    # Keep the evaluator's CPU fixtures free of the training stack at module
    # import, while sharing the exact runtime identity capture with SFT.
    from sepalith.training.eval.dev_model_state import capture_runtime_contract, clear_generation_markers, restore_runtime_contract

    configs, tokenizers = capture_runtime_contract(model)
    restore_training = getattr(model, "for_training", None)
    try:
        yield
    finally:
        try:
            # This also handles PEFT's forwarded marker before the supported
            # Unsloth hook runs.
            clear_generation_markers(model)
            if callable(restore_training):
                mode = next((value for value in checkpointing if value), False)
                try:
                    restore_training(use_gradient_checkpointing=mode)
                except TypeError:
                    # Keep compatibility with a bound test/future hook whose
                    # supported signature has no gradient-checkpoint argument.
                    restore_training()
                evaluate = getattr(model, "eval", None)
                if callable(evaluate):
                    evaluate()
            elif was_training is not None:
                train = getattr(model, "train", None)
                if callable(train):
                    train(was_training)
        finally:
            # The supported hook normally restores these itself, but keep the
            # before-case contract authoritative if cleanup raises.  The
            # exception remains visible to the caller; this is restoration,
            # not error suppression.
            restore_runtime_contract(configs, tokenizers)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def loss_sums(logits, ids, target_start):
    """Sum next-token NLL with disjoint prompt/target denominators, including EOS."""
    import torch

    if logits.ndim != 3 or logits.shape[:2] != ids.shape or ids.shape[0] != 1:
        raise ValueError("Development NLL expects one complete unpadded sequence")
    if not 1 < target_start < ids.shape[1]:
        raise ValueError("Invalid development prompt/target boundary")
    sums = {"prompt_nll_sum": 0.0, "target_nll_sum": 0.0,
            "prompt_tokens": target_start - 1, "target_tokens": ids.shape[1] - target_start}
    # Keep float32 conversion bounded for the 130560-token vocabulary.
    for first in range(1, ids.shape[1], 128):
        last = min(first + 128, ids.shape[1])
        nll = torch.nn.functional.cross_entropy(logits[0, first - 1:last - 1].float(),
                                               ids[0, first:last], reduction="none")
        boundary = max(0, min(last, target_start) - first)
        sums["prompt_nll_sum"] += nll[:boundary].sum().item()
        sums["target_nll_sum"] += nll[boundary:].sum().item()
    return sums


def classify(raw, context, expected, generated_ids):
    from sepalith.campaign_protocol import parse_output, valid_generation_tokens

    result = parse_output(raw, context)
    valid = valid_generation_tokens(generated_ids) and result.status == "accepted"
    noop = valid and result.operation == "no_op"
    actual = list(context.region_old) if noop else list(result.body)
    return {
        "protocol_valid": valid,
        "exact_region": valid and actual == expected,
        "predicted_noop": noop,
        "suggestion": valid and not noop,
        "failure": None if valid else (result.reason or "noncanonical_or_missing_EOS"),
    }


def validate_development_capacity(row, context_capacity, cap, case_id):
    """Keep complete references for scoring even when generation is capped."""
    if (len(row["input_ids"]) > context_capacity
            or row["target_start"] + cap > context_capacity):
        raise ValueError(f"Development case exceeds the admitted context capacity: {case_id}")
    return len(row["input_ids"]) - row["target_start"]


def generation_termination(generated_ids, cap, stop_ids):
    stop_token_id = generated_ids[-1] if generated_ids and generated_ids[-1] in stop_ids else None
    return {"response_complete": stop_token_id is not None, "stop_token_id": stop_token_id,
            "cap_hit": len(generated_ids) == cap and stop_token_id is None}


def development_evaluator(recipe):
    """Factory for campaign_sft; data/renderer identities are fixed before load."""
    from sepalith.campaign_protocol import PromptContext, RENDERER_ID

    panel = verified_file(recipe["development_panel"])
    if recipe["renderer_id"] != RENDERER_ID:
        raise ValueError("Development evaluator renderer differs from the admitted recipe")
    cases, identities = [], set()
    with panel.open() as stream:
        for line in stream:
            case = json.loads(line)
            if case["split"] != "dev":
                raise ValueError("Only development records may enter the checkpoint evaluator")
            if not case["id"] or case["id"] in identities or not case.get("package_id") or not case.get("family"):
                raise ValueError("Development cases require unique identities and package/family provenance")
            identities.add(case["id"])
            context = PromptContext.from_mapping(case["context"])
            cases.append((case, context))
    if not cases or sorted(identities) != sorted(recipe["development_case_ids"]):
        raise ValueError("Development denominator differs from the predeclared panel")
    cap = recipe["development_max_new_tokens"]
    if type(cap) is not int or cap < 1:
        raise ValueError("A positive development generation cap is required")
    # Absent keys select the original per-case path.
    batching = generation_batching(recipe.get("generation_batch_size", 1),
                                   recipe.get("generation_kv_cache", DEFAULT_GENERATION_KV_CACHE))

    def evaluate(model, tokenizer, checkpoint, step):
        import torch
        from sepalith.campaign_protocol import build_training_row, NATIVE_EOG_IDS

        prepared = []
        for case, context in cases:
            row = build_training_row(context, operation=case["operation"], region_new=case["region_new"],
                                     tokenizer=tokenizer, row_id=case["id"], family=case["family"],
                                     package_id=case["package_id"], split="dev")
            validate_development_capacity(
                row, recipe["parameters"]["max_sequence_tokens"], cap, case["id"]
            )
            prepared.append((case, context, row))
        device = next(model.parameters()).device
        destination = Path(recipe["evaluation_output_directory"]) / "cases.json"

        def case_record(case, context, row, nll, generated, generation_seconds):
            termination = generation_termination(generated, cap, NATIVE_EOG_IDS)
            response_complete = termination["response_complete"]
            raw_ids = generated[:-1] if response_complete else generated
            raw = tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            expected_noop = list(context.region_old) == case["region_new"]
            outcome = classify(raw, context, case["region_new"], generated)
            return {
                "id": case["id"], "package_id": case["package_id"], "family": case["family"],
                "expected_noop": expected_noop, "strata": case.get("strata", {}),
                "prompt_tokens": row["target_start"], "generated_tokens": len(generated),
                "reference_tokens": len(row["input_ids"]) - row["target_start"],
                "reference_exceeds_generation_cap": len(row["input_ids"]) - row["target_start"] > cap,
                **termination,
                "generation_seconds": generation_seconds, "raw_output": raw, "generated_ids": generated,
                "loss": nll, **outcome,
            }

        def teacher_forced(row):
            full = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
            output = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
            nll = loss_sums(output.logits, full, row["target_start"])
            del output
            return full, nll

        if batching is None:
            results = []
            for case, context, row in prepared:
                with case_evaluation_guard(model):
                    with torch.no_grad():
                        full, nll = teacher_forced(row)
                        prompt = full[:, :row["target_start"]]
                        started = time.monotonic()
                        sequence = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                                  **greedy_generate_kwargs(cap, NATIVE_EOG_IDS))
                        generated = sequence[0, prompt.shape[1]:].tolist()
                        generation_seconds = time.monotonic() - started
                results.append(case_record(case, context, row, nll, generated, generation_seconds))
                # Preserve intermediate evidence if a later case fails or the watchdog stops the attempt.
                write_json(destination, {"status": "partial", "step": step, "results": results,
                                         "expected_case_ids": sorted(identities)})
        else:
            done = {}
            for indexes in generation_batches([(case["id"], row["target_start"]) for case, _, row in prepared],
                                              batching["batch_size"]):
                # One guard per batch: the teacher-forced forwards run before the
                # batch's generate, and the guard restores state after it.
                with case_evaluation_guard(model):
                    with torch.no_grad():
                        losses = [teacher_forced(prepared[index][2])[1] for index in indexes]
                        started = time.monotonic()
                        generated = batched_greedy_generate(
                            model, [prepared[index][2]["input_ids"][:prepared[index][2]["target_start"]] for index in indexes],
                            cap=cap, stop_ids=NATIVE_EOG_IDS, kv_cache=batching["kv_cache"], device=device)
                        generation_seconds = (time.monotonic() - started) / len(indexes)
                for index, nll, ids in zip(indexes, losses, generated, strict=True):
                    case, context, row = prepared[index]
                    done[index] = case_record(case, context, row, nll, ids, generation_seconds)
                # Partial evidence after each batch, in panel order.
                write_json(destination, {"status": "partial", "step": step, "results": [done[i] for i in sorted(done)],
                                         "expected_case_ids": sorted(identities), "generation_runtime": batching})
            if sorted(done) != list(range(len(prepared))):
                raise ValueError("Batched generation did not cover every development case exactly once")
            results = [done[index] for index in range(len(prepared))]
        denominators = {"cases": len(results), "complete_cases": sum(r["response_complete"] for r in results),
                        "incomplete_cases": sum(not r["response_complete"] for r in results),
                        "packages": len({r["package_id"] for r in results}),
                        "strict_noop": sum(r["expected_noop"] for r in results),
                        "edits": sum(not r["expected_noop"] for r in results),
                        "prompt_loss_tokens": sum(r["loss"]["prompt_tokens"] for r in results),
                        "target_loss_tokens": sum(r["loss"]["target_tokens"] for r in results)}
        counts = {name: sum(r[name] for r in results)
                  for name in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit")}
        counts.update(strict_noop_correct=sum(r["expected_noop"] and r["predicted_noop"] for r in results),
                      strict_noop_false_suggestions=sum(r["expected_noop"] and r["suggestion"] for r in results),
                      edit_exact=sum(not r["expected_noop"] and r["exact_region"] for r in results))
        complete = [r for r in results if r["response_complete"]]
        complete_case_counts = {"protocol_valid": sum(r["protocol_valid"] for r in complete),
                                "exact_region": sum(r["exact_region"] for r in complete),
                                "edit_exact": sum(not r["expected_noop"] and r["exact_region"] for r in complete),
                                "strict_noop_correct": sum(r["expected_noop"] and r["predicted_noop"] for r in complete),
                                "strict_noop_false_suggestions": sum(r["expected_noop"] and r["suggestion"] for r in complete)}
        summary = {"case_ids": [r["id"] for r in results], "denominators": denominators, "counts": counts,
                   "complete_case_counts": complete_case_counts, "generation_max_new_tokens": cap,
                   "prompt_nll": sum(r["loss"]["prompt_nll_sum"] for r in results) / denominators["prompt_loss_tokens"],
                   "target_nll": sum(r["loss"]["target_nll_sum"] for r in results) / denominators["target_loss_tokens"],
                   "panel_sha256": recipe["development_panel"]["sha256"], "cases_path": str(destination),
                   "interpretation": "Development exact/protocol/no-op diagnostic; no R semantic-validity or final-release claim."}
        if batching is not None:
            summary["generation_runtime"] = dict(batching)
        write_json(destination, {"status": "complete", "step": step, "summary": summary, "results": results})
        return summary

    return evaluate
