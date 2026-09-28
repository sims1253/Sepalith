"""Root-runtime pre-update gate. Importing this file performs no model work."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
import time


def require(condition, reason):
    if not condition:
        raise ValueError(f"target_only_startup_gate: {reason}")


def verify_batch(batch, rows):
    import torch
    require(set(batch) == {"input_ids", "attention_mask", "labels"}, "unexpected_model_inputs")
    ids, attention, labels = (batch[k] for k in ("input_ids", "attention_mask", "labels"))
    require(ids.ndim == 2 and ids.shape == labels.shape == attention.shape and ids.shape[0] == len(rows), "batch_shape")
    require(ids.dtype == labels.dtype == torch.long, "token_dtype")
    total = 0
    for i, row in enumerate(rows):
        start, end = row["target_start"], len(row["input_ids"])
        require(end <= ids.shape[1] and 1 < start < end - 1, "boundary_or_truncation")
        require(row["input_ids"][start:] == row["target_body_tokens"] + row["target_terminal_tokens"] + [1], "target_metadata_lost")
        require(ids[i, :end].tolist() == row["input_ids"], "actual_dataloader_changed_tokens_or_order")
        require(bool(attention[i, :end].eq(1).all() and attention[i, end:].eq(0).all()), "attention_or_padding")
        require(bool(labels[i, :start].eq(-100).all() and labels[i, end:].eq(-100).all()), "prompt_or_padding_supervised")
        require(labels[i, start:end].tolist() == row["input_ids"][start:], "target_or_eos_not_supervised")
        require(bool(ids[i, end:].eq(1).all()), "padding_token_changed")
        total += end - start
    require(int(labels[:, 1:].ne(-100).sum()) == total, "causal_denominator")
    return total


@contextmanager
def observe_fused_call(expected_labels, expected_denominator):
    """Witness the installed llama forward's actual call, then restore its symbol."""
    import torch
    import unsloth.models.llama as llama
    original = llama.unsloth_fused_ce_loss
    calls = []
    def observed(*args, **kwargs):
        require(not args, "unreviewed_fused_positional_signature")
        require(torch.equal(kwargs.get("labels"), expected_labels), "fused_labels_changed")
        count = kwargs.get("n_items")
        require(count is not None and int(count) == expected_denominator, "fused_denominator_changed")
        require(kwargs.get("shift_labels", True) is True, "fused_shift_disabled")
        calls.append({"denominator": int(count), "shift_labels": True})
        return original(**kwargs)
    llama.unsloth_fused_ce_loss = observed
    try:
        yield calls
    finally:
        llama.unsloth_fused_ce_loss = original


def target_only_startup_gate(trainer, model, actual_dataloader, *, gate_clock_step,
                             admitted_rows, preserve_state, observe=observe_fused_call):
    """No backward or optimizer step. Errors propagate before the train loop."""
    import torch
    started = time.monotonic()
    require(gate_clock_step in (0, 25), "unexpected_start_step")
    require(trainer.args.remove_unused_columns is False, "metadata_columns_may_be_removed")
    require(trainer.args.dataloader_num_workers == 0, "unreviewed_worker_prefetch")
    require(trainer.args.gradient_accumulation_steps == 4, "accumulation_geometry")
    require(trainer.model_accepts_loss_kwargs is True, "model_would_drop_denominator")
    require(trainer.compute_loss_func is None and trainer.label_smoother is None, "alternate_objective")
    if gate_clock_step == 0:
        require(not trainer.optimizer.state, "fresh_pilot_has_restored_optimizer_state")
    require(os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1", "fused_path_disabled")
    # The live Trainer loader at train_begin starts from the frozen dataset.
    # Resume cursor skip happens later in Trainer; this probe is not a resume test.
    expected = deepcopy(admitted_rows)
    require(len(expected) == 16, "admitted_probe_rows_missing")
    for index, row in enumerate(expected):
        require(all(k in row for k in ("input_ids", "target_start", "target_body_tokens", "target_terminal_tokens")), "post_trainer_metadata_missing")
        require(dict(trainer.train_dataset[index]) == row, "post_trainer_row_differs_from_admission")
    versions = [(p, p._version, p.grad) for p in model.parameters()]
    require(all(grad is None for _, _, grad in versions), "preexisting_gradients")
    require(not any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()), "stochastic_dropout_in_reference")
    generators = {}
    for owner in (actual_dataloader, getattr(actual_dataloader, "sampler", None)):
        for name in ("generator", "synchronized_generator"):
            generator = getattr(owner, name, None)
            if isinstance(generator, torch.Generator): generators[id(generator)] = (generator, generator.get_state())
    configs = [(m, deepcopy(m.config.use_cache)) for m in model.modules() if hasattr(getattr(m, "config", None), "use_cache")]
    try:
        with preserve_state(model):
            # preserve_random_state() enters eval mode; explicitly restore the
            # same training-mode forward used by the forthcoming optimizer loop.
            model.train()
            for module, _ in configs: module.config.use_cache = False
            batches, denominator = trainer.get_batch_samples(iter(actual_dataloader), 4, trainer.args.device)
            require(len(batches) == 4, "missing_accumulation_batches")
            counts = [verify_batch(batch, expected[i * 4:(i + 1) * 4]) for i, batch in enumerate(batches)]
            require(denominator is not None and int(denominator) == sum(counts), "trainer_denominator_mismatch")
            batch = trainer._prepare_inputs(batches[0])
            count = int(denominator)
            # Only one real microbatch is forwarded; all four contribute to the
            # actual accumulation denominator exactly as in production.
            with torch.no_grad(), observe(batch["labels"], count) as calls:
                with trainer.compute_loss_context_manager():
                    fused, output = trainer.compute_loss(model, dict(batch), return_outputs=True, num_items_in_batch=denominator)
                require(len(calls) == 1, "actual_fused_entrypoint_not_witnessed")
                require(not torch.is_tensor(output.logits), "expected_unsloth_fused_empty_logits")
                require(fused.numel() == 1 and bool(torch.isfinite(fused)), "nonfinite_fused_loss")
                fused_value = float(fused)
            # Independent next-token reference; request only suffix logits for
            # one unpadded row at a time to bound peak vocabulary allocation.
            summed = 0.0
            with torch.no_grad():
                for i, row in enumerate(expected[:4]):
                    start, end = row["target_start"], len(row["input_ids"])
                    kwargs = {"input_ids": batch["input_ids"][i:i+1, :end],
                              "attention_mask": batch["attention_mask"][i:i+1, :end],
                              "use_cache": False, "num_logits_to_keep": end-start+1}
                    with trainer.compute_loss_context_manager(): reference = model(**kwargs)
                    logits = reference.logits
                    require(torch.is_tensor(logits) and logits.ndim == 3 and logits.shape[:2] == (1, end-start+1), "reference_suffix_logits_unavailable")
                    targets = batch["labels"][i, start:end]
                    for offset in range(0, len(targets), 32):
                        n = min(32, len(targets)-offset)
                        summed += float(torch.nn.functional.cross_entropy(logits[0, offset:offset+n].float(), targets[offset:offset+n], reduction="sum"))
                    del reference, logits
            expected_loss = summed / count
            require(torch.isfinite(torch.tensor(expected_loss)).item(), "nonfinite_reference_loss")
            tolerance = max(0.005, 0.01 * abs(expected_loss))
            require(abs(fused_value-expected_loss) <= tolerance, "fused_reference_loss_mismatch")
            for parameter, version, grad in versions:
                require(parameter._version == version and parameter.grad is grad, "gate_mutated_parameter_or_grad")
            result = {"schema": 1, "status": "pass", "before_optimizer_step": gate_clock_step,
                      "post_trainer_rows_checked": 16, "microbatches_checked": 4,
                      "target_denominator": count, "microbatch_target_denominators": counts,
                      "actual_fused_calls": len(calls), "fused_loss": fused_value,
                      "reference_target_nll_sum": summed, "reference_loss": expected_loss,
                      "absolute_tolerance": tolerance, "prompt_supervision_tokens": 0,
                      "padding_supervision_tokens": 0, "eos_supervised": True,
                      "probe_draw_offset": 0, "resume_equivalence_proven": False,
                      "backward_or_optimizer_step": False}
            result["elapsed_seconds"] = time.monotonic() - started
            result["batch_identity_sha256"] = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            return result
    finally:
        for module, value in configs: module.config.use_cache = value
        for generator, state in generators.values(): generator.set_state(state)
