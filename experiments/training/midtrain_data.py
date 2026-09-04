"""B8 midtrain instrument: completion-only loss masking + packing (pure data functions).

Landed 2026-09-05 by zcode-b8-patch as the B8 prerequisite (base-bakeoff plan
§2 B8; production-finetune plan §2 midtrain slot). The 2026-08-19 probe failed
on its instrument — loss over the FULL sequence (unmasked prompts) and no
packing (docs/research/2026-08-19-night-results.md ~L797). This module is the
fixed instrument's data side; train_sft.py wires it behind the MIDTRAIN_MASK=1
/ --midtrain opt-in. With the flag OFF, train_sft.py runs its legacy pipeline
verbatim (byte-compatibility contract; see test_midtrain_data.py).

Design decisions (evidence in scripts/run_b4_unsloth_safe.sh + the B4 saga):

1. COMPLETION-ONLY MASKING is computed from the dataset's own `prompt` field
   (astfim_v1 rows are constructed as text == prompt + target, verified on the
   real corpus): tokenize the FULL text exactly once (identical token stream to
   the legacy pipeline), tokenize the prompt, mask labels[-100] over the
   longest token-level common prefix. No unsloth `train_on_responses_only`
   (chat-template assumptions), no TRL prompt/completion separate
   tokenization (BPE seam artifacts at the prompt/target boundary).

2. PACKING — two variants:
   - "bucket" (DEFAULT, conservative): one sample per sequence row; padding
     waste minimized by length-grouped batching (transformers 5.5
     train_sampling_strategy="group_by_length" + this module's `length`
     column). Cross-sample isolation is exact BY CONSTRUCTION — identical
     isolation semantics to every banked run (2D padding mask is the only
     isolation mechanism this stack's GDN layers honor).
   - "seq" (sequence packing, full-attention bases ONLY): greedy
     first-fit-decreasing concat of whole samples into <=block_size rows,
     position_ids restarting at 0 per sample, and a 4D block-diagonal causal
     attention mask. Exact attention/position isolation for full-attention
     models. REFUSED for linear-attention (GDN/FLA) model types: the
     Qwen3.5 GDN layer's model-level forward passes NO per-sample boundary
     metadata to the delta-rule kernel (transformers 5.5
     modeling_qwen3_5.py, chunk_gated_delta_rule call site), so sequence-dim
     packing would bleed recurrent state across samples in 3 of 4 layers
     whatever mask/position_ids we build — and unsloth's padding-free
     (varlen) packing is a named crash suspect on this box (run_b4_unsloth_safe.sh
     header: "padding-free packing off (GDN-state suspect #2)").

All builders are pure list/tensor functions taking a `tokenize` callable so
they unit-test CPU-side without unsloth/trl/GPU.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch

IGNORE_INDEX = -100  # torch/trl CE ignore token (labels convention)

# Model types whose attention layers are linear/recurrent (gated-delta-net and
# friends) and therefore cannot honor cross-sample isolation for sequence-dim
# packing (no cu_seqlens/seq_idx plumbing from the model forward). Mirrors
# unsloth/models/loader.py FLA_MODEL_TYPE_PREFIXES.
LINEAR_ATTENTION_MODEL_TYPES = ("qwen3_5", "qwen3_next", "kimi_linear", "olmo_hybrid", "fla", "lfm2")

VALID_PACK_MODES = ("bucket", "seq")


# ---------------------------------------------------------------------------
# flag parsing
# ---------------------------------------------------------------------------

def parse_midtrain_flag(argv: Optional[list[str]] = None,
                        env: Optional[dict[str, str]] = None) -> tuple[bool, str]:
    """MIDTRAIN is opt-in: env MIDTRAIN_MASK=1 or the --midtrain argv flag.

    Pack mode comes from MIDTRAIN_PACK (default "bucket" = the conservative
    variant). Returns (enabled, pack_mode).
    """
    argv = list(sys.argv if argv is None else argv)
    env = dict(os.environ if env is None else env)
    enabled = ("--midtrain" in argv) or (env.get("MIDTRAIN_MASK", "") == "1")
    pack_mode = env.get("MIDTRAIN_PACK", "bucket").strip().lower()
    if pack_mode not in VALID_PACK_MODES:
        raise SystemExit(
            f"MIDTRAIN_PACK={pack_mode!r} invalid; expected one of {VALID_PACK_MODES}")
    return enabled, pack_mode


def assert_midtrain_safe(env: Optional[dict[str, str]] = None,
                         model_type: str = "",
                         pack_mode: str = "bucket") -> None:
    """Hard guards for the unsloth path on this box (B4 saga, 2026-09-03/04).

    1. The two B4 knobs are mandatory for this arch on this box
       (scripts/run_b4_unsloth_safe.sh: compile off — CUDA graph-capture crash;
       auto padding-free off — GDN-state suspect #2). MIDTRAIN must not be
       fired with them unset.
    2. Sequence packing is refused for linear-attention (GDN/FLA) model
       types: see module docstring point 2.
    """
    env = dict(os.environ if env is None else env)
    missing = [k for k in ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_DISABLE_AUTO_PADDING_FREE")
               if env.get(k, "0") != "1"]
    if missing:
        raise SystemExit(
            f"MIDTRAIN requires the B4-saga knobs {missing} set to 1 on this box "
            "(scripts/run_b4_unsloth_safe.sh; qwen3.5-2b GDN crashes without them).")
    mt = (model_type or "").lower()
    if pack_mode == "seq" and any(mt.startswith(p) for p in LINEAR_ATTENTION_MODEL_TYPES):
        raise SystemExit(
            f"MIDTRAIN_PACK=seq refused for model_type={model_type!r}: linear-attention "
            "(GDN) layers receive no per-sample boundary metadata in the model forward, "
            "so sequence packing bleeds recurrent state across samples. Use the default "
            "bucket variant (one sample per row, length-grouped batches).")


# ---------------------------------------------------------------------------
# completion-only labels
# ---------------------------------------------------------------------------

def lcp_len(a: list[int], b: list[int]) -> int:
    """Longest common prefix length of two id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_completion_labels(full_ids: list[int],
                            prompt_ids: Optional[list[int]] = None,
                            target_ids: Optional[list[int]] = None) -> tuple[list[int], int, bool]:
    """Labels with everything but the completion span masked out of the loss.

    Two routes (chosen by the caller from char-level composition):

    - prefix route (`prompt_ids`): the full text is tokenized exactly once
      (the legacy token stream); the prompt is tokenized separately and its
      longest matching token prefix in `full_ids` is set to IGNORE_INDEX.
      When the pre-tokenizer splits cleanly at the prompt/target char
      boundary (the normal case for astfim_v1 rows, whose prompt ends with
      '<|end|>\\n' — verified 40/40 on the real corpus), the mask is exact.
      If BPE merges across the seam, the common prefix stops BEFORE the
      merged token, leaving the straddling token in the loss (it contains
      completion characters — the documented convention).
    - suffix route (`target_ids` only): masks everything except the longest
      common token SUFFIX — used for corpora where the completion is
      composed as text.endswith(target) and there is no usable prompt
      prefix (e.g. astfim_v1/fixed, which drops the '<|end|>\\n' separator).

    Returns (labels, n_masked, exact).
    """
    if prompt_ids is not None:
        k = lcp_len(full_ids, prompt_ids)
        exact = (k == len(prompt_ids)) and len(full_ids) >= len(prompt_ids)
        return [IGNORE_INDEX] * k + list(full_ids[k:]), k, exact
    if target_ids is not None:
        n = min(len(full_ids), len(target_ids))
        i = 0
        while i < n and full_ids[-1 - i] == target_ids[-1 - i]:
            i += 1
        exact = (i == len(target_ids)) and len(full_ids) >= len(target_ids)
        head = len(full_ids) - i
        return [IGNORE_INDEX] * head + list(full_ids[head:]), head, exact
    raise ValueError("need prompt_ids (prefix route) or target_ids (suffix route)")


# ---------------------------------------------------------------------------
# example building (rows -> per-sample training rows)
# ---------------------------------------------------------------------------

def build_midtrain_examples(rows: Iterable[dict[str, Any]],
                            tokenize: Callable[[str], list[int]],
                            max_seq_length: int = 2048,
                            ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rows {text, prompt|target} -> training rows {input_ids, labels, length}.

    Route selection is char-level and exact: if text.startswith(prompt) the
    prefix route masks the prompt; elif text.endswith(target) the suffix
    route masks all but the target; otherwise the row is dropped (counted as
    seam_dirty — pointing MIDTRAIN at such a corpus is a data bug, and
    train_sft aborts on a high dirty rate). Deterministic; order-preserving.
    Drops (and counts) rows longer than max_seq_length (dropped, NOT
    truncated — a truncated FIM row would either lose its target (right-
    trunc) or fabricate a mid-file context window (left-trunc)) or with zero
    loss tokens.
    """
    examples: list[dict[str, Any]] = []
    stats = {"rows_in": 0, "route_prefix": 0, "route_suffix": 0, "seam_dirty": 0,
             "dropped_no_field": 0, "dropped_too_long": 0,
             "dropped_no_loss_tokens": 0, "seam_adjusted": 0, "rows_out": 0}
    for row in rows:
        stats["rows_in"] += 1
        text = row.get("text")
        prompt = row.get("prompt")
        target = row.get("target")
        if text is None or (prompt is None and target is None):
            stats["dropped_no_field"] += 1
            continue
        full_ids = list(tokenize(text))
        if len(full_ids) > max_seq_length:
            stats["dropped_too_long"] += 1
            continue
        if prompt is not None and text.startswith(prompt):
            labels, _, exact = build_completion_labels(
                full_ids, prompt_ids=list(tokenize(prompt)) if prompt else [])
            stats["route_prefix"] += 1
        elif target is not None and text.endswith(target):
            labels, _, exact = build_completion_labels(
                full_ids, target_ids=list(tokenize(target)) if target else [])
            stats["route_suffix"] += 1
        else:
            stats["seam_dirty"] += 1
            continue
        if not exact:
            stats["seam_adjusted"] += 1
        if all(x == IGNORE_INDEX for x in labels):
            stats["dropped_no_loss_tokens"] += 1
            continue
        examples.append({"input_ids": full_ids, "labels": labels, "length": len(full_ids)})
        stats["rows_out"] += 1
    return examples, stats


# ---------------------------------------------------------------------------
# datasets.map wiring primitives (used by train_sft.py MIDTRAIN mode)
# ---------------------------------------------------------------------------

def midtrain_map_row(row: dict[str, Any],
                     tokenize: Callable[[str], list[int]],
                     ) -> dict[str, Any]:
    """The datasets.map row-fn for train_sft MIDTRAIN mode.

    Single full-text tokenization (the legacy token stream); route selection
    is char-level and exact: prefix route when text.startswith(prompt)
    (astfim_v1 root shape), suffix route when text.endswith(target)
    (astfim_v1/fixed shape), else flagged dirty (route=2, zero-length —
    filtered out and caught by seam_guard). No EOS is appended (astfim rows
    end in <|end|>).
    """
    full = list(tokenize(row["text"]))
    if row.get("prompt") and row["text"].startswith(row["prompt"]):
        labels, _, exact = build_completion_labels(
            full, prompt_ids=list(tokenize(row["prompt"])))
        route = 0
    elif row.get("target") and row["text"].endswith(row["target"]):
        labels, _, exact = build_completion_labels(
            full, target_ids=list(tokenize(row["target"])))
        route = 1
    else:
        return {"input_ids": [], "labels": [], "length": 0, "n_loss": 0,
                "route": 2, "exact": 0}
    return {"input_ids": full, "labels": labels, "length": len(full),
            "n_loss": sum(1 for x in labels if x != IGNORE_INDEX),
            "route": route, "exact": int(exact)}


def seam_guard(tokd: Any, split: str = "train",
               allow_dirty: bool = False, threshold: float = 0.05) -> dict[str, float]:
    """Fail-loud corpus-shape guard over a mapped dataset (before filtering).

    Aborts when more than `threshold` of rows matched NEITHER composition —
    e.g. pointing MIDTRAIN at a corpus whose prompt/text fields don't compose
    would silently train on the wrong spans.
    """
    routes = tokd["route"]
    exacts = tokd["exact"]
    n0 = len(tokd)
    n_routed = sum(1 for r in routes if r in (0, 1))
    n_exact = sum(exacts)
    dirty_rate = 1.0 - (n_routed / max(n0, 1))
    if dirty_rate > threshold and not allow_dirty:
        raise SystemExit(
            f"[midtrain:{split}] {dirty_rate:.1%} of rows match neither "
            f"text.startswith(prompt) nor text.endswith(target) — wrong "
            f"corpus shape for completion masking (set "
            f"MIDTRAIN_ALLOW_DIRTY_SEAM=1 to override).")
    return {"n0": n0, "n_routed": n_routed, "n_exact": n_exact,
            "dirty_rate": dirty_rate}


# ---------------------------------------------------------------------------
# sequence packing (MIDTRAIN_PACK=seq; full-attention bases only)
# ---------------------------------------------------------------------------

def pack_examples_ffd(examples: list[dict[str, Any]],
                      block_size: int = 2048,
                      ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Greedy first-fit-decreasing packing of WHOLE samples into blocks.

    Never splits a sample across blocks (a split sample's completion span
    could land in a block without its context). Deterministic: stable sort by
    length desc, first-fit into open blocks. Oversize samples are dropped and
    counted (build_midtrain_examples already caps at max_seq_length; this is
    the standalone-function guard).
    """
    order = sorted(range(len(examples)), key=lambda i: (-examples[i]["length"], i))
    blocks: list[dict[str, Any]] = []
    space: list[int] = []
    stats = {"n_in": len(examples), "dropped_oversize": 0, "n_blocks": 0,
             "packed_tokens": 0, "capacity": 0}
    for i in order:
        ex = examples[i]
        n = ex["length"]
        if n > block_size:
            stats["dropped_oversize"] += 1
            continue
        placed = False
        for b, s in enumerate(space):
            if s >= n:
                blocks[b]["input_ids"].extend(ex["input_ids"])
                blocks[b]["labels"].extend(ex["labels"])
                blocks[b]["sample_lengths"].append(n)
                space[b] -= n
                placed = True
                break
        if not placed:
            blocks.append({"input_ids": list(ex["input_ids"]),
                           "labels": list(ex["labels"]),
                           "sample_lengths": [n]})
            space.append(block_size - n)
    stats["n_blocks"] = len(blocks)
    stats["packed_tokens"] = sum(sum(b["sample_lengths"]) for b in blocks)
    stats["capacity"] = len(blocks) * block_size
    return blocks, stats


def build_packed_position_ids(sample_lengths: list[int], padded_len: int) -> list[int]:
    """Position ids that restart at 0 at each sample boundary.

    Within a sample: 0..L-1 (RoPE layers then read correct relative
    positions). Padding positions get 0 (their attention-mask column/row is
    fully masked anyway).
    """
    pos: list[int] = []
    for n in sample_lengths:
        pos.extend(range(n))
    if len(pos) > padded_len:
        raise ValueError(f"sample_lengths sum {len(pos)} exceeds padded_len {padded_len}")
    pos.extend([0] * (padded_len - len(pos)))
    return pos


def build_packed_attn_mask_4d(sample_lengths: list[int], padded_len: int) -> list[list[bool]]:
    """Block-diagonal causal attention mask (L x L bool, True = attend).

    mask[i][j] is True iff i and j belong to the SAME sample AND j <= i
    (causal within the sample) AND i, j are real (non-pad) positions. Samples
    packed into one row can never attend to each other or to padding.
    """
    # sample id per position; -1 for padding
    sample_of: list[int] = []
    for s, n in enumerate(sample_lengths):
        sample_of.extend([s] * n)
    sample_of.extend([-1] * (padded_len - len(sample_of)))
    mask = [[False] * padded_len for _ in range(padded_len)]
    for i in range(padded_len):
        si = sample_of[i]
        if si < 0:
            continue  # pad row: attends to nothing
        for j in range(i + 1):
            if sample_of[j] == si:
                mask[i][j] = True
    return mask


@dataclass
class MidtrainPackedCollator:
    """Batch collator for MIDTRAIN_PACK=seq blocks.

    Pads blocks to the batch max and emits the isolation tensors:
    input_ids, labels (-100 on pads), position_ids (reset per sample), and a
    4D bool attention mask (1,1,L,L) built by build_packed_attn_mask_4d.
    Rows WITHOUT sample_lengths (e.g. the unpacked eval rows) are treated as
    one sample each, so the same collator serves train and eval.
    NOTE: exact isolation requires an attention backend that honors a
    provided 4D mask (SDPA/eager). Validate on the target backend with a
    2-row smoke before any real use.
    """

    pad_token_id: int = 0
    return_tensors: str = field(default="pt", repr=False)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        batch_ids, batch_labels, batch_pos, batch_mask = [], [], [], []
        for ex in examples:
            lens = ex.get("sample_lengths") or [len(ex["input_ids"])]
            if sum(lens) != len(ex["input_ids"]):
                raise ValueError(
                    f"sample_lengths {lens} do not sum to input_ids length "
                    f"{len(ex['input_ids'])} — corrupted block")
            ids = list(ex["input_ids"]) + [self.pad_token_id] * (max_len - len(ex["input_ids"]))
            labels = list(ex["labels"]) + [IGNORE_INDEX] * (max_len - len(ex["labels"]))
            pos = build_packed_position_ids(lens, max_len)
            mask = build_packed_attn_mask_4d(lens, max_len)
            batch_ids.append(ids)
            batch_labels.append(labels)
            batch_pos.append(pos)
            batch_mask.append(mask)
        t = {"input_ids": torch.tensor(batch_ids, dtype=torch.long),
             "labels": torch.tensor(batch_labels, dtype=torch.long),
             "position_ids": torch.tensor(batch_pos, dtype=torch.long),
             "attention_mask": torch.tensor(batch_mask, dtype=torch.bool).unsqueeze(1)}
        return t
