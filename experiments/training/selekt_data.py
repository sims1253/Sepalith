"""B9 SeleKT instrument: token-space gradient-importance masking (pure functions).

Landed 2026-09-06 by zcode-b9-select as the B9 prerequisite (base-bakeoff plan
§2 B9; EXPERIMENT-QUEUE §2b B9 row; production-finetune plan §2 masking-policy
slot). With SELEKT off, train_sft.py runs its legacy pipeline verbatim
(byte-compatibility contract, same as the B8 midtrain instrument — see
test_midtrain_data.py / test_selekt_data.py).

====================== PRE-REGISTERED ADAPTATION (B9) ======================

SOURCE METHOD (SeleKT, NextCoder ICML'25, as summarized in
docs/research/model-survey-2026-08-20.md §Q1 + runbook §2 B9): (a) a dense
gradient-based step identifies the weights most important for code-editing;
(b) training updates are SPARSELY PROJECTED back onto the base model so
codegen/instruction abilities survive. Known failure mode it guards: naive
SFT LOSES edit ability (Qwen2.5-Coder-7B CanItEdit 48.1 base -> 36.7 SFT;
SeleKT lifts to 50.5).

RUNBOOK PROTOCOL (§2 B9, verbatim intent): "one probe batch through the
merged winner, per-module gradient importance via hooks; retrain LoRA
restricted to top-importance modules vs the unrestricted a0-style run (same
data/steps). Read-outs: edit battery (within 1.0pp = no harm) + retention
proxies (midtyping Py-align, drafting judged). Adopt if no edit harm AND
retention improves."

EXACT ADAPTATION IMPLEMENTED HERE (pre-registered 2026-09-06, before the
arm was fired; the runbook's module-restriction variant is NOT what runs —
the queue-manager brief pins the arm to the FULL b4 attachment so the
trainable-parameter gate stays comparable: 21,823,488):

1. TOKEN-SPACE, NOT PARAMETER-SPACE. SeleKT sparsifies the UPDATE in weight
   space; we sparsify the LOSS in token space on the SAME full attachment
   (all 96 modules = the b4 target list; trainable line must equal
   21,823,488). The anti-forgetting rationale carries over: fewer, more
   informative loss tokens -> a sparser effective update -> less drift of
   the base's general code distribution. This is the minimal defensible
   version that keeps the arm single-variable (labels are the ONLY thing
   that differs from the banked b4 rung: same rows, same order, same token
   streams, same steps/seed/lr).

2. IMPORTANCE = the exact per-token loss gradient w.r.t. the final-layer
   logits, in closed form. For target token y_t at position t predicted
   from logits z_{t-1}: dL_t/dz_{t-1} = softmax(z_{t-1}) - onehot(y_t), so

       I_t = || softmax(z_{t-1}) - onehot(y_t) ||_2
           = sqrt( ||p||^2 + 1 - 2 p[y_t] )        (p = softmax(z_{t-1}))

   computed from FORWARD passes only (no backward, no per-weight hooks).
   This is the standard gradient-informed token score; I_t -> 0 when the
   base is already confident-and-correct on the token, -> sqrt(2) when the
   base is confident-and-wrong. High-I tokens are exactly where SFT must
   move the model (the zeta2 contract, edit-block boundaries); low-I tokens
   are generic code the base already models — training on them is the
   forgetting vector SeleKT prunes.

3. PROBE STATE = the b4 INIT state: the base model with the freshly
   attached ZERO-INIT LoRA (lora_B = 0, dropout 0 -> forward is exactly the
   base's). Deviation from the runbook's "through the merged winner"
   (pre-registered rationale): importance for deciding what TRAINING should
   see must be measured where training starts; probing a trained model
   would rank the tokens it already handles. Same convention as SeleKT's
   dense gradient step (computed at the base, before the update).

4. SELECTION RULE: a single GLOBAL keep-threshold tau = the (1 - keep_frac)
   quantile of I_t over ALL real target positions (positions t >= 1) of the
   selected 48k train rows (the same shuffle(seed=42)+48k selection the
   legacy path trains on). A token trains iff I_t >= tau. Default
   keep_frac = 0.5 (SELEKT_KEEP env) — the sparse-projection analog at half
   density. LIVENESS GUARANTEE: every row keeps at least its single
   argmax-I token (fallback), so no all-masked batch (NaN CE) can occur and
   every row still contributes. Position 0 is never a loss target (causal
   shift) and is set -100 explicitly. The eval split (first 500 rows, same
   selection as legacy) is masked with the SAME tau (threshold is a
   property of the training signal; eval kept-fraction is reported, not
   forced).

5. TOKEN-STREAM EQUIVALENCE: rows are tokenized EXACTLY as the TRL legacy
   text path does it (TRL 0.24 SFTTrainer._prepare_dataset): append the EOS
   token string when the text does not already end with it (sft_v7 rows end
   in '>>>>>>> UPDATED', so EOS is always appended; verified no cross-bound
   BPE merge on the real corpus -> ids == tokenize(text) + [eos_id]), then
   truncate to max_seq_length=2048 (matches the legacy truncate step; the
   banked b4 rows longer than 2048 were likewise truncated, not dropped).
   The probe sees and masks exactly the stream b4 trained on.

6. PRE-REGISTERED READOUT / VERDICT (mirrors the runbook rule; control =
   the BANKED b4 rung, no retrain):
   - NO-EDIT-HARM: eval_scenarios exact within 1.0pp of b4's 76.5 (valid vs
     85.1 as context; McNemar n=255 paired rows).
   - RETENTION IMPROVES: at least one of — format_propagation valid > b4's
     71.6 with McNemar p < 0.05; noopFP scored (n=204) < b4's 58.8 with
     paired McNemar p < 0.05; midtyping line_f1 strictly above b4's floor
     (0.006 raw / 0.033 suffix) with the 18/18 (i,sha) join intact.
   - ADOPT iff no-edit-harm AND retention improves; else the plain-SFT
     masking policy stands (default OFF). b4 itself showed no external
     edit-loss on this battery, so the realistic positive case is a
     restraint/format_propagation gain at quality parity; the realistic
     negative is parity-everywhere (then masking is pure cost: a probe
     forward pass) or quality harm from the halved signal budget.
   ============================================================================

All builders are pure list/array functions taking a `tokenize` callable or a
`model` callable so they unit-test CPU-side without unsloth/trl/GPU (house
pattern from midtrain_data.py).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

IGNORE_INDEX = -100  # torch/trl CE ignore token (labels convention)

DEFAULT_KEEP_FRAC = 0.5
DEFAULT_PROBE_BUDGET = 4096  # padded tokens per probe forward (bounds logits)


# ---------------------------------------------------------------------------
# flag parsing + guards
# ---------------------------------------------------------------------------

def parse_selekt_flag(argv: Optional[list[str]] = None,
                      env: Optional[dict[str, str]] = None) -> tuple[bool, float, int]:
    """SELEKT is opt-in: env SELEKT_MASK=1 or the --selekt argv flag.

    keep_frac from SELEKT_KEEP (default 0.5, must be in (0, 1]); probe padded-
    token budget from SELEKT_PROBE_BUDGET (default 4096). Returns
    (enabled, keep_frac, probe_budget).
    """
    argv = list(sys.argv if argv is None else argv)
    env = dict(os.environ if env is None else env)
    enabled = ("--selekt" in argv) or (env.get("SELEKT_MASK", "") == "1")
    try:
        keep = float(env.get("SELEKT_KEEP", str(DEFAULT_KEEP_FRAC)))
    except ValueError:
        raise SystemExit(f"SELEKT_KEEP={env.get('SELEKT_KEEP')!r} is not a number")
    if not (0.0 < keep <= 1.0):
        raise SystemExit(f"SELEKT_KEEP={keep!r} invalid; expected 0 < keep <= 1")
    try:
        budget = int(env.get("SELEKT_PROBE_BUDGET", str(DEFAULT_PROBE_BUDGET)))
    except ValueError:
        raise SystemExit(f"SELEKT_PROBE_BUDGET={env.get('SELEKT_PROBE_BUDGET')!r} is not an int")
    if budget < 64:
        raise SystemExit(f"SELEKT_PROBE_BUDGET={budget!r} too small (min 64)")
    return enabled, keep, budget


def assert_selekt_safe(env: Optional[dict[str, str]] = None,
                       model_type: str = "",
                       midtrain_enabled: bool = False) -> None:
    """Hard guards for the unsloth path on this box (same class as the B8 ones).

    1. The two B4-saga knobs are mandatory (compile off, auto padding-free
       off — scripts/run_b4_unsloth_safe.sh; qwen3.5-2b GDN crashes without
       them). The probe forwards through the same model, so the same guards
       apply.
    2. SELEKT and MIDTRAIN are mutually exclusive masking policies (a single
       arm must isolate one mechanism).
    """
    env = dict(os.environ if env is None else env)
    if midtrain_enabled:
        raise SystemExit(
            "SELEKT and MIDTRAIN are mutually exclusive masking policies — "
            "pick one per arm (B9 pre-registration isolates the token-grad "
            "mechanism alone).")
    missing = [k for k in ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_DISABLE_AUTO_PADDING_FREE")
               if env.get(k, "0") != "1"]
    if missing:
        raise SystemExit(
            f"SELEKT requires the B4-saga knobs {missing} set to 1 on this box "
            "(scripts/run_b4_unsloth_safe.sh; the probe forwards through the "
            "same GDN model).")


# ---------------------------------------------------------------------------
# legacy-equivalent tokenization
# ---------------------------------------------------------------------------

def legacy_tokenize_row(text: str,
                        tokenize: Callable[[str], list[int]],
                        eos_token: str,
                        max_seq_length: int = 2048) -> list[int]:
    """Tokenize a text row EXACTLY as the TRL legacy text path does.

    TRL 0.24 SFTTrainer._prepare_dataset (non-conversational text case):
    first `add_eos` appends the EOS token STRING when the text does not end
    with it, then tokenize_fn runs processing_class(text)["input_ids"], then
    truncate_dataset caps at max_length. Reproduced here so the SELEKT arm's
    token stream is identical to the banked b4 rung's (rows, order, ids).
    """
    if eos_token and not text.endswith(eos_token):
        text = text + eos_token
    return list(tokenize(text))[:max_seq_length]


# ---------------------------------------------------------------------------
# per-token gradient importance (closed form from logits)
# ---------------------------------------------------------------------------

def importances_from_logits(logits: torch.Tensor,
                            input_ids: list[int],
                            chunk: int = 256) -> np.ndarray:
    """I_t = ||softmax(z_{t-1}) - onehot(y_t)||_2 for every position t.

    logits: [L, V] (any float dtype, GPU or CPU) — the row's forward output.
    input_ids: the row's token ids, length L. Returns float32 array of
    length L where entry 0 is 0.0 (position 0 has no predictor — never a
    loss target) and entry t >= 1 is the importance of TARGET token t as
    predicted from logits row t-1. fp32 softmax computed in `chunk`-position
    slices to bound the transient (248k vocab).
    """
    L = len(input_ids)
    out = np.zeros(L, dtype=np.float32)
    if L < 2:
        return out
    preds = logits[: L - 1, :].float()  # predictors for targets 1..L-1
    targets = torch.tensor(input_ids[1:], dtype=torch.long, device=preds.device)
    for s in range(0, preds.shape[0], chunk):
        e = min(s + chunk, preds.shape[0])
        p = torch.softmax(preds[s:e], dim=-1)
        # ||p - onehot||^2 = ||p||^2 - 2 p[y] + 1
        v = (p * p).sum(dim=-1)
        v = v - 2.0 * p.gather(1, targets[s:e].unsqueeze(1)).squeeze(1) + 1.0
        out[s + 1:e + 1] = torch.sqrt(torch.clamp(v, min=0.0)).cpu().numpy()
    return out


def probe_importances(model: Callable[..., Any],
                      rows: list[list[int]],
                      token_budget: int = DEFAULT_PROBE_BUDGET,
                      pad_token_id: int = 0,
                      progress_every: int = 0,
                      log: Callable[[str], None] = print) -> list[np.ndarray]:
    """Forward-only probe: per-token importance for every row, from `model`.

    Deterministic batching: rows sorted by length DESC (stable — ties keep
    index order), greedily packed so max_len_in_batch * n_rows <=
    token_budget (bounds the [B, L, V] logits transient), right-padded with
    attention_mask. The model is called under torch.no_grad() exactly as
    `model(input_ids=..., attention_mask=...) -> .logits`; only REAL
    positions (mask 1) are scored, so padding never enters the importance.
    Returns one float32 array per row, original order preserved.

    NOTE (pre-registered): the caller puts the model in eval() and restores
    train() afterwards; the probe state is the b4 init state (base +
    zero-init LoRA — see module docstring point 3).
    """
    order = sorted(range(len(rows)), key=lambda i: (-len(rows[i]), i))
    imps: list[Optional[np.ndarray]] = [None] * len(rows)
    batch: list[int] = []
    batch_max = 0

    def _flush() -> int:
        nonlocal batch, batch_max
        if not batch:
            return 0
        max_len = batch_max
        ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        for bi, ri in enumerate(batch):
            r = rows[ri]
            ids[bi, :len(r)] = torch.tensor(r, dtype=torch.long)
            att[bi, :len(r)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits
        for bi, ri in enumerate(batch):
            n = len(rows[ri])
            imps[ri] = importances_from_logits(logits[bi, :n, :], rows[ri])
        flushed = len(batch)
        batch, batch_max = [], 0
        return flushed

    t0 = time.time()
    done = 0
    for ri in order:
        n = len(rows[ri])
        if n < 2:
            imps[ri] = np.zeros(n, dtype=np.float32)
            continue
        if batch and max(batch_max, n) * (len(batch) + 1) > token_budget:
            done += _flush()
            if progress_every and done % progress_every < 64:
                log(f"[selekt:probe] {done}/{len(rows)} rows "
                    f"({time.time() - t0:.0f}s)")
        batch.append(ri)
        batch_max = max(batch_max, n)
    done += _flush()
    log(f"[selekt:probe] forwarded {done}/{len(rows)} rows in {time.time() - t0:.0f}s "
        f"(budget {token_budget} padded tokens/fwd)")
    assert all(x is not None for x in imps)
    return imps  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# threshold + labels
# ---------------------------------------------------------------------------

def compute_keep_threshold(importances: list[np.ndarray],
                           keep_frac: float) -> dict[str, float]:
    """tau = the (1 - keep_frac) quantile over ALL real target positions.

    Positions t >= 1 of every row (position 0 is never a target). Ties at
    tau are KEPT (I_t >= tau trains), so a tied mass at the quantile can
    keep slightly more than keep_frac — reported in stats, not corrected
    (determinism beats density exactness).
    """
    vals = np.concatenate([a[1:] for a in importances if len(a) > 1]) \
        if any(len(a) > 1 for a in importances) else np.zeros(0)
    if vals.size == 0:
        raise SystemExit("SELEKT: no target positions found (all rows < 2 tokens)")
    tau = float(np.quantile(vals, 1.0 - keep_frac))
    return {"tau": tau, "n_positions": int(vals.size),
            "imp_mean": float(vals.mean()), "imp_median": float(np.median(vals)),
            "imp_p05": float(np.quantile(vals, 0.05)),
            "imp_p95": float(np.quantile(vals, 0.95)),
            "imp_min": float(vals.min()), "imp_max": float(vals.max())}


def build_selekt_labels(input_ids: list[int],
                        imp_row: np.ndarray,
                        tau: float) -> tuple[list[int], int, int]:
    """Labels with low-importance target positions masked to IGNORE_INDEX.

    labels[0] is always IGNORE_INDEX (position 0 is never a loss target
    under the causal shift). Position t >= 1 keeps its id iff imp_row[t] >=
    tau. LIVENESS: if no position survives, the row's argmax-importance
    position is kept (returns used_fallback=1) — guarantees >= 1 loss token
    per row, so no all-masked batch (NaN CE) is possible. Returns
    (labels, n_kept, used_fallback).
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    if len(input_ids) < 2:
        return labels, 0, 0  # degenerate 1-token rows contribute nothing (none exist in sft_v7)
    kept = [t for t in range(1, len(input_ids)) if imp_row[t] >= tau]
    if not kept:
        kept = [int(np.argmax(imp_row[1:])) + 1]
        fb = 1
    else:
        fb = 0
    for t in kept:
        labels[t] = input_ids[t]
    return labels, len(kept), fb


def summarize_masking(rows: list[list[int]], labels_list: list[list[int]],
                      fallback_rows: int) -> dict[str, float]:
    """Aggregate mask stats for telemetry (fractions over real target positions)."""
    n_targets = sum(max(len(r) - 1, 0) for r in rows)
    n_kept = sum(sum(1 for t in lab if t != IGNORE_INDEX) for lab in labels_list)
    return {"rows": len(rows), "target_positions": n_targets, "kept": n_kept,
            "kept_frac": n_kept / max(n_targets, 1), "fallback_rows": fallback_rows}
