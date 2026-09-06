"""CPU unit tests for the B9 SeleKT instrument (zcode-b9-select, 2026-09-06).

Covers: flag parsing + guards, the TRL-legacy-equivalent tokenization (EOS
append + truncate — the token-stream comparability contract vs the banked b4
rung), the closed-form per-token gradient importance (vs brute-force
||softmax - onehot||), the global keep-threshold quantile, label building
(boundary semantics, position-0 convention, liveness fallback), the batched
forward probe on a stub model, and the flag-off byte-compatibility contract
for train_sft.py (source-pinned, same pattern as test_midtrain_data.py).
All pure CPU; the real-tokenizer test skips when local artifacts are absent.
Run: .venv-sft/bin/python -m pytest <path> -q  (or uv run --with pytest;
nice -n 19 quiet-window convention).
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from selekt_data import (  # noqa: E402
    IGNORE_INDEX, build_selekt_labels, compute_keep_threshold,
    importances_from_logits, legacy_tokenize_row, parse_selekt_flag,
    probe_importances, assert_selekt_safe, summarize_masking)

REPO = Path(__file__).resolve().parents[2]
TRAIN_SFT = REPO / "experiments" / "training" / "train_sft.py"
SELEKT_SRC = REPO / "experiments" / "training" / "selekt_data.py"
QWEN_TOK_DIR = REPO / "experiments" / "models" / "qwen3.5-2b-base-text-hf"
SFT_V7_TRAIN = Path("/mnt/h/sepalith/datasets/sft_v7/train.jsonl")

KNOBS = {"UNSLOTH_COMPILE_DISABLE": "1", "UNSLOTH_DISABLE_AUTO_PADDING_FREE": "1"}


def char_tok(text: str) -> list[int]:
    return [ord(c) % 251 for c in text]


# ---------------------------------------------------------------------------
# flag parsing + guards
# ---------------------------------------------------------------------------

def test_flag_off_by_default():
    assert parse_selekt_flag(["train_sft.py"], {}) == (False, 0.5, 4096)
    assert parse_selekt_flag(["train_sft.py", "model", "3000"], {})[0] is False


def test_flag_opt_in_env_and_arg():
    assert parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1"})[0] is True
    assert parse_selekt_flag(["t.py", "--selekt"], {})[0] is True
    assert parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1", "SELEKT_KEEP": "0.3"}) == \
        (True, 0.3, 4096)
    assert parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1", "SELEKT_PROBE_BUDGET": "8192"}) == \
        (True, 0.5, 8192)


def test_invalid_keep_and_budget_rejected():
    for bad in ("0", "-0.1", "1.5", "nan"):
        with pytest.raises(SystemExit):
            parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1", "SELEKT_KEEP": bad})
    with pytest.raises(SystemExit):
        parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1", "SELEKT_PROBE_BUDGET": "8"})
    with pytest.raises(SystemExit):
        parse_selekt_flag(["t.py"], {"SELEKT_MASK": "1", "SELEKT_PROBE_BUDGET": "x"})


def test_guard_requires_b4_knobs_and_refuses_midtrain():
    with pytest.raises(SystemExit):
        assert_selekt_safe(env={}, model_type="qwen3_5")
    with pytest.raises(SystemExit):
        assert_selekt_safe(env=KNOBS, model_type="qwen3_5", midtrain_enabled=True)
    assert_selekt_safe(env=KNOBS, model_type="qwen3_5") is None


# ---------------------------------------------------------------------------
# TRL-legacy-equivalent tokenization
# ---------------------------------------------------------------------------

def test_legacy_tokenize_appends_eos_like_trl():
    # TRL add_eos: append the eos STRING iff the text does not end with it
    ids = legacy_tokenize_row("abc", char_tok, "<eos>")
    assert ids == char_tok("abc") + char_tok("<eos>")
    ids2 = legacy_tokenize_row("abc<eos>", char_tok, "<eos>")
    assert ids2 == char_tok("abc<eos>")  # already ends with eos: no double append


def test_legacy_tokenize_truncates_to_max_seq():
    ids = legacy_tokenize_row("x" * 5000, char_tok, "<eos>", max_seq_length=2048)
    assert len(ids) == 2048
    assert ids == (char_tok("x" * 5000) + char_tok("<eos>"))[:2048]


def test_legacy_tokenize_empty_eos_convention():
    # a falsy eos_token disables appending (defensive; not the real config)
    assert legacy_tokenize_row("abc", char_tok, "") == char_tok("abc")


@pytest.mark.skipif(not QWEN_TOK_DIR.exists() or not SFT_V7_TRAIN.exists(),
                    reason="local qwen3.5-2b tokenizer or sft_v7 not present")
def test_real_token_stream_matches_trl_legacy_path():
    # the comparability contract: SELEKT rows are byte-identical in token
    # space to what TRL's text path feeds the banked b4 rung
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN_TOK_DIR), local_files_only=True)
    eos = tok.eos_token
    n = 0
    with open(SFT_V7_TRAIN) as f:
        for line in f:
            if n >= 20:
                break
            row = json.loads(line)
            got = legacy_tokenize_row(row["text"],
                                      lambda t: tok(t)["input_ids"], eos, 2048)
            # TRL: add_eos (string append) then tokenize then truncate
            text = row["text"] if row["text"].endswith(eos) else row["text"] + eos
            want = tok(text)["input_ids"][:2048]
            assert got == want
            n += 1


# ---------------------------------------------------------------------------
# closed-form importance
# ---------------------------------------------------------------------------

def _brute_norm(logits_row: torch.Tensor, y: int) -> float:
    p = torch.softmax(logits_row.float(), dim=-1)
    e = torch.zeros_like(p)
    e[y] = 1.0
    return float(torch.norm(p - e).item())


def test_importance_zero_when_perfectly_confident_and_correct():
    V = 8
    logits = torch.full((1, V), -1e9)
    logits[0, 3] = 1e9
    ids = [9, 3]  # target position 1 = token 3, predicted perfectly
    imp = importances_from_logits(logits, ids)
    assert imp.shape == (2,)
    assert imp[0] == 0.0            # position 0: never a target
    assert imp[1] == pytest.approx(0.0, abs=1e-4)


def test_importance_hand_computed_two_vocab():
    # uniform softmax over 2, target 0: ||[.5,.5]-[1,0]|| = sqrt(.5)
    logits = torch.zeros((1, 2))
    imp = importances_from_logits(logits, [7, 0])
    assert imp[1] == pytest.approx(math.sqrt(0.5), abs=1e-6)


def test_importance_confident_and_wrong_is_sqrt2():
    V = 4
    logits = torch.full((1, V), -1e9)
    logits[0, 2] = 1e9  # all mass on token 2
    imp = importances_from_logits(logits, [9, 0])  # target 0
    assert imp[1] == pytest.approx(math.sqrt(2.0), abs=1e-4)


def test_importance_matches_brute_force_on_random_logits():
    g = torch.Generator().manual_seed(3407)
    logits = torch.randn((40, 37), generator=g)
    ids = torch.randint(0, 37, (40,), generator=g).tolist()
    imp = importances_from_logits(logits, ids, chunk=7)
    for t in range(1, 40):
        assert imp[t] == pytest.approx(_brute_norm(logits[t - 1], ids[t]), abs=1e-5)


def test_importance_short_rows():
    assert list(importances_from_logits(torch.zeros((1, 5)), [1])) == [0.0]
    assert list(importances_from_logits(torch.zeros((0, 5)), [])) == []


# ---------------------------------------------------------------------------
# threshold + labels
# ---------------------------------------------------------------------------

def test_threshold_is_exact_quantile():
    imps = [np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 5.0, 6.0, 7.0, 8.0])]
    thr = compute_keep_threshold(imps, 0.5)
    vals = [1, 2, 3, 4, 5, 6, 7, 8]
    assert thr["n_positions"] == 8
    assert thr["tau"] == pytest.approx(4.5)  # median of the 8 target positions
    # position-0 zeros never enter the quantile
    thr2 = compute_keep_threshold([np.array([99.0, 1.0])], 0.5)
    assert thr2["tau"] == pytest.approx(1.0)


def test_threshold_empty_raises():
    with pytest.raises(SystemExit):
        compute_keep_threshold([np.array([0.0]), np.array([0.0])], 0.5)


def test_labels_boundary_and_pos0():
    ids = [10, 11, 12, 13]
    imp = np.array([0.0, 0.9, 0.5, 0.1])
    labels, n_kept, fb = build_selekt_labels(ids, imp, tau=0.5)
    # keep iff imp >= tau (boundary KEPT); position 0 always masked
    assert labels == [IGNORE_INDEX, 11, 12, IGNORE_INDEX]
    assert n_kept == 2 and fb == 0


def test_labels_liveness_fallback_keeps_argmax():
    ids = [10, 11, 12]
    imp = np.array([0.0, 0.1, 0.2])
    labels, n_kept, fb = build_selekt_labels(ids, imp, tau=0.5)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 12]  # argmax target kept
    assert n_kept == 1 and fb == 1


def test_labels_degenerate_short_row():
    labels, n_kept, fb = build_selekt_labels([7], np.array([0.0]), tau=0.1)
    assert labels == [IGNORE_INDEX] and n_kept == 0 and fb == 0


def test_masking_keeps_expected_density_on_distinct_values():
    rng = np.random.default_rng(42)
    imps = [rng.random(101) for _ in range(50)]
    thr = compute_keep_threshold(imps, 0.5)
    rows = [list(range(1, 102)) for _ in range(50)]
    labs, kept, fbs = [], 0, 0
    for r, im in zip(rows, imps):
        lab, k, fb = build_selekt_labels(r, im, thr["tau"])
        labs.append(lab)
        kept += k
        fbs += fb
    stats = summarize_masking(rows, labs, fbs)
    assert stats["target_positions"] == 50 * 100
    assert stats["kept_frac"] == pytest.approx(0.5, abs=0.01)
    assert stats["fallback_rows"] == 0
    # every row keeps at least one token (liveness — no all-masked batch)
    assert all(any(t != IGNORE_INDEX for t in lab) for lab in labs)


# ---------------------------------------------------------------------------
# batched probe on a stub model
# ---------------------------------------------------------------------------

class _StubModel:
    """Deterministic forward: logits depend on the position's own id + index.

    The small per-position term keeps importances continuous (no tied mass —
    tie behavior is covered separately in the threshold tests). Real-model
    padding isolation is the attention_mask's job (not under test here);
    what IS under test: batching/budget logic, order restoration, value
    equality vs the single-row direct computation, mask construction.
    """

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.calls: list[tuple] = []

    def parameters(self):
        return iter(())  # no params -> probe falls back to CPU tensors

    def _row_logits(self, ids: list[int]) -> torch.Tensor:
        L = len(ids)
        logits = torch.full((L, self.vocab), 0.25)
        for i, t in enumerate(ids):
            logits[i, t % self.vocab] += 3.0 + 0.01 * i
        return logits

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False):
        assert use_cache is False  # probe must not build KV caches
        self.calls.append((tuple(input_ids.shape), attention_mask.sum(dim=1).tolist()))
        B, L = input_ids.shape
        rows = [self._row_logits(input_ids[b][:L].tolist()) for b in range(B)]
        logits = torch.zeros((B, L, self.vocab))
        for b, r in enumerate(rows):
            logits[b, :r.shape[0], :] = r
        return SimpleNamespace(logits=logits)


class _ParamStubModel(_StubModel):
    """Stub with a registered parameter — probe must place inputs on ITS
    device and dtype-path (structural check for the direct-call path)."""

    def __init__(self, vocab: int):
        super().__init__(vocab)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])


def test_probe_preserves_order_and_matches_direct():
    stub = _StubModel(64)
    rows = [[1, 2, 3, 4], [5, 6], [7, 8, 9], [10] * 17, [11, 12, 13, 14, 15]]
    got = probe_importances(stub, rows, token_budget=24, pad_token_id=0,
                            progress_every=0, log=lambda s: None)
    assert len(got) == len(rows)
    for r, g in zip(rows, got):
        want = importances_from_logits(stub._row_logits(r), r)
        np.testing.assert_allclose(g, want, atol=1e-5)


def test_probe_batches_within_budget_and_builds_masks():
    stub = _StubModel(64)
    rows = [[i, i + 1, i + 2] for i in range(20)]  # uniform length 3
    probe_importances(stub, rows, token_budget=12, pad_token_id=0,
                      progress_every=0, log=lambda s: None)
    assert len(stub.calls) > 1  # actually split
    for (B, L), reals in stub.calls:
        assert B * L <= 12  # padded-token budget respected
        assert L == max(reals)  # right-padded to the batch max
        assert all(0 < n <= L for n in reals)  # attention mask: ones on real only


def test_probe_single_token_rows_get_zero_importance():
    stub = _StubModel(8)
    got = probe_importances(stub, [[1], [2, 3]], token_budget=8,
                            progress_every=0, log=lambda s: None)
    assert list(got[0]) == [0.0]
    assert len(got[1]) == 2


def test_probe_places_inputs_on_model_device_and_disables_cache():
    # regression for the first launch failure: the probe calls the model
    # DIRECTLY (no accelerate placement) — inputs must be built on the
    # model's own parameter device; use_cache must be False
    stub = _ParamStubModel(32)
    rows = [[1, 2, 3], [4, 5, 6, 7]]
    got = probe_importances(stub, rows, token_budget=8,
                            progress_every=0, log=lambda s: None)
    assert all(len(g) == len(r) for g, r in zip(got, rows))  # ran clean


def test_probe_end_to_end_threshold_flow():
    # tokenize -> probe -> threshold -> labels on the stub: density ~ keep
    stub = _StubModel(64)
    rng = np.random.default_rng(7)
    rows = [rng.integers(0, 64, size=rng.integers(20, 40)).tolist() for _ in range(40)]
    imps = probe_importances(stub, rows, token_budget=64, pad_token_id=0,
                             progress_every=0, log=lambda s: None)
    thr = compute_keep_threshold(imps, 0.5)
    labs = [build_selekt_labels(r, im, thr["tau"]) for r, im in zip(rows, imps)]
    stats = summarize_masking(rows, [l[0] for l in labs], sum(l[2] for l in labs))
    assert 0.45 < stats["kept_frac"] < 0.55


# ---------------------------------------------------------------------------
# flag-off byte-compatibility contract (source-pinned)
# ---------------------------------------------------------------------------

def test_train_sft_off_path_pinned():
    src = TRAIN_SFT.read_text()
    # legacy dataset map, verbatim, exactly once, in the final else-branch
    legacy = ('else:\n    ds = ds.map(lambda x: {"text": x["text"]}, remove_columns=[\n'
              '        c for c in ds["train"].column_names if c != "text"])')
    assert src.count(legacy) == 1
    # legacy train/eval selection lines verbatim (shuffle seed 42 + 48k cap)
    assert 'ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))' in src
    assert 'ds["eval"].select(range(500))' in src
    # SFTConfig literals unchanged; instrument kwargs inject ONLY via spread
    assert "max_seq_length=2048, **midtrain_kwargs)" in src
    # the LoRA legacy literals untouched by B9 (lr/seed/bs defaults)
    assert "learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type=\"cosine\"" in src
    assert 'per_device_train_batch_size=int(os.environ.get("SFT_PD_BATCH", "4"))' in src
    # the flag is strictly opt-in
    assert '("--selekt" in argv) or (env.get("SELEKT_MASK", "") == "1")' in \
        SELEKT_SRC.read_text()


def test_train_sft_selekt_branch_guards_before_data_prep():
    src = TRAIN_SFT.read_text()
    assert src.index("parse_selekt_flag(_ARGV)") < src.index("if SELEKT:")
    assert "assert_selekt_safe" in src  # knob/mutual-exclusion guards active
    # one masking/training mechanism per arm
    assert "SELEKT and FULL_FT are mutually exclusive" in src


def test_train_sft_selekt_arg_position_independent():
    src = TRAIN_SFT.read_text()
    assert "_ARGV = list(sys.argv)" in src
    assert 'sys.argv.remove("--selekt")' in src
    argv = ["train_sft.py", "--selekt", "model", "3000", "data"]
    argv2 = argv[:]
    argv2.remove("--selekt")
    assert argv2[1:3] == ["model", "3000"]  # positional parse unaffected


def test_selekt_uses_legacy_stream_and_same_selection():
    src = TRAIN_SFT.read_text()
    # the SAME shuffle(42)+48k selection the legacy SFTTrainer call trains on
    assert src.count('ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))') >= 2
    assert "legacy_tokenize_row" in src  # TRL-equivalent stream (+eos, 2048)
    # probe artifact saved before training (audit/crash-resume)
    assert 'OUT / "selekt_probe.json"' in src
