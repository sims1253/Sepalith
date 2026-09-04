"""CPU unit tests for the B8 midtrain instrument (zcode-b8-patch, 2026-09-05).

Covers: completion-only mask correctness on hand-built samples, packing
isolation (position ids / attention mask construction), the flag-off
byte-compatibility contract for train_sft.py, and the env guards. All pure
CPU; the real-tokenizer/real-corpus test skips when the local model or the
NAS dataset is absent. Run: uv run --with pytest python -m pytest <path> -q
(quiet-window convention: nice -n 19).
"""
import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from midtrain_data import (  # noqa: E402
    IGNORE_INDEX, LINEAR_ATTENTION_MODEL_TYPES, MidtrainPackedCollator,
    assert_midtrain_safe, build_completion_labels, build_midtrain_examples,
    build_packed_attn_mask_4d, build_packed_position_ids, lcp_len,
    midtrain_map_row, pack_examples_ffd, parse_midtrain_flag, seam_guard)

REPO = Path(__file__).resolve().parents[2]
TRAIN_SFT = REPO / "experiments" / "training" / "train_sft.py"
QWEN_TOK_DIR = REPO / "experiments" / "models" / "qwen3.5-2b-base-text-hf"
ASTFIM_EVAL = Path("/mnt/h/sepalith/datasets/astfim_v1/eval.jsonl")

# deterministic stub tokenizer: one token per character (id = ord % 251)
def char_tok(text: str) -> list[int]:
    return [ord(c) % 251 for c in text]


def make_row(prompt: str, target: str) -> dict:
    return {"text": prompt + target, "prompt": prompt, "target": target}


# ---------------------------------------------------------------------------
# flag parsing + guards
# ---------------------------------------------------------------------------

def test_flag_off_by_default():
    assert parse_midtrain_flag(["train_sft.py"], {}) == (False, "bucket")
    assert parse_midtrain_flag(["train_sft.py", "model", "3000"], {}) == (False, "bucket")


def test_flag_opt_in_env_and_arg():
    assert parse_midtrain_flag(["train_sft.py"], {"MIDTRAIN_MASK": "1"})[0] is True
    assert parse_midtrain_flag(["train_sft.py", "--midtrain"], {})[0] is True
    assert parse_midtrain_flag(["--midtrain", "model", "3000"], {})[0] is True  # any position
    # both accepted together; pack mode plumbed through
    assert parse_midtrain_flag(["t.py"], {"MIDTRAIN_MASK": "1", "MIDTRAIN_PACK": "seq"}) == (True, "seq")


def test_invalid_pack_mode_rejected():
    with pytest.raises(SystemExit):
        parse_midtrain_flag(["t.py"], {"MIDTRAIN_MASK": "1", "MIDTRAIN_PACK": "naive"})


def test_env_guard_requires_b4_knobs():
    with pytest.raises(SystemExit):
        assert_midtrain_safe(env={}, model_type="qwen3_5", pack_mode="bucket")
    with pytest.raises(SystemExit):  # only one knob set
        assert_midtrain_safe(env={"UNSLOTH_COMPILE_DISABLE": "1"},
                             model_type="qwen3_5", pack_mode="bucket")
    ok = {"UNSLOTH_COMPILE_DISABLE": "1", "UNSLOTH_DISABLE_AUTO_PADDING_FREE": "1"}
    assert_midtrain_safe(env=ok, model_type="qwen3_5", pack_mode="bucket")  # fine


def test_env_guard_refuses_seq_packing_for_gdn():
    ok = {"UNSLOTH_COMPILE_DISABLE": "1", "UNSLOTH_DISABLE_AUTO_PADDING_FREE": "1"}
    for mt in ("qwen3_5", "qwen3_next", "kimi_linear", "olmo_hybrid", "lfm2"):
        with pytest.raises(SystemExit):
            assert_midtrain_safe(env=ok, model_type=mt, pack_mode="seq")
    # full-attention archs may use seq packing
    assert_midtrain_safe(env=ok, model_type="granite4_text", pack_mode="seq")
    assert_midtrain_safe(env=ok, model_type="llama", pack_mode="seq")


def test_linear_attention_list_matches_unsloth_fla_prefixes():
    # mirrors unsloth/models/loader.py FLA_MODEL_TYPE_PREFIXES
    assert set(LINEAR_ATTENTION_MODEL_TYPES) >= {"qwen3_5", "qwen3_next", "kimi_linear", "olmo_hybrid"}


# ---------------------------------------------------------------------------
# completion-only masking
# ---------------------------------------------------------------------------

def test_loss_exactly_on_completion_tokens():
    row = make_row("PROMPTPART", "TARGETPART")
    full, prompt = char_tok(row["text"]), char_tok(row["prompt"])
    labels, k, exact = build_completion_labels(full, prompt)
    assert exact and k == len(prompt)
    assert labels[:k] == [IGNORE_INDEX] * k                       # prompt masked
    assert labels[k:] == full[k:]                                  # loss = target tokens
    assert [t for t in labels if t != IGNORE_INDEX] == char_tok("TARGETPART")
    # the loss tokens are EXACTLY the target tokens (the B8 instrument contract)
    assert sum(1 for t in labels if t != IGNORE_INDEX) == len("TARGETPART")


def test_mask_handles_bpe_seam_straddle():
    # joint tokenization merges across the prompt/target char boundary:
    # full ids differ from prompt ids at the seam -> LCP stops before the
    # merged token, which stays IN the loss (contains completion chars)
    prompt, target = "AB", "CD"
    p_ids, t_ids = char_tok(prompt), char_tok(target)
    merged = [999]  # a token spanning the seam that matches neither
    full = p_ids[:-1] + merged + t_ids
    labels, k, exact = build_completion_labels(full, p_ids)
    assert not exact
    assert k == len(p_ids) - 1                       # common prefix excludes merged token
    assert labels[k] == 999                          # straddling token kept in the loss
    assert labels[k + 1:] == t_ids


def test_empty_prompt_equals_legacy_full_sequence_labels():
    # with an empty prompt the fixed instrument reduces EXACTLY to the legacy
    # full-sequence loss (labels == input_ids) — the anchor tying old/new
    ids = char_tok("anything at all")
    labels, k, exact = build_completion_labels(ids, [])
    assert k == 0 and exact
    assert labels == ids  # legacy behavior byte-for-byte


def test_lcp_len():
    assert lcp_len([1, 2, 3], [1, 2, 4]) == 2
    assert lcp_len([1, 2], [1, 2, 3]) == 2
    assert lcp_len([], [1]) == 0


def test_suffix_route_masks_all_but_target():
    # astfim_v1/fixed composition: no usable prompt prefix, text ends with target
    full = char_tok("CTXSUFFIX" + "THEFILL")
    tgt = char_tok("THEFILL")
    labels, n_masked, exact = build_completion_labels(full, target_ids=tgt)
    assert exact and n_masked == len(full) - len(tgt)
    assert labels[:n_masked] == [IGNORE_INDEX] * n_masked
    assert [t for t in labels if t != IGNORE_INDEX] == tgt


def test_suffix_route_divergent_target_tokenization():
    # standalone target tokenizes differently than in-context (BPE merge of
    # the target head with preceding context): common token suffix stops
    # early, exact=False, only the truly-shared suffix stays in the loss
    full = char_tok("ABCD")                       # [A, B, C, D]
    tgt = [999, 67, 68]                           # 'CD' merged as one token in-context
    labels, n_masked, exact = build_completion_labels(full, target_ids=tgt)
    assert not exact
    assert n_masked == 2                          # only [C, D] matched the tail
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 67, 68]


def test_build_examples_routes_by_composition():
    rows = [
        make_row("pp", "tt"),                        # prefix route (text=prompt+target)
        {"text": "suf" + "fill", "target": "fill"},  # suffix route (no prompt)
        {"text": "neither", "prompt": "zz", "target": "zz"},  # dirty
    ]
    ex, stats = build_midtrain_examples(rows, char_tok, 2048)
    assert stats["route_prefix"] == 1 and stats["route_suffix"] == 1
    assert stats["seam_dirty"] == 1 and stats["rows_out"] == 2
    assert ex[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX] + char_tok("tt")
    assert [t for t in ex[1]["labels"] if t != IGNORE_INDEX] == char_tok("fill")


def test_build_examples_drops_and_counts():
    rows = [
        make_row("pp", "tt"),                                  # ok
        make_row("p" * 30, "t"),                               # too long for cap 10
        {"text": "no prompt or target field"},                 # dropped: no field
        make_row("allprompt", ""),                             # zero loss tokens
    ]
    ex, stats = build_midtrain_examples(rows, char_tok, max_seq_length=10)
    assert stats["rows_in"] == 4 and stats["rows_out"] == 1
    assert stats["dropped_too_long"] == 1
    assert stats["dropped_no_field"] == 1
    assert stats["dropped_no_loss_tokens"] == 1
    assert ex[0]["input_ids"] == char_tok("pptt")
    assert ex[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX] + char_tok("tt")
    assert ex[0]["length"] == 4


def test_build_examples_deterministic_fixed_seed():
    rng = random.Random(3407)
    rows = [make_row("".join(rng.choices("ab", k=rng.randint(1, 8))),
                     "".join(rng.choices("cd", k=rng.randint(1, 8)))) for _ in range(50)]
    a, sa = build_midtrain_examples(rows, char_tok, 2048)
    b, sb = build_midtrain_examples(rows, char_tok, 2048)
    assert a == b and sa == sb  # identical tensors + identical mask sums
    assert sum(1 for e in a for t in e["labels"] if t != IGNORE_INDEX) == \
           sum(1 for e in b for t in e["labels"] if t != IGNORE_INDEX)


# ---------------------------------------------------------------------------
# packing: position ids + attention mask isolation
# ---------------------------------------------------------------------------

def test_pack_ffd_never_splits_and_respects_block():
    rng = random.Random(42)
    ex, _ = build_midtrain_examples(
        [make_row("a" * rng.randint(1, 12), "b" * rng.randint(1, 6)) for _ in range(200)],
        char_tok, 2048)
    blocks, stats = pack_examples_ffd(ex, block_size=16)
    expected_oversize = sum(1 for e in ex if e["length"] > 16)
    assert stats["n_in"] == len(ex) and stats["dropped_oversize"] == expected_oversize
    for blk in blocks:
        assert sum(blk["sample_lengths"]) == len(blk["input_ids"]) <= 16
        assert len(blk["labels"]) == len(blk["input_ids"])
    # every sample landed whole: multiset of lengths preserved (non-oversize)
    got = sorted(l for blk in blocks for l in blk["sample_lengths"])
    assert got == sorted(e["length"] for e in ex if e["length"] <= 16)
    # deterministic
    blocks2, _ = pack_examples_ffd(ex, block_size=16)
    assert blocks == blocks2


def test_pack_ffd_drops_oversize():
    ex = [{"input_ids": [1] * 20, "labels": [1] * 20, "length": 20}]
    blocks, stats = pack_examples_ffd(ex, block_size=8)
    assert blocks == [] and stats["dropped_oversize"] == 1


def test_position_ids_restart_per_sample():
    pos = build_packed_position_ids([3, 2, 4], padded_len=12)
    assert pos == [0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 0, 0]
    with pytest.raises(ValueError):
        build_packed_position_ids([5, 5], padded_len=8)  # exceeds padded len


def test_attention_mask_no_cross_sample_leakage():
    # brute-force oracle on randomized packings: mask[i][j] is True iff i,j
    # are in the SAME sample, j <= i (causal), and neither is padding.
    rng = random.Random(3407)
    for _ in range(25):
        lens = [rng.randint(1, 9) for _ in range(rng.randint(1, 6))]
        total = sum(lens)
        if total > 40:
            continue
        pad = rng.randint(0, 5)
        mask = build_packed_attn_mask_4d(lens, total + pad)
        bounds, s = [], 0
        for n in lens:
            bounds.append((s, s + n))
            s += n
        def sample_of(i):
            if i >= total:
                return -1  # padding
            for k, (a, b) in enumerate(bounds):
                if a <= i < b:
                    return k
            raise AssertionError
        for i in range(total + pad):
            for j in range(total + pad):
                should = (sample_of(i) >= 0 and sample_of(i) == sample_of(j) and j <= i)
                assert mask[i][j] is should, (lens, pad, i, j)
        # self-attention always allowed on real positions; pads attend nowhere
        for i in range(total):
            assert mask[i][i] is True
        for i in range(total, total + pad):
            assert not any(mask[i])


def test_attention_mask_two_samples_no_leak():
    mask = build_packed_attn_mask_4d([2, 3], 5)
    T, F = True, False
    assert mask == [[T, F, F, F, F],
                    [T, T, F, F, F],
                    [F, F, T, F, F],
                    [F, F, T, T, F],
                    [F, F, T, T, T]]


def test_collator_tensors():
    import torch
    blocks = [{"input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8],
               "sample_lengths": [2, 2]},
              {"input_ids": [9, 1], "labels": [9, 1], "sample_lengths": [2]}]
    out = MidtrainPackedCollator(pad_token_id=0)(blocks)
    assert out["input_ids"].shape == out["labels"].shape == (2, 4)
    assert out["position_ids"].tolist() == [[0, 1, 0, 1], [0, 1, 0, 0]]
    assert out["attention_mask"].shape == (2, 1, 4, 4)
    assert out["attention_mask"].dtype == torch.bool
    # padded label positions masked
    assert out["labels"][1, 2:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    # mask row for the padded batch entry 1: only its own causal 2x2 block
    m = out["attention_mask"][1, 0]
    assert m[0, :2].tolist() == [True, False] and not m[0, 2:].any()
    assert m[1, :2].tolist() == [True, True] and not m[1, 2:].any()
    assert not m[2:].any() and not m[:, 2:].any()
    # position ids of padded slots are 0
    assert out["position_ids"][1, 2:].tolist() == [0, 0]


def test_collator_handles_unpacked_eval_rows():
    import torch
    rows = [{"input_ids": [3, 4, 5], "labels": [-100, 4, 5]},   # no sample_lengths
            {"input_ids": [6], "labels": [6]}]
    out = MidtrainPackedCollator(pad_token_id=0)(rows)
    assert out["position_ids"][0].tolist() == [0, 1, 2]
    m = out["attention_mask"][0, 0]
    assert m[0].tolist() == [True, False, False]
    assert m[1].tolist() == [True, True, False]
    assert m[2].tolist() == [True, True, True]


def test_collator_rejects_corrupted_block():
    with pytest.raises(ValueError):
        MidtrainPackedCollator()([{"input_ids": [1, 2, 3], "labels": [1, 2, 3],
                                   "sample_lengths": [2]}])


# ---------------------------------------------------------------------------
# flag-off byte-compatibility contract (source-pinned)
# ---------------------------------------------------------------------------

def test_train_sft_off_path_pinned():
    src = TRAIN_SFT.read_text()
    # legacy dataset map, verbatim, exactly once, inside the MIDTRAIN else-branch
    legacy = ('else:\n    ds = ds.map(lambda x: {"text": x["text"]}, remove_columns=[\n'
              '        c for c in ds["train"].column_names if c != "text"])')
    assert src.count(legacy) == 1
    # legacy train/eval selection lines verbatim (shuffle seed 42 + 48k cap)
    assert 'ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))' in src
    assert 'ds["eval"].select(range(500))' in src
    # SFTConfig literals unchanged; midtrain kwargs inject ONLY through ** spread
    assert "max_seq_length=2048, **midtrain_kwargs)" in src
    assert "midtrain_kwargs, midtrain_collator = {}, None" in src
    assert 'midtrain_kwargs = {"train_sampling_strategy": "group_by_length"' in src
    # the flag is strictly opt-in
    assert '("--midtrain" in argv) or (env.get("MIDTRAIN_MASK", "") == "1")' in \
        (REPO / "experiments" / "training" / "midtrain_data.py").read_text()


def test_train_sft_midtrain_branch_guards_before_data_prep():
    src = TRAIN_SFT.read_text()
    assert src.index("parse_midtrain_flag(_ARGV)") < src.index("if MIDTRAIN:")
    assert "assert_midtrain_safe" in src  # B4-knob + GDN guards active on the ON path


def test_train_sft_midtrain_arg_position_independent():
    # the argv snapshot/strip at the top keeps positional args intact wherever
    # --midtrain appears
    src = TRAIN_SFT.read_text()
    assert '_ARGV = list(sys.argv)' in src
    assert 'sys.argv.remove("--midtrain")' in src
    argv = ["train_sft.py", "--midtrain", "model", "3000", "data"]
    argv2 = argv[:]
    argv2.remove("--midtrain")
    assert argv2[1:3] == ["model", "3000"]  # positional parse unaffected


# ---------------------------------------------------------------------------
# real corpus + real tokenizer (skip when artifacts absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QWEN_TOK_DIR.exists() or not ASTFIM_EVAL.exists(),
                    reason="local qwen3.5-2b tokenizer or astfim_v1 eval.jsonl not present")
def test_real_astfim_rows_mask_exactly():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN_TOK_DIR), local_files_only=True)
    n, loss_tok, tot_tok = 0, 0, 0
    with open(ASTFIM_EVAL) as f:
        for line in f:
            if n >= 12:
                break
            row = json.loads(line)
            assert row["text"] == row["prompt"] + row["target"]  # corpus invariant
            full = tok(row["text"])["input_ids"]
            p = tok(row["prompt"])["input_ids"]
            labels, k, exact = build_completion_labels(full, prompt_ids=p)
            assert exact, "real seam must be token-clean (prompt ends <|end|>\\n)"
            keep = [i for i, t in enumerate(labels) if t != IGNORE_INDEX]
            assert tok.decode([full[i] for i in keep]) == row["target"]
            loss_tok += len(keep)
            tot_tok += len(full)
            n += 1
@pytest.mark.skipif(not QWEN_TOK_DIR.exists() or not ASTFIM_EVAL.exists(),
                    reason="local tokenizer or astfim_v1 eval.jsonl not present")
def test_real_wiring_pipeline_astfim_root():
    # end-to-end wiring check: the same map -> guard -> filter chain
    # train_sft.py MIDTRAIN mode runs, on the real corpus + real tokenizer
    import datasets
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN_TOK_DIR), local_files_only=True)
    raw = datasets.load_dataset(
        "json", data_files={"e": str(ASTFIM_EVAL)}, split="e").select(range(24))
    tokd = raw.map(lambda r: midtrain_map_row(r, lambda t: tok(t)["input_ids"]),
                   remove_columns=raw.column_names)
    guard = seam_guard(tokd, "eval")
    assert guard["n_routed"] == 24 and guard["n_exact"] == 24
    kept = tokd.filter(lambda x: x["length"] <= 2048 and x["n_loss"] > 0)
    n_loss, n_tok = sum(kept["n_loss"]), sum(kept["length"])
    assert 0.05 < n_loss / n_tok < 0.40  # completion share in the expected band


ASTFIM_FIXED = Path("/mnt/h/sepalith/datasets/astfim_v1/fixed/eval.jsonl")


def test_wiring_pipeline_on_stub_rows():
    # the ACTUAL train_sft MIDTRAIN wiring primitives through datasets.map
    import datasets
    rows = [make_row("p" * i, "t" * (i % 3 + 1)) for i in range(1, 9)]
    raw = datasets.Dataset.from_list(rows)
    tokd = raw.map(lambda r: midtrain_map_row(r, char_tok),
                   remove_columns=raw.column_names)
    guard = seam_guard(tokd, "train")
    assert guard["n_routed"] == 8 and guard["n_exact"] == 8
    kept = tokd.filter(lambda x: x["length"] <= 2048 and x["n_loss"] > 0)
    assert len(kept) == 8
    for r in kept:
        assert r["route"] == 0  # prefix route for text=prompt+target rows


def test_seam_guard_aborts_on_dirty_corpus():
    import datasets
    rows = [make_row("pp", "tt")] + [{"text": "zzz", "prompt": "q", "target": "q"}] * 3
    raw = datasets.Dataset.from_list(rows)
    tokd = raw.map(lambda r: midtrain_map_row(r, char_tok),
                   remove_columns=raw.column_names)
    with pytest.raises(SystemExit):  # 75% dirty > 5% threshold
        seam_guard(tokd, "train")
    g = seam_guard(tokd, "train", allow_dirty=True)
    assert g["n_routed"] == 1 and g["dirty_rate"] == 0.75


@pytest.mark.skipif(not QWEN_TOK_DIR.exists() or not ASTFIM_FIXED.exists(),
                    reason="local qwen3.5-2b tokenizer or astfim_v1/fixed eval.jsonl not present")
def test_real_astfim_fixed_rows_mask_exactly_via_suffix_route():
    # the fixed/ corpus drops the '<|end|>\\n' separator, so prompt is NOT a
    # char-prefix of text — the suffix route must mask all but the target
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN_TOK_DIR), local_files_only=True)
    n, loss_tok, tot_tok = 0, 0, 0
    with open(ASTFIM_FIXED) as f:
        for line in f:
            if n >= 8:
                break
            row = json.loads(line)
            assert not row["text"].startswith(row["prompt"])  # fixed-shape invariant
            assert row["text"].endswith(row["target"])
            full = tok(row["text"])["input_ids"]
            labels, k, exact = build_completion_labels(
                full, target_ids=tok(row["target"])["input_ids"])
            assert exact, "fixed-corpus target seam must be token-clean"
            keep = [i for i, t in enumerate(labels) if t != IGNORE_INDEX]
            assert tok.decode([full[i] for i in keep]) == row["target"]
            loss_tok += len(keep)
            tot_tok += len(full)
            n += 1
    assert n >= 1 and loss_tok < tot_tok
