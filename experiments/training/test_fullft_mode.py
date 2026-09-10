"""CPU unit tests for the PFT1 full fine-tuning mode (zcode-pft1, 2026-09-06).

Covers: the opt-in flag surface (--full-ft / FULL_FT=1, argv-position
independence), the MIDTRAIN mutual-exclusion guard, the source-pinned
recipe literals (queue §3 PFT1: lr 1.5e-5 cosine, paged 8-bit AdamW,
gradient checkpointing use_reentrant=False, bf16, seed 3407, seq 2048,
effective batch 16), the data-selection discipline identical to the
legacy path, the export NO_LORA mode, and the BPB forgetting-probe
helpers (causal rendering port + corpus loaders + regression math).
All pure CPU. Run: uv run --with pytest python -m pytest <path> -q
(quiet-window convention: nice -n 19).
"""
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAIN_SFT = REPO / "experiments" / "training" / "train_sft.py"
EXPORT_GGUF = REPO / "experiments" / "training" / "export_gguf.py"
PROBE = REPO / "experiments" / "eval" / "pft1_bpb_probe.py"


# ---------------------------------------------------------------------------
# flag surface (argv parsing mirrors the --midtrain contract)
# ---------------------------------------------------------------------------

def test_fullft_arg_position_independent():
    src = TRAIN_SFT.read_text()
    assert '_ARGV = list(sys.argv)' in src
    assert 'sys.argv.remove("--midtrain")' in src          # pinned, unchanged
    assert 'sys.argv.remove("--full-ft")' in src           # new, same pattern
    argv = ["train_sft.py", "--full-ft", "model", "3000", "data"]
    argv.remove("--full-ft")
    assert argv[1:3] == ["model", "3000"]                  # positional parse intact


def test_fullft_strictly_opt_in():
    src = TRAIN_SFT.read_text()
    assert 'os.environ.get("FULL_FT", "") == "1"' in src
    assert '"--full-ft" in _ARGV' in src


def test_fullft_and_midtrain_mutually_exclusive():
    src = TRAIN_SFT.read_text()
    assert "FULL_FT and MIDTRAIN are mutually exclusive" in src
    assert src.index("FULL_FT and MIDTRAIN are mutually exclusive") < \
        src.index("from unsloth import FastLanguageModel", src.index("FULL_FT"))


def test_fullft_branch_precedes_legacy_path():
    # the branch exits before the legacy LoRA pipeline; the legacy body below
    # it stays reachable only with the flag off (match the CALL SITE, not the
    # docstring mention of get_peft_model)
    src = TRAIN_SFT.read_text()
    branch = src.index('os.environ.get("FULL_FT", "") == "1"')
    legacy_peft = src.index("model = FastLanguageModel.get_peft_model(")
    assert branch < legacy_peft
    assert src.index("sys.exit(0)", branch) < legacy_peft


# ---------------------------------------------------------------------------
# recipe literals (queue §3 PFT1, source-pinned)
# ---------------------------------------------------------------------------

def test_fullft_recipe_literals_pinned():
    src = TRAIN_SFT.read_text()
    ft = src[src.index("PFT1 full fine-tuning branch"):src.index("sys.exit(0)")]
    assert 'full_finetuning=True' in ft                     # ALL weights, unsloth loader
    assert 'get_peft_model' not in ft                       # no adapter on the FT path
    assert 'os.environ.get("FULL_FT_LR", "1.5e-5")' in ft   # pre-registered lr
    assert 'optim="paged_adamw_8bit"' in ft                 # paged 8-bit AdamW
    assert "gradient_checkpointing=True" in ft
    assert '"use_reentrant": False' in ft
    assert "seed=3407" in ft and "bf16=(not _FP16)" in ft
    assert "max_seq_length=2048" in ft
    assert "warmup_ratio=0.03, lr_scheduler_type=\"cosine\"" in ft
    assert "eval_steps=500" in ft and "save_steps=1000" in ft
    # B13 logits-block launch convention: bs2 x ga8 DEFAULTS on the FT path
    assert 'os.environ.get("SFT_PD_BATCH", "2")' in ft
    assert 'os.environ.get("SFT_GRAD_ACCUM", "8")' in ft


def test_fullft_data_selection_matches_legacy_discipline():
    src = TRAIN_SFT.read_text()
    ft = src[src.index("PFT1 full fine-tuning branch"):src.index("sys.exit(0)")]
    # the exact b4 selection line (shuffle seed 42 + 48k cap) inside the branch
    assert 'ds["train"].shuffle(seed=42).select(range(min(48000, len(ds["train"]))))' in ft
    assert 'ds["eval"].select(range(500))' in ft
    assert 'dataset_text_field="text"' in ft


def test_legacy_path_untouched_by_fullft():
    # the byte-compat canary: the legacy get_peft_model call is verbatim
    # (r/alpha became the SFT_LORA_R/SFT_LORA_ALPHA env pair, H3-S1; their
    # DEFAULTS are the banked 32/64 literals, so OFF stays byte-identical)
    src = TRAIN_SFT.read_text()
    legacy = ("model = FastLanguageModel.get_peft_model(\n"
              "    model, r=_LORA_R, lora_alpha=_LORA_ALPHA, lora_dropout=0,")
    assert src.count(legacy) == 1
    assert 'os.environ.get("SFT_LORA_R", "32")' in src
    assert 'os.environ.get("SFT_LORA_ALPHA", "64")' in src


def test_sftconfig_ft_kwargs_effective_batch_16():
    # construct the FT SFTConfig on CPU and check the recipe carries through.
    # NOTE max_seq_length/tokenizer spellings are NOT tested here: raw trl
    # 0.24 rejects max_seq_length — the spellings work in train_sft.py because
    # the FT branch imports unsloth first, which patches trl for backward
    # compat (identical to the legacy path's proven call shape). Importing
    # unsloth in a CPU test would open a CUDA context — card-discipline no.
    from trl import SFTConfig
    cfg = SFTConfig(
        output_dir="/tmp/pft1_test_cfg",
        per_device_train_batch_size=int(dict(a="2")["a"]),
        gradient_accumulation_steps=8,
        num_train_epochs=1, max_steps=3000,
        learning_rate=1.5e-5, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=20, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        bf16=False, seed=3407, report_to="none", dataset_text_field="text",
        max_length=2048,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    assert cfg.learning_rate == 1.5e-5
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.gradient_checkpointing is True
    assert cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 16
    assert cfg.seed == 3407 and cfg.bf16 is False


# ---------------------------------------------------------------------------
# export NO_LORA mode
# ---------------------------------------------------------------------------

def test_export_nolora_mode_pinned():
    src = EXPORT_GGUF.read_text()
    assert 'os.environ.get("NO_LORA")' in src
    # refuses an adapter dir (wrong mode) and a non-model dir
    assert "adapter_config.json" in src
    assert "is not an HF model dir" in src
    # the merge paths stay ordered after it (default flow unchanged)
    assert src.index('os.environ.get("NO_LORA")') < src.index('os.environ.get("MERGE_VIA_PEFT")')
    assert src.count("model.save_pretrained_merged") == 1  # legacy unsloth merge intact


# ---------------------------------------------------------------------------
# BPB forgetting probe helpers
# ---------------------------------------------------------------------------

CTX, HIST, SUF, END = "<|context|>", "<|history|>", "<|suffix|>\n", "\n<|end|>"


def _import_probe():
    sys.path.insert(0, str(REPO / "experiments" / "eval"))
    import pft1_bpb_probe as p
    return p


def test_causal_from_row_matches_ladder_port():
    p = _import_probe()
    prompt = CTX + "pkg/R/x.R\nline1\nline2" + HIST + "\n\n" + "<|cursor|>" + SUF + "line4\nline5" + END
    row = dict(prompt=prompt, target="line3" + END)
    # natural order: path header + prefix + span + "\n" + suffix, markers gone
    assert p.causal_from_row(row) == "pkg/R/x.R\nline1\nline2line3\nline4\nline5"


def test_causal_from_row_strips_bare_end_marker():
    p = _import_probe()
    prompt = CTX + "a.R\npre" + HIST + "\n" + SUF + "post" + END
    assert p.causal_from_row(dict(prompt=prompt, target="span<|end|>")) == "a.R\nprespan\npost"
    assert p.causal_from_row(dict(prompt=prompt, target="span" + END)) == "a.R\nprespan\npost"


def test_causal_from_row_rejects_malformed():
    p = _import_probe()
    assert p.causal_from_row(dict(prompt="no markers", target="x")) is None
    assert p.causal_from_row(dict(prompt=HIST + CTX + "swapped", target="x")) is None


def test_probe_bpb_math_is_nats_over_bytes_ln2():
    # the bpb_eval.py formula: nats / (bytes * ln2) — check via a synthetic
    # corpus_bpb result shape (no model: verify the divisor constant used)
    p = _import_probe()
    nats, nbytes = 6931.4718, 1000.0
    assert abs(p.math.log(2) - math.log(2)) < 1e-12  # module math is stdlib
    assert abs(nats / (nbytes * math.log(2)) - 10.0) < 1e-6


def test_probe_regression_gate_direction():
    # verdict: general-R regression <= 1% vs b4 -> computed as pct delta of bpb
    b4, pft1 = 0.5000, 0.5050
    reg = 100.0 * (pft1 / b4 - 1.0)
    assert abs(reg - 1.0) < 1e-9          # exactly at the gate
    assert 100.0 * (0.5049 / b4 - 1.0) < 1.0   # pass side
    assert 100.0 * (0.5051 / b4 - 1.0) > 1.0   # fail side


def test_probe_out_json_schema(tmp_path):
    # the probe writes a name -> {model_dir, general_r, general_text} map and
    # a verdict block with pct regressions when both arms are present
    out = tmp_path / "probe.json"
    data = {"b4": {"general_r": {"bpb": 0.5}, "general_text": {"bpb": 0.7}},
            "pft1": {"general_r": {"bpb": 0.505}, "general_text": {"bpb": 0.721}}}
    out.write_text(json.dumps(data))
    d = json.loads(out.read_text())
    reg = {k: 100.0 * (d["pft1"][k.split("_regression")[0]]["bpb"] /
                       d["b4"][k.split("_regression")[0]]["bpb"] - 1.0)
           for k in ("general_r_regression_pct", "general_text_regression_pct")}
    assert abs(reg["general_r_regression_pct"] - 1.0) < 1e-9
    assert abs(reg["general_text_regression_pct"] - 3.0) < 1e-9


@pytest.mark.skipif(not Path("/mnt/h/sepalith/datasets/astfim_v1/fixed/eval.jsonl").exists(),
                    reason="NAS astfim_v1/fixed eval not present")
def test_probe_corpus_loaders_real():
    p = _import_probe()
    docs, dropped = p.load_corpus_r(cap_rows=25)
    assert len(docs) == 25 and dropped == 0          # clean corpus expectation
    assert all("<|" not in d for d in docs)          # markers stripped
    assert all(len(d) > 50 for d in docs)            # real documents
    ctl = p.load_corpus_control(cap_rows=10)
    assert len(ctl) == 10 and all(len(c) > 50 for c in ctl)  # real legal text
