"""Tests for experiments/eval/h3_s1_adapt.py (H3-S1 adaptation pack).

Run:  cd <worktree root> && \
      uv run --with pytest python -m pytest tests/test_h3_s1.py -q

Pure-CPU, deterministic, no network, no serving, no GPU, no NAS. Covers the
four load-bearing contracts of the H3-S1 task:
  1. arm selection (3 marker-reusing render families, twins collapsed,
     the zeta1 control excluded) and arm-config generation (the exact
     train_sft.py env-knob surface, steps math, gate arithmetic);
  2. variant-render determinism on corpus-recovered examples: every staged
     row's prompt re-parses and re-renders byte-identically, the original
     cursor encoding survives (CE1 keeps the bare marker line), and CE2
     regions pass through byte-exact;
  3. bench scoring math: the ledger->matrix summarize (pass_n = n - fails,
     the "ok" reason never counted as a failure — the S0 tally regression)
     and the transfer table against the embedded S0 zero-shot column;
  4. prepare on a miniature corpus (shape A/B/alpaca/corrupt mix): staging,
     skip accounting, manifest shape, byte-determinism across reruns; plus
     the --smoke entry point end to end.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "experiments" / "eval" / "h3_s1_adapt.py"
_spec = importlib.util.spec_from_file_location("h3_s1_adapt", _SCRIPT)
s1 = importlib.util.module_from_spec(_spec)
sys.modules["h3_s1_adapt"] = s1
_spec.loader.exec_module(s1)

sys.path.insert(0, str(_SCRIPT.parent))
import render_variants as rv  # noqa: E402  (the frozen S0 registry)

# a corpus-shaped zeta2 example with history (shape A, CE2)
EX_A = dict(rv.fixture_small())
PROMPT_A = rv.render_zeta2(EX_A)
TARGET_A = "\n".join(EX_A["region_new"]).rstrip() + "\n" + rv.TERMINATOR
ROW_A = dict(text=PROMPT_A + TARGET_A, prompt=PROMPT_A, target=TARGET_A,
             family="edit_pairs", package_or_repo="fixture/pkg", has_types=False)


def row_shape_b():      # the comment_to_code/finish_block builder shape (CE1)
    parts = [rv.FIM_SUFFIX, s1.FIM_PREFIX_FILENAME + "R/util.R",
             "# TODO: handle NA", rv.REGION_OPEN, rv.CURSOR2,
             rv.REGION_CLOSE, rv.FIM_MIDDLE]
    prompt = "\n".join(parts)
    target = "  y <- mean(x, na.rm = TRUE)\n" + rv.TERMINATOR
    return dict(text=prompt + target, prompt=prompt, target=target,
                family="comment_to_code_real", package_or_repo="fixture/pkg",
                has_types=False)


# ------------------------------------------------------------- arm selection

def test_arm_selection_rule():
    # exactly the three marker-reusing render families; twins collapsed
    assert s1.ARMS == ["v10_psmtail_merge", "v05_psm_merge",
                       "v14_psm_merge_nohist"]
    marker_reusing = {"v10_psmtail_merge", "v11_psmtail_merge_empty",
                      "v05_psm_merge", "v06_psm_merge_empty",
                      "v14_psm_merge_nohist", "zeta2"}
    assert set(s1.ARMS) <= marker_reusing
    assert "v13_zeta1_alpaca" not in s1.ARMS          # must-FAIL control
    assert "zeta2" not in s1.ARMS                     # baseline, not an arm
    # cursor-encoding twins collapse into their parent arm (matrix §5)
    assert "v06_psm_merge_empty" not in s1.ARMS
    assert "v11_psmtail_merge_empty" not in s1.ARMS
    for arm in s1.ARMS:
        assert rv.VARIANT_INFO[arm]["vocab"] == "MV1"
        assert rv.k2_verdict(arm)[0]                  # K2 paper rule holds
        assert s1.S0_ZERO_SHOT[arm] >= 99             # marker-reusing margin
    # the v05 family holds the WORST marker-reusing zero-shot margin (v06)
    assert s1.S0_ZERO_SHOT["v06_psm_merge_empty"] == 99


def test_arm_config_surface():
    cfgs = {a: s1.arm_config(a, Path("/tmp/h3s1run")) for a in s1.ARMS}
    for arm, cfg in cfgs.items():
        # the trainer is reused verbatim: python -u train_sft.py positional args
        assert cfg["argv"][2].endswith("experiments/training/train_sft.py")
        assert cfg["argv"][3] == str(s1.BASE_MODEL)      # adapt the b4 merge
        assert cfg["argv"][4] == str(s1.STEPS)           # 150
        assert cfg["argv"][5] == cfg["data_dir"]
        assert cfg["argv"][6] == cfg["out_dir"]
        assert cfg["argv"][7] == ""                      # no resume
        # env-knob surface: geometry HALF the banked b4, banked everything else
        env = cfg["env"]
        assert env["SFT_LORA_R"] == "16" and env["SFT_LORA_ALPHA"] == "32"
        assert env["SFT_TARGETS"] == s1.B4_TARGETS
        assert env["SFT_EVAL_STEPS"] == str(s1.STEPS)
        assert "SFT_LR" not in env                       # banked 2e-4 default
        assert "SFT_PD_BATCH" not in env                 # banked 4x4 default
        # gate arithmetic: linear in r -> exactly half the measured b4 line
        assert cfg["expect_trainable"] * 2 == s1.EXPECT_TRAINABLE_B4
        assert cfg["expect_trainable"] == 10911744
        # export reuses the repo exporter, Q8_0 stem under experiments/models
        assert cfg["export_argv"][1].endswith("export_gguf.py")
        assert cfg["gguf"].endswith(f"h3s1_{arm.split('_')[0]}-Q8_0.gguf")
        # the exact shell line round-trips into the argv
        line = s1.train_command(cfg)
        assert "SFT_LORA_R='16'" in line and "SFT_LORA_ALPHA='32'" in line
        assert str(s1.STEPS) in line and str(s1.BASE_MODEL) in line


def test_steps_epoch_math():
    # 150 steps x effective 16 = 2400 rows = 0.2 epoch of the 12k selection
    assert s1.STEPS * s1.EFFECTIVE_BATCH == 2400
    assert s1.EPOCH_FRAC == pytest.approx(0.2)
    assert s1.STEPS * s1.EFFECTIVE_BATCH == int(s1.ROW_CAP * s1.EPOCH_FRAC)


# ------------------------------------------------ corpus inversion + renders

def test_parse_shape_a_is_the_frozen_s0_inverse():
    ex, shape = s1.parse_corpus_prompt(PROMPT_A)
    assert shape == "A"
    ref = rv.parse_prompt(PROMPT_A, "zeta2")            # byte-exact reuse
    for k in ("path", "prefix", "suffix", "history", "region_old",
              "cursor_idx", "cursor_encoding"):
        assert ex[k] == ref[k]
    assert s1.zeta2_identity(ex) == PROMPT_A            # lossless gate


def test_parse_shape_b_header_carries_the_path():
    row = row_shape_b()
    ex, shape = s1.parse_corpus_prompt(row["prompt"])
    assert shape == "B"
    assert ex["path"] == "R/util.R"
    assert ex["prefix"] == ["# TODO: handle NA"]
    assert ex["suffix"] == [] and ex["history"] == []
    assert ex["cursor_encoding"] == "CE1"
    assert s1.zeta2_identity(ex) == row["prompt"]


def test_non_renderable_rows_raise():
    for prompt in ("### Instruction:\nfix\n\n### Response:\n\n",
                   "plain text with no markers", ""):
        with pytest.raises(ValueError):
            s1.parse_corpus_prompt(prompt)
    # a marker-bearing but corrupt prompt (no region close)
    with pytest.raises(ValueError):
        s1.parse_corpus_prompt(PROMPT_A.replace(rv.REGION_CLOSE + "\n", ""))


@pytest.mark.parametrize("arm", s1.ARMS)
def test_render_determinism_and_roundtrip(arm):
    for row in (ROW_A, row_shape_b()):
        ex, _ = s1.recover_row(row)
        p1 = s1.render_arm_example(ex, arm)
        assert p1 == s1.render_arm_example(ex, arm)     # deterministic
        back = rv.parse_prompt(p1, arm)                 # re-parse ...
        assert s1.render_arm_example(back, arm) == p1   # ... re-render == id
        staged = s1.arm_row(row, ex, arm)
        assert staged["text"] == staged["prompt"] + staged["target"]
        assert staged["target"] == row["target"]        # verbatim target
        assert staged["h3s1"]["variant"] == arm


@pytest.mark.parametrize("arm", s1.ARMS)
def test_cursor_encoding_survives(arm):
    exb, _ = s1.recover_row(row_shape_b())
    pb = s1.render_arm_example(exb, arm)
    assert "\n" + rv.CURSOR2 + "\n" + rv.REGION_CLOSE in pb   # CE1 signature
    assert pb.count(rv.CURSOR2) == 1
    exa, _ = s1.recover_row(ROW_A)
    pa = s1.render_arm_example(exa, arm)
    inline = [l for l in pa.split("\n") if l.endswith(rv.CURSOR2)]
    assert len(inline) == 1 and inline[0] != rv.CURSOR2      # CE2 inline


def test_variant_axes_in_arm_renders():
    exa, _ = s1.recover_row(ROW_A)                       # history present
    p10 = s1.render_arm_example(exa, "v10_psmtail_merge")
    p05 = s1.render_arm_example(exa, "v05_psm_merge")
    p14 = s1.render_arm_example(exa, "v14_psm_merge_nohist")
    # v10 keeps the incumbent's generation tail byte-identical (S0 §1.2a)
    tail = lambda p: p[p.index(rv.REGION_OPEN):]        # noqa: E731
    assert tail(p10) == tail(rv.render_zeta2(EX_A))
    # HP2/HP3 always emit the history header; HP0 never does
    assert p10.count(rv.HIST_HEADER) == 1 and p05.count(rv.HIST_HEADER) == 1
    assert rv.HIST_HEADER not in p14
    # same marker multiset -> identical byte length for v05/v10
    assert len(p05) == len(p10)


# ------------------------------------------------------------- bench scoring

def _ledger(model_spec):
    rows = []
    for model, spec in model_spec.items():
        for variant, (fails, reasons) in spec.items():
            for i in range(100):
                fail = i < fails
                rows.append(dict(model=model, variant=variant, case=i,
                                 fail=fail,
                                 reason=(reasons[i] if fail else "ok")))
    return rows


def test_summarize_counts_ok_as_pass_not_fail():
    rows = _ledger({"m": {"v10_psmtail_merge": (2, ["no_markers", "unparseable"])}})
    cell = s1.summarize_ledger(rows)["m"]["v10_psmtail_merge"]
    assert cell["n"] == 100
    assert cell["format_fail"] == 2 and cell["pass_n"] == 98
    assert cell["fail_reasons"] == {"no_markers": 1, "unparseable": 1}
    # the S0 live-tally regression: a truthy "ok" key is NOT a failure
    assert sum(cell["fail_reasons"].values()) == cell["format_fail"]


def test_transfer_table_against_s0_reference():
    spec = {"b4_baseline": {v: (0 if s1.S0_ZERO_SHOT[v] == 100 else 1,
                                ["no_markers"]) for v in s1.BENCH_VARIANTS}}
    for arm in s1.ARMS:        # diagonal perfect, zeta2 backward compat 97
        spec[f"h3s1_{arm.split('_')[0]}"] = {arm: (0, []), "zeta2": (3, ["x"] * 3)}
    table = s1.transfer_table(s1.summarize_ledger(_ledger(spec)))
    t10 = table["v10_psmtail_merge"]
    assert t10["diagonal_pass"] == 100 and t10["backward_zeta2_pass"] == 97
    assert t10["s0_zeroshot"] == s1.S0_ZERO_SHOT["v10_psmtail_merge"] == 100
    assert t10["diagonal_minus_s0"] == 0
    assert t10["b4_zeroshot_recheck"] == 100
    t05 = table["v05_psm_merge"]
    assert t05["b4_zeroshot_recheck"] == 100         # S0 re-verified in-run


def test_gate_attachment_math():
    log = "Trainable parameters = 10,911,744 of 1,903,648,576 (0.57%)"
    s1._gate_attachment(log, 10911744)
    with pytest.raises(SystemExit):
        s1._gate_attachment(log, 21823488)
    with pytest.raises(SystemExit):
        s1._gate_attachment("no line here", 10911744)


# ------------------------------------------------------------------- prepare

def _mini_corpus(path):
    rows = [ROW_A, row_shape_b(),
            dict(text="x", prompt="### Instruction:\nfix\n### Response:\n\n",
                 target="x <- 1\n", family="hidden_r_instruction",
                 package_or_repo="p", has_types=False),
            dict(text="y", prompt="garbage", target="y", family="broken",
                 package_or_repo="p", has_types=False)]
    for split, rep in (("train", 10), ("eval", 1)):
        with (path / f"{split}.jsonl").open("w") as f:
            for _ in range(rep):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    return rows


def test_prepare_stages_skips_and_manifest(tmp_path):
    corpus = tmp_path / "sft_v3"
    corpus.mkdir()
    _mini_corpus(corpus)
    run = tmp_path / "run"
    m = s1.prepare(run, corpus, row_cap=15, eval_cap=2)
    assert m["counts"] == dict(train_pool=20, train_staged=15, eval_staged=2)
    st = m["stats"]
    assert st["skip:train:hidden_r_instruction:unparseable"] == 10
    assert st["skip:train:broken:unparseable"] == 10
    assert st["keep:train:edit_pairs"] == 10
    assert st["enc:CE1:B"] == 11 and st["enc:CE2:A"] == 11
    assert m["lora"] == dict(r=16, alpha=32, expect_trainable=10911744)
    assert m["epoch_frac"] == pytest.approx(s1.STEPS * s1.EFFECTIVE_BATCH / 15)
    for arm in s1.ARMS:
        d = run / f"data_{arm.split('_')[0]}"
        staged = [json.loads(l) for l in (d / "train.jsonl").read_text().splitlines()]
        assert len(staged) == 15
        assert {r["family"] for r in staged} <= {"edit_pairs",
                                                 "comment_to_code_real"}
        for r in staged:                # every staged row re-parses under
            back = rv.parse_prompt(r["prompt"], arm)   # its OWN variant and
            assert s1.render_arm_example(back, arm) == r["prompt"]


def test_prepare_is_byte_deterministic(tmp_path):
    corpus = tmp_path / "sft_v3"
    corpus.mkdir()
    _mini_corpus(corpus)
    a, b = tmp_path / "runA", tmp_path / "runB"
    s1.prepare(a, corpus, row_cap=7, eval_cap=2)
    s1.prepare(b, corpus, row_cap=7, eval_cap=2)
    for arm in s1.ARMS:
        for split in ("train", "eval"):
            fa = (a / f"data_{arm.split('_')[0]}" / f"{split}.jsonl").read_bytes()
            fb = (b / f"data_{arm.split('_')[0]}" / f"{split}.jsonl").read_bytes()
            assert fa == fb
    ma = json.loads((a / "prepare_manifest.json").read_text())
    mb = json.loads((b / "prepare_manifest.json").read_text())
    for m in (ma, mb):        # run dirs differ by design; content must not
        m.pop("arm_dirs")
    assert ma == mb


def test_cli_smoke_entry_point():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--smoke"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TRAIN:" in r.stdout
    assert "SFT_LORA_R='16'" in r.stdout
    payload = [l for l in r.stdout.splitlines() if l.startswith("TRAIN:")]
    assert len(payload) == len(s1.ARMS)


def test_cli_train_dry_run_needs_prepare(tmp_path):
    r = subprocess.run([sys.executable, str(_SCRIPT), "train",
                        "--run", str(tmp_path / "nope"), "--dry-run"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0 and "prepare first" in r.stderr
