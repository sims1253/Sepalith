#!/usr/bin/env python3
"""H3-S1 render-adaptation arms: re-render the sft_v3 corpus in the three
surviving variant orderings, adapt b4 with a 0.2-epoch LoRA per arm (via the
banked trainer, env knobs only), and re-run the S0 compliance bench per
adapted model plus the baseline cross-variant transfer matrix.

Inputs (all frozen): experiments/eval/render_variants.py (S0 registry:
renders, markers, stop strings, format_fail), experiments/training/
train_sft.py (the b4 LoRA trainer; SFT_TARGETS/SFT_LR-class env knobs),
scripts/migration/h3_s0_zeroshot.py (the bench protocol: GpuServer usage,
the frozen 100-case cohort, compliance scoring) and its result file
docs/validation/2026-09-10-h3-s0-zeroshot.json (embedded below as the
S0_ZERO_SHOT reference column).

Arms (selection rule, S0 evidence): marker-reusing variants only (v13 is
the must-FAIL control, 0/100); cursor-encoding twins collapse into their
parent arm — v06/v11 share the checkpoint of v05/v10 per matrix §5 (CE1 is
a render convention; the corpus already carries CE1 rows, which the arm
data keeps in their original encoding) — leaving three render families,
ranked by worst zero-shot margin in the family (v06 99/100 makes the v05
family the weakest) then by cheapest adaptation (v10 keeps the incumbent's
byte-identical generation tail; v14 drops the history slot outright):
  v10_psmtail_merge   PSM-T + MV1 + HP3 + CE2   primary (tail == zeta2's)
  v05_psm_merge       PSM   + MV1 + HP2 + CE2   literal PSM (family holds
                                                the worst zero-shot margin)
  v14_psm_merge_nohist PSM  + MV1 + HP0 + CE2   history-slot isolation arm
zeta2 is NOT an arm: it is the banked b4 baseline, re-served fresh by the
bench as the in-run control column.

Data: the sft_v3 mixture re-rendered per arm (no new corpus). Each row's
zeta2 prompt is inverted with the S0 prompt-level inverse (rv.parse_prompt)
— extended here for the two corpus shapes it did not need to know: the
finish_block/comment_to_code builders put <[fim-prefix]><filename>{path}
as the header when there is no history slot (shape B), and hidden_r_
instruction rows carry no render geometry at all (alpaca; excluded). Every
kept row passes a byte-identity check (re-render == original prompt) and
keeps its ORIGINAL cursor encoding (CE1 rows keep the bare <|user_cursor|>
region line; CE2 rows pass through byte-exact) and its original target
verbatim — the completion contract is shared by every MV1 variant, so only
the prompt side changes. Measured on the live corpus: 168,549/207,551
train rows are losslessly re-renderable (excludes the 39,000 alpaca rows
and 2 adversarial-content rows; the mixture shift is logged per arm in
prepare's manifest). All three arms stage the SAME seeded row selection
(shuffle(42) of the re-renderable pool, row_cap 12,000) so arms are paired
row-for-row; only the render differs.

Adaptation recipe (per arm): train_sft.py on the banked b4 merge
(/mnt/h/sepalith/runs/pft1_b4_merged) with the b4 target list and HALF the
b4 LoRA geometry — r=16, alpha=32 via the SFT_LORA_R/SFT_LORA_ALPHA env
knobs (adaptation, not capability; trainable 10,911,744 = exactly half the
measured b4 line 21,823,488). 150 max_steps at the banked effective batch
16 = 2,400 rows = 0.2 epoch of the 12k staged selection; seed 3407, lr
2e-4 cosine, seq 2048 — every other knob banked-default (byte-identical
OFF contract). Attachment gate: "Trainable parameters" == 10,911,744 in
the train log (B3 incident rule). Export: export_gguf.py Q8_0 ->
experiments/models/h3s1_<arm>-Q8_0.gguf.

Bench: the S0 protocol verbatim (temp 0, n_predict 192, variant stop
strings, format_fail with the stopped_ flag), one tracked GPU serve per
model: the fresh b4 baseline control + the three adapted arms. Every model
x every one of the 7 variants x the same frozen 100 scenario cases ->
per-model compliance column; the diagonal cell is the adapted arm's own
variant, the zeta2 column is backward compatibility (did adaptation break
the incumbent render?), and the b4 row re-verifies the S0 zero-shot
column. The verdict stays with the lead: this bench measures compliance
and transfer only — quality (exact/validator), noopFP and cache are the
separate S1 legs per the H3 row.

Budget: 3 arms x ~0.2 epoch (~150 steps) x ~40 min on the 5090 (load +
adapt + Q8 export) + 4 bench serves (~700 greedy requests each, ~15-25
min per serve) — under 4 h wall, no queue side effects.

Usage:
  python3 experiments/eval/h3_s1_adapt.py prepare --run <dir> \
      [--corpus /mnt/h/sepalith/datasets/sft_v3] [--row-cap 12000]
  python3 experiments/eval/h3_s1_adapt.py train --run <dir> [--dry-run] \
      [--only v10_psmtail_merge,...]
  python3 experiments/eval/h3_s1_adapt.py bench --run <dir> --assets <dir> \
      [--baseline experiments/models/b4_qwen35_2b-Q8_0.gguf] [--port 18478]
  python3 experiments/eval/h3_s1_adapt.py --smoke [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import render_variants as rv  # noqa: E402  (the frozen S0 registry)

# --- arms + frozen references --------------------------------------------------
ARMS = ["v10_psmtail_merge", "v05_psm_merge", "v14_psm_merge_nohist"]
BENCH_VARIANTS = ["zeta2", "v10_psmtail_merge", "v11_psmtail_merge_empty",
                  "v05_psm_merge", "v06_psm_merge_empty",
                  "v14_psm_merge_nohist", "v13_zeta1_alpaca"]
# S0 zero-shot pass_n per variant (docs/validation/2026-09-10-h3-s0-zeroshot.
# json, b4 Q8, first 100 frozen scenario cases; recomputed from the per-row
# ledger after the n_fail tally bug) — the reference column for transfer.
S0_ZERO_SHOT = {"zeta2": 100, "v10_psmtail_merge": 100,
                "v11_psmtail_merge_empty": 100, "v05_psm_merge": 100,
                "v06_psm_merge_empty": 99, "v14_psm_merge_nohist": 100,
                "v13_zeta1_alpaca": 0}

# --- adaptation recipe constants ------------------------------------------------
BASE_MODEL = Path("/mnt/h/sepalith/runs/pft1_b4_merged")   # banked b4 merge
CORPUS = Path("/mnt/h/sepalith/datasets/sft_v3")
RUNS = Path("/mnt/h/sepalith/runs")
MODELS_DIR = REPO / "experiments" / "models"
B4_TARGETS = ("q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,"
              "in_proj_b,in_proj_z,out_proj,gate_proj,up_proj,down_proj")
EXPECT_TRAINABLE_B4 = 21823488       # measured b4 line (r32/a64, 96 modules)
LORA_R, LORA_ALPHA = 16, 32          # HALF the b4 geometry: adaptation only
EXPECT_TRAINABLE_ARM = 10911744      # linear in r: exactly EXPECT_B4 // 2
ROW_CAP = 12000                      # staged selection per arm (seeded 42)
EVAL_CAP = 500                       # the trainer evals exactly the first 500
STEPS = 150                          # 150 x effective 16 = 2400 = 0.2 epoch
EFFECTIVE_BATCH = 16                 # banked 4 x 4 (SFT_PD_BATCH default)
EPOCH_FRAC = STEPS * EFFECTIVE_BATCH / ROW_CAP      # 0.2, by construction
TRAIN_PY = REPO / "experiments" / "training" / "train_sft.py"
EXPORT_PY = REPO / "experiments" / "training" / "export_gguf.py"
SELECTION_SEED = 42                  # the trainer's own shuffle discipline

FIM_PREFIX_FILENAME = "<[fim-prefix]><filename>"


class CorpusRowError(ValueError):
    """A corpus row that no variant render can represent losslessly."""


# --- corpus inversion (zeta2 prompt -> canonical example) ----------------------
def _split_region(lines):
    io = lines.index(rv.REGION_OPEN)
    ic = lines.index(rv.REGION_CLOSE, io + 1)
    if lines[-1] != rv.FIM_MIDDLE:
        raise CorpusRowError("tail is not <[fim-middle]>")
    region, cur, enc = rv._cursor_from_region(lines[io + 1:ic], rv.CURSOR2)
    return io, region, cur, enc


def parse_corpus_prompt(prompt: str) -> tuple[dict, str]:
    """Invert a banked sft_v3 zeta2 prompt into the canonical example.

    Shape A (history slot present): exactly rv.parse_prompt('zeta2') — the
    frozen S0 inverse, reused not forked.
    Shape B (finish_block / comment_to_code / synthetic_analyst builders):
    no edit_history header; <[fim-prefix]><filename>{path} IS the header,
    the suffix block (usually empty) sits between <[fim-suffix]> and it.
    Returns (example, shape); raises CorpusRowError on any other shape.
    """
    lines = prompt.split("\n")
    io, region, cur, enc = _split_region(lines)
    if rv.HIST_HEADER in lines[:io]:
        return dict(rv.parse_prompt(prompt, "zeta2"), _shape="A"), "A"
    if not lines or lines[0] != rv.FIM_SUFFIX:
        raise CorpusRowError("prompt head is not <[fim-suffix]>")
    h = next((i for i, l in enumerate(lines[1:io], 1)
              if l.startswith(FIM_PREFIX_FILENAME)), None)
    if h is None:
        raise CorpusRowError("no header line before the region")
    return dict(path=lines[h][len(FIM_PREFIX_FILENAME):],
                prefix=lines[h + 1:io], history=[], suffix=lines[1:h],
                region_old=region, cursor_idx=cur, cursor_encoding=enc,
                ordering="SPM", history_slot="none", vocab="MV1",
                _shape="B"), "B"


def zeta2_identity(ex: dict) -> str:
    """Re-render the recovered example in its ORIGINAL corpus bytes (the
    prepare-time lossless gate). Guarded cursor; shape B keeps the header
    that carries the path; CE1 keeps the bare cursor-marker region line."""
    parts = [rv.FIM_SUFFIX] + list(ex.get("suffix") or [])
    if ex.get("_shape") == "B":
        parts += [FIM_PREFIX_FILENAME + str(ex.get("path", ""))]
    else:
        ev = ex.get("history") or []
        parts += [rv.HIST_HEADER] + list(ev) + ([""] if ev else [])
        parts += [rv.FILENAME_TAG + str(ex.get("path", ""))]
    parts += list(ex.get("prefix") or []) + [rv.REGION_OPEN]
    if ex.get("cursor_encoding") == "CE1":
        parts += [rv.CURSOR2]
    else:
        parts += rv._with_cursor(ex.get("region_old") or [],
                                 ex.get("cursor_idx"), rv.CURSOR2)
    return "\n".join(parts + [rv.REGION_CLOSE, rv.FIM_MIDDLE])


def render_arm_example(ex: dict, name: str) -> str:
    """Variant-canonical render of a recovered corpus example.

    The row's ORIGINAL cursor encoding is preserved: a CE1 row renders its
    region as the bare <|user_cursor|> line (region_old [""] with the
    marker at idx 0 — byte-identical to the CE1 signature), a CE2 row
    passes region/cursor through untouched. History re-enters through the
    variant's own slot (HP2/HP3 always emit the header; HP0 drops it)."""
    ex2 = dict(ex)
    if ex.get("cursor_encoding") == "CE1":
        ex2["region_old"], ex2["cursor_idx"] = [""], 0
    ex2["event_diff"] = "\n".join(ex["history"]) if ex.get("history") else ""
    return rv.VARIANTS[name][0](ex2)


def recover_row(row: dict) -> tuple[dict, str]:
    """Parse + validate one corpus row; raises CorpusRowError to skip.

    Validation is the pair of contracts the arms rely on: the recovered
    example re-renders to the original prompt byte-exact (no information
    loss), and the target parses under the shared MV1 completion contract
    (region_new ... >>>>>>> UPDATED — identical for every arm, so targets
    are staged verbatim)."""
    try:
        ex, shape = parse_corpus_prompt(row["prompt"])
    except CorpusRowError:
        raise
    except ValueError as e:            # rv.parse_prompt / list.index classes
        raise CorpusRowError(f"unparseable ({e})") from e
    if zeta2_identity(ex) != row["prompt"]:
        raise CorpusRowError("identity: re-render != original prompt")
    target = row["target"]
    if not target.endswith(rv.TERMINATOR) or rv.parse_merge(target) is None:
        raise CorpusRowError("target violates the MV1 completion contract")
    return ex, shape


def arm_row(row: dict, ex: dict, name: str) -> dict:
    prompt = render_arm_example(ex, name)
    return dict(text=prompt + row["target"], prompt=prompt,
                target=row["target"], family=row["family"],
                package_or_repo=row.get("package_or_repo"),
                has_types=row.get("has_types", False),
                h3s1=dict(variant=name, cursor_encoding=ex["cursor_encoding"],
                          shape=ex["_shape"]))


# --- prepare --------------------------------------------------------------------
def _iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare(run: Path, corpus: Path = CORPUS, row_cap: int = ROW_CAP,
            eval_cap: int = EVAL_CAP) -> dict:
    """Stage per-arm train/eval files from the variant-rendered corpus.

    Deterministic: same corpus bytes -> same staged bytes (seeded selection,
    single pass, no timestamps). The manifest records per-family keep/skip
    counts, the CE1/CE2 and shape mix, and the selection seed."""
    run.mkdir(parents=True, exist_ok=True)
    stats, pool, evals = Counter(), [], []
    for split, cap, sink in (("train", row_cap, pool), ("eval", eval_cap, evals)):
        for row in _iter_jsonl(corpus / f"{split}.jsonl"):
            fam = row.get("family", "?")
            stats[f"{split}:{fam}"] += 1
            try:
                ex, shape = recover_row(row)
            except CorpusRowError as e:
                slug = str(e).split(":")[0].split(" ")[0]
                stats[f"skip:{split}:{fam}:{slug}"] += 1
                continue
            stats[f"keep:{split}:{fam}"] += 1
            stats[f"enc:{ex['cursor_encoding']}:{shape}"] += 1
            if split == "train":
                sink.append((row, ex))
            elif len(sink) < cap:      # eval: frozen file order, no shuffle
                sink.append((row, ex))
    order = list(range(len(pool)))
    random.Random(SELECTION_SEED).shuffle(order)
    selected = [pool[i] for i in order[:row_cap]] if row_cap else pool
    epoch_frac = STEPS * EFFECTIVE_BATCH / max(1, len(selected))
    manifest = dict(arms=ARMS, corpus=str(corpus), row_cap=row_cap,
                    eval_cap=eval_cap, selection_seed=SELECTION_SEED,
                    steps=STEPS, epoch_frac=epoch_frac,
                    lora=dict(r=LORA_R, alpha=LORA_ALPHA,
                              expect_trainable=EXPECT_TRAINABLE_ARM),
                    counts=dict(train_pool=len(pool), train_staged=len(selected),
                                eval_staged=len(evals)),
                    stats=dict(sorted(stats.items())),
                    mixture_shift="hidden_r_instruction (alpaca, no render "
                                  "geometry) and non-recoverable rows are "
                                  "excluded; logged per family in stats",
                    boundary="Render adaptation data only; quality/cache "
                             "legs are separate S1 measurements")
    for arm in ARMS:
        adir = run / f"data_{arm.split('_')[0]}"
        adir.mkdir(parents=True, exist_ok=True)
        for split, rows in (("train", selected), ("eval", evals)):
            out = adir / f"{split}.jsonl"
            with out.open("x") as f:
                for row, ex in rows:
                    f.write(json.dumps(arm_row(row, ex, arm)) + "\n")
        manifest.setdefault("arm_dirs", {})[arm] = str(adir)
    _write_json(run / "prepare_manifest.json", manifest)
    return manifest


def _write_json(path: Path, value) -> None:
    with open(path, "x") as f:
        json.dump(value, f, indent=1, sort_keys=True)
        f.write("\n")


# --- train (per-arm via train_sft.py env knobs; the trainer is reused) ----------
def arm_config(arm: str, run: Path, repo: Path = REPO) -> dict:
    """One arm's full training recipe: dirs, env, argv, gates.

    Every env knob defaults to the banked b4 literal inside train_sft.py
    itself; the ones set here are exactly the adaptation deltas (geometry
    r16/a32, eval_steps 150 so eval_loss lands at the 150-step end, the b4
    target list, the shared-machine/ops hygiene the chain scripts export)."""
    run = Path(run).resolve()
    short = arm.split("_")[0]
    data = run / f"data_{short}"
    out = RUNS / f"h3s1_{short}"
    env = {"SFT_TARGETS": B4_TARGETS, "SFT_LORA_R": str(LORA_R),
           "SFT_LORA_ALPHA": str(LORA_ALPHA), "SFT_EVAL_STEPS": str(STEPS),
           "UNSLOTH_COMPILE_DISABLE": "1",
           "UNSLOTH_DISABLE_AUTO_PADDING_FREE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    _convert = (REPO / "experiments/bin/src/llamacpp-b10453"
                "/convert_hf_to_gguf.py")
    if _convert.is_file():             # else export_gguf uses its own default
        env["LLAMA_CONVERT"] = str(_convert)
    # MIDTRAIN/SELEKT/FULL_FT are unset at spawn time (cmd_train): one
    # plain-adaptation mechanism per arm, the b8b convention.
    argv = [str(repo / ".venv-sft" / "bin" / "python"), "-u",
            str(TRAIN_PY), str(BASE_MODEL), str(STEPS), str(data), str(out), ""]
    return dict(arm=arm, short=short, data_dir=str(data), out_dir=str(out),
                steps=STEPS, env=env, argv=argv,
                expect_trainable=EXPECT_TRAINABLE_ARM,
                export_argv=[str(repo / ".venv-sft" / "bin" / "python"),
                             str(EXPORT_PY), str(BASE_MODEL),
                             str(Path(out) / "final_lora"), f"h3s1_{short}"],
                gguf=str(MODELS_DIR / f"h3s1_{short}-Q8_0.gguf"))


def train_command(cfg: dict) -> str:
    """The exact shell line for one arm (what --dry-run prints and what
    cmd_train executes, via the argv/env above — same process either way)."""
    env = " ".join(f"{k}='{v}'" for k, v in sorted(cfg["env"].items()))
    resume = cfg["argv"][7] or '""'
    return f"{env} {cfg['argv'][0]} -u {cfg['argv'][2]} {cfg['argv'][3]} " \
           f"{cfg['argv'][4]} {cfg['argv'][5]} {cfg['argv'][6]} {resume}"


def _gate_attachment(log_text: str, expect: int) -> None:
    """B3 incident rule: the attachment line must show exactly `expect`
    trainable parameters (LoRA params are linear in r, so the r16 arm is
    exactly half the measured 21,823,488 b4 line)."""
    line = next((l for l in log_text.splitlines()
                 if "Trainable parameters" in l), None)
    if line is None:
        raise SystemExit("gate-A FAIL: no attachment line in the train log")
    n = int(line.replace(",", "").split("=")[1].split("of")[0].strip())
    if n != expect:
        raise SystemExit(f"gate-A FAIL: trainable {n} != {expect} ({line})")


def cmd_train(run: Path, arms=None, dry_run=False, repo: Path = REPO) -> int:
    arms = [a for a in (arms or ARMS) if a in ARMS]
    for arm in arms:
        cfg = arm_config(arm, run, repo)
        if not Path(arm_config(arm, run)["data_dir"], "train.jsonl").is_file():
            raise SystemExit(f"{arm}: staged data missing — run prepare first")
        if dry_run:
            print(f"# {arm}: r{LORA_R}/a{LORA_ALPHA}, {STEPS} steps "
                  f"(0.2 epoch of {ROW_CAP} rows), gate trainable = "
                  f"{EXPECT_TRAINABLE_ARM:,}")
            print(train_command(cfg))
            continue
        log = Path(cfg["out_dir"] + "_train.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        child = dict(os.environ)
        for refused in ("MIDTRAIN_MASK", "MIDTRAIN_PACK", "SELEKT_MASK",
                        "FULL_FT", "FULL_FT_LR"):
            child.pop(refused, None)   # one plain-adaptation mechanism per arm
        env = child | cfg["env"]
        with log.open("a") as lf:
            rc = subprocess.run(cfg["argv"], stdout=lf, stderr=lf, cwd=repo,
                                env=env).returncode
        text = log.read_text()
        _gate_attachment(text, cfg["expect_trainable"])
        if rc != 0 or "DONE" not in text:
            raise SystemExit(f"{arm}: trainer rc={rc}, no DONE (see {log})")
        elog = Path(cfg["out_dir"] + "_export.log")
        with elog.open("a") as lf:
            rc = subprocess.run(cfg["export_argv"], stdout=lf, stderr=lf,
                                cwd=repo,
                                env={**os.environ, **cfg["env"]}).returncode
        if rc != 0 or not Path(cfg["gguf"]).is_file():
            raise SystemExit(f"{arm}: export rc={rc} (see {elog})")
        print(f"{arm}: adapted + exported -> {cfg['gguf']}")
    if dry_run:
        print("# export per arm: .venv-sft/bin/python experiments/training/"
              "export_gguf.py /mnt/h/sepalith/runs/pft1_b4_merged "
              "<out>/final_lora h3s1_<arm>  -> experiments/models/"
              "h3s1_<arm>-Q8_0.gguf")
    return 0


# --- bench scoring (pure; the serve loop feeds it a ledger) ---------------------
def summarize_ledger(rows: list) -> dict:
    """{model: {variant: {n, format_fail, pass_n, fail_reasons}}} from
    per-request ledger rows (dict(model, variant, fail, reason)).

    n_fail counts rows with fail=True — never a truthy reason key: the S0
    live tally counted the success reason "ok" as a failure and had to be
    recomputed from this ledger (accounting note in the S0 result file;
    pinned here as a regression)."""
    out = {}
    for r in rows:
        cell = out.setdefault(r["model"], {}).setdefault(
            r["variant"], dict(n=0, format_fail=0, pass_n=0, fail_reasons={}))
        cell["n"] += 1
        if r["fail"]:
            cell["format_fail"] += 1
            cell["fail_reasons"][r["reason"]] = \
                cell["fail_reasons"].get(r["reason"], 0) + 1
        else:
            cell["pass_n"] += 1
    return out


def transfer_table(matrix: dict) -> dict:
    """Per-arm transfer readouts against the embedded S0 zero-shot column.

    diagonal: the adapted model on ITS OWN variant (what adaptation bought).
    backward_zeta2: the adapted model on the incumbent render (does the
    fallback stay alive?). b4_on_arm: the fresh baseline on the arm's
    variant (the S0 zero-shot number, re-verified in-run)."""
    table = {}
    b4 = matrix.get("b4_baseline", {})
    for arm in ARMS:
        m = matrix.get(f"h3s1_{arm.split('_')[0]}", {})
        diag = m.get(arm, {}).get("pass_n")
        back = m.get("zeta2", {}).get("pass_n")
        zero = b4.get(arm, {}).get("pass_n")
        table[arm] = dict(
            diagonal_pass=diag, backward_zeta2_pass=back,
            b4_zeroshot_recheck=zero,
            s0_zeroshot=S0_ZERO_SHOT[arm],
            diagonal_minus_s0=(diag - S0_ZERO_SHOT[arm]
                               if diag is not None else None))
    return table


def cmd_bench(run: Path, assets: Path, baseline: Path, port: int = 18478,
              n_predict: int = 192) -> int:
    """The S0 compliance bench, re-run per adapted model + the baseline.

    Serving pattern mirrors h3_s0_zeroshot byte-for-byte (tracked GPU
    llama-server from $S1_RUNTIME, ctx 8192, seed 20260905, temp 0, variant
    stop strings, format_fail(text, name, stopped))."""
    sys.path.insert(0, str(REPO / "scripts" / "migration"))
    from s1_gpu import GpuServer, DeadlineExceeded, check_offload  # noqa: E402
    import h3_s0_zeroshot as s0  # noqa: E402  (cases(), the frozen cohort)

    import urllib.request
    rows = s0.cases(assets)
    if len(rows) != s0.N_CASES:
        raise SystemExit("case cohort incomplete")
    models = [("b4_baseline", Path(baseline))] + [
        (f"h3s1_{a.split('_')[0]}", Path(arm_config(a, run)["gguf"]))
        for a in ARMS]
    for name, path in models:
        if not path.is_file():
            raise SystemExit(f"{name}: model missing ({path}) — train first")
    run.mkdir(parents=True, exist_ok=True)
    ledger = []
    with (run / "requests.jsonl").open("x") as out:
        for mname, mpath in models:
            server = None

            def expire(*_):
                raise DeadlineExceeded(f"H3-S1 bench exceeded bound ({mname})")
            old = signal.signal(signal.SIGALRM, expire)
            signal.alarm(3600)
            try:
                server = GpuServer(
                    mpath, port, ["-lv", "4", "--seed", "20260905"], ctx=8192,
                    server=Path(os.environ["S1_RUNTIME"]) / "llama-server",
                    foreground=True, log_path=run / mname / "server.log")
                server.start(ready_timeout=300)
                check_offload(server.log_path.read_text())
                for name in BENCH_VARIANTS:
                    render, _ = rv.VARIANTS[name]
                    for i, row in enumerate(rows):
                        prompt = s0.render_case(name, row)
                        body = json.dumps(dict(
                            prompt=prompt, n_predict=n_predict, temperature=0,
                            stream=False, stop=rv.STOP_STRINGS[name])).encode()
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{port}/completion", data=body,
                            headers={"Content-Type": "application/json"})
                        started = time.monotonic()
                        with urllib.request.urlopen(req, timeout=300) as r:
                            data = json.load(r)
                        text = data.get("content", "")
                        stopped = data.get("stop_type") in ("word", "eos")
                        is_fail, reason = rv.format_fail(text, name, stopped)
                        ledger.append(dict(model=mname, variant=name, case=i,
                                           fail=is_fail, reason=reason))
                        out.write(json.dumps(dict(
                            ledger[-1],
                            prompt_tokens=data.get("tokens_evaluated"),
                            wall_s=round(time.monotonic() - started, 3))) + "\n")
                        out.flush()
                print(f"{mname}: " + json.dumps(
                    {v: summarize_ledger(ledger)[mname][v]["pass_n"]
                     for v in BENCH_VARIANTS}), flush=True)
            finally:
                signal.alarm(0)
                if server is not None:
                    server.stop()
                signal.signal(signal.SIGALRM, old)
    matrix = summarize_ledger(ledger)
    _write_json(run / "evaluation.json", dict(
        results=matrix, transfer=transfer_table(matrix),
        s0_zero_shot_reference=S0_ZERO_SHOT,
        s0_reference_source="docs/validation/2026-09-10-h3-s0-zeroshot.json",
        boundary="Compliance + cross-variant transfer only (temp 0, Q8, the "
                 "frozen 100-case S0 cohort); quality (exact/validator), "
                 "noopFP and cache are separate S1 legs; verdict stays with "
                 "the lead per the pre-registered H3 rule"))
    return 0


# --- smoke (fixtures only; no NAS, no GPU, no server) ---------------------------
def _smoke_corpus():
    """A miniature sft_v3: one row per (family-class, shape, encoding)."""
    rows = []

    def add(fam, prompt, target):
        rows.append(dict(text=prompt + target, prompt=prompt, target=target,
                         family=fam, package_or_repo="fixture/pkg",
                         has_types=False))

    for fam, fix in (("edit_pairs", rv.fixture_2k),
                     ("rename_propagation", rv.fixture_small)):
        ex = dict(fix())
        prompt = rv.render_zeta2(ex)
        target = "\n".join(ex["region_new"]).rstrip() + "\n" + rv.TERMINATOR
        add(fam, prompt, target)
    # shape B (comment_to_code builder conventions) with a CE1 region
    parts = ([rv.FIM_SUFFIX, FIM_PREFIX_FILENAME + "R/util.R",
              "# TODO: handle NA", rv.REGION_OPEN, rv.CURSOR2,
              rv.REGION_CLOSE, rv.FIM_MIDDLE])
    add("comment_to_code_real", "\n".join(parts),
        "  y <- mean(x, na.rm = TRUE)\n" + rv.TERMINATOR)
    # non-renderable + corrupted: prepare must skip both, deterministically
    add("hidden_r_instruction", "### Instruction:\nfix it\n\n### Response:\n\n",
        "x <- 1\n")
    add("broken", "no markers here at all", "garbage")
    return rows


def run_smoke(as_json: bool = False) -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "sft_v3"
        corpus.mkdir()
        for split, rows in (("train", _smoke_corpus() * 40),
                            ("eval", _smoke_corpus())):
            with (corpus / f"{split}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        run = Path(tmp) / "run"
        manifest = prepare(run, corpus, row_cap=50, eval_cap=3)
        # arm-config + the exact commands (dry-run form)
        cfgs = [arm_config(a, run) for a in ARMS]
        # compliance scoring on fabricated ledger rows: perfect completions
        # for the marker-reusing variants, the designed v13 failure
        rows_ledger = []
        for model in ["b4_baseline"] + [f"h3s1_{c['short']}" for c in cfgs]:
            for name in BENCH_VARIANTS:
                for i in range(3):
                    if name == "v13_zeta1_alpaca":
                        text, stopped = "no tags at all", False
                    else:
                        ex = rv.fixture_small()
                        text, stopped = rv.completion(ex, name), True
                    fail, reason = rv.format_fail(text, name, stopped)
                    rows_ledger.append(dict(model=model, variant=name,
                                            case=i, fail=fail, reason=reason))
        matrix = summarize_ledger(rows_ledger)
        summary = dict(smoke="h3s1-adapt",
                       staged=manifest["counts"],
                       skip_hidden=manifest["stats"].get(
                           "skip:train:hidden_r_instruction:unparseable", 0),
                       commands=[train_command(c) for c in cfgs],
                       matrix_pass={m: {v: matrix[m][v]["pass_n"]
                                        for v in BENCH_VARIANTS}
                                    for m in matrix},
                       transfer=transfer_table(matrix),
                       epoch_frac=manifest["epoch_frac"],
                       expect_trainable=EXPECT_TRAINABLE_ARM)
        # invariants the smoke must prove (fail loudly, not silently)
        assert manifest["counts"]["train_staged"] == 50
        assert summary["skip_hidden"] == 40, "alpaca rows must be excluded"
        for m in matrix:
            assert matrix[m]["v13_zeta1_alpaca"]["pass_n"] == 0
            assert matrix[m]["zeta2"]["pass_n"] == 3
        assert summary["transfer"]["v10_psmtail_merge"]["diagonal_pass"] == 3
        if as_json:
            print(json.dumps(summary))
        else:
            print(json.dumps(dict(smoke="h3s1-adapt", staged=manifest["counts"],
                                  epoch_frac=manifest["epoch_frac"],
                                  expect_trainable=EXPECT_TRAINABLE_ARM)))
            for c in summary["commands"]:
                print("TRAIN:", c)
            print("MATRIX pass_n:", json.dumps(summary["matrix_pass"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="fixture-only end-to-end: prepare + arm configs + "
                         "scoring math; no NAS, no GPU, no server")
    ap.add_argument("--json", action="store_true",
                    help="smoke: machine-readable single line")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("prepare", help="render + stage train files per arm")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--row-cap", type=int, default=ROW_CAP)
    p.add_argument("--eval-cap", type=int, default=EVAL_CAP)
    p = sub.add_parser("train", help="adapt each arm via train_sft.py")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="print the exact per-arm command lines; no training")
    p.add_argument("--only", help="comma-separated subset of " + ",".join(ARMS))
    p = sub.add_parser("bench", help="S0 compliance bench per adapted model")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--baseline", type=Path,
                   default=MODELS_DIR / "b4_qwen35_2b-Q8_0.gguf")
    p.add_argument("--port", type=int, default=18478)
    args = ap.parse_args(argv)
    if args.smoke:
        return run_smoke(as_json=args.json)
    if args.cmd == "prepare":
        m = prepare(args.run, args.corpus, args.row_cap, args.eval_cap)
        print(json.dumps(m["counts"]))
        return 0
    if args.cmd == "train":
        arms = args.only.split(",") if args.only else None
        return cmd_train(args.run, arms=arms, dry_run=args.dry_run)
    if args.cmd == "bench":
        return cmd_bench(args.run, args.assets, args.baseline, args.port)
    ap.error("nothing to do: pass --smoke or a subcommand "
             "(prepare/train/bench)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
