#!/usr/bin/env python3
"""C09 day-after: bf16 GGUF, imatrix, quant tiers, DEV evaluation, KLD and the quant gate.

    uv run python docs/training/plan-20260927/c09/quantize_and_gate.py \\
        --hf-dir <arm-A checkpoint> --name arm-a \\
        --work-dir /mnt/e/sepalith/campaign-20260915/quant-work/arm-a [--gpu-window] [--dry-run]

Steps (each skips work whose output already exists, so a stopped run resumes):

1. ``export_gguf.py`` converts the HF checkpoint to ``<name>-bf16.gguf`` and
   makes ``<name>-Q8_0.gguf`` (``--source-gguf`` reuses a bf16 GGUF).
2. ``llama-imatrix`` over ``calibration-v1/calibration.txt`` with
   ``--parse-special --no-escape -c 4096 -b 4096`` (see its manifest); the
   imatrix is checked for full non-embedding coverage and gets the
   provenance receipt ``export_gguf.py`` verifies.
3. ``export_gguf.py --source-gguf`` makes the tiers (``token_embd`` q8_0 for all):

   | label | type | imatrix | output head | overrides |
   | --- | --- | --- | --- | --- |
   | Q6_K, Q5_K_M, Q4_K_M | as named | yes | q8_0 | |
   | Q4_K_M-head-q6_k | Q4_K_M | yes | q6_k | (head A/B) |
   | Q4_K_M-uncalibrated | Q4_K_M | no | q8_0 | (control) |
   | Q4_K_M+ | Q4_K_M | yes | q8_0 | attn_k, attn_v q8_0; ffn_down of layers 0, 1, L-2, L-1 q6_k |
   | Q4_K_M-attn-out-q8_0 | Q4_K_M | yes | q8_0 | attn_output q8_0 (HINT4: the sensitive module) |

4. ``gguf_dev_eval.py`` on DEV250 (the gate panel) and DEV75 for bf16 and
   every tier: CPU build at ``nice -n 10`` by default, the CUDA build only
   with ``--gpu-window``.
5. ``llama-perplexity --kl-divergence-base`` on the held-out TRAIN slice with
   bf16 (its stored token stream is checked against the slice's expected
   ids), then ``--kl-divergence`` for every tier.
6. ``quant_gate.py`` writes ``gate-receipt.json``.

``run.json`` in the work dir records every command, exit code, wall time
and artifact sha256.  ``--dry-run`` prints the plan and writes nothing.
Smoke options: ``--existing LABEL=PATH`` reuses a GGUF, ``--variants``
limits the tiers, ``--panels``/``--limit``/``--cap``/``--order`` shrink the
DEV runs, ``--imatrix-chunks``/``--kld-chunks`` shrink the tool runs.  Any of
these makes the gate receipt non-binding.

Addendum note: the gate is specified "greedy on CPU".  A ``--gpu-window`` run
is self-consistent (bf16 and every tier on the same build), but the receipt
records the backend so the user can decide whether it binds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))
from sepalith.quant.calibration import first_mismatch  # noqa: E402
from sepalith.quant.llama_tools import (NICE, file_sha256, gpu_layers, llama_bin, llama_root, llama_version,  # noqa: E402
                                        parse_kld_log, read_kld_base_tokens, refuse_sealed, tool_env)

HERE = Path(__file__).resolve().parent
EXPORT = REPO / "experiments" / "training" / "export_gguf.py"
CALIBRATION_DIR = Path("/mnt/e/sepalith/campaign-20260915/quant-work/calibration-v1")
EMBEDDING = "q8_0"
VARIANT_LABELS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_M-head-q6_k", "Q4_K_M-uncalibrated", "Q4_K_M+",
                  "Q4_K_M-attn-out-q8_0", "Q4_K_M-mixA", "Q4_K_M-mixB")
QUANT_TYPE = {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q5_K_M": "Q5_K_M", "Q4_K_M": "Q4_K_M",
              "Q4_K_M-head-q6_k": "Q4_K_M", "Q4_K_M-uncalibrated": "Q4_K_M", "Q4_K_M+": "Q4_K_M",
              "Q4_K_M-attn-out-q8_0": "Q4_K_M", "Q4_K_M-mixA": "Q4_K_M", "Q4_K_M-mixB": "Q4_K_M"}

IMATRIX_CHECK = r"""
import json, sys
sys.path.insert(0, sys.argv[3])
from gguf import GGUFReader
im, src = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
sums = {t.name[:-8] for t in im.tensors if t.name.endswith('.in_sum2')}
counts = {t.name[:-7] for t in im.tensors if t.name.endswith('.counts')}
weights = {t.name for t in src.tensors if len(t.shape) >= 2 and t.name not in ('token_embd.weight', 'output.weight')}
field = lambda k: (im.get_field(k).contents() if im.get_field(k) is not None else None)
print(json.dumps({'chunk_count': field('imatrix.chunk_count'), 'chunk_size': field('imatrix.chunk_size'),
                  'entries': len(sums), 'expected_matrices': len(weights),
                  'missing': sorted(weights - sums), 'unpaired': sorted(sums ^ counts)}))
"""


def variant_specs(n_layers: int) -> dict[str, dict]:
    """Stem suffix, tier and export flags for each label (Q8_0 comes from step 1).

    ``embedding`` defaults to :data:`EMBEDDING` (q8_0) when absent.  For
    attn_v/ffn_down, override the whole category (all 42 layers), never a
    subset: a partial override skips the b10453 ``i_attention_wv``/
    ``i_ffn_down`` counters and silently shifts the Q4_K_M every-3rd q6_K
    mix for every later layer of that category (observed on the night run's
    Q4_K_M+; see docs/validation/2026-10-17-quant-autoresearch.md).
    """
    last = (n_layers - 2, n_layers - 1)
    edge_down = r"blk\.(0|1|%d|%d)\.ffn_down\.weight=q6_k" % last
    return {
        "Q6_K": {"stem": "imat", "tier": "Q6_K", "imatrix": True, "output": "q8_0", "tensor_types": []},
        "Q5_K_M": {"stem": "imat", "tier": "Q5_K_M", "imatrix": True, "output": "q8_0", "tensor_types": []},
        "Q4_K_M": {"stem": "imat", "tier": "Q4_K_M", "imatrix": True, "output": "q8_0", "tensor_types": []},
        "Q4_K_M-head-q6_k": {"stem": "head-q6_k", "tier": "Q4_K_M", "imatrix": True, "output": "q6_k", "tensor_types": []},
        "Q4_K_M-uncalibrated": {"stem": "uncal", "tier": "Q4_K_M", "imatrix": False, "output": "q8_0", "tensor_types": []},
        "Q4_K_M+": {"stem": "q4plus", "tier": "Q4_K_M", "imatrix": True, "output": "q8_0",
                    "tensor_types": [r"attn_k\.weight=q8_0", r"attn_v\.weight=q8_0", edge_down]},
        "Q4_K_M-attn-out-q8_0": {"stem": "attn-out-q8_0", "tier": "Q4_K_M", "imatrix": True, "output": "q8_0",
                                 "tensor_types": [r"attn_output\.weight=q8_0"]},
        # 2026-10-17 quant autoresearch (CPU search): mixed-precision q4 tiers.
        # mixA "speed" (report Pareto: smallest + fastest CPU decode of the gate-passing
        # recipes): token_embd/output q5_K, attn_output q6_K, attn_k/attn_v q8_0 (all layers).
        "Q4_K_M-mixA": {"stem": "mixa", "tier": "Q4_K_M", "imatrix": True, "output": "q5_K", "embedding": "q5_K",
                        "tensor_types": [r"attn_output\.weight=q6_K", r"attn_k\.weight=q8_0", r"attn_v\.weight=q8_0"]},
        # mixB "fidelity" (best KLD under the Q4_K_M size): mixA + ffn_down q6_K on all 42
        # layers.  Note: q6_K ffn_down costs ~30% CPU decode (bandwidth) — see the report.
        "Q4_K_M-mixB": {"stem": "mixb", "tier": "Q4_K_M", "imatrix": True, "output": "q5_K", "embedding": "q5_K",
                        "tensor_types": [r"attn_output\.weight=q6_K", r"attn_k\.weight=q8_0", r"attn_v\.weight=q8_0",
                                          r"ffn_down\.weight=q6_K"]},
    }


class Runner:
    def __init__(self, work: Path, dry_run: bool):
        self.work, self.dry_run = work, dry_run
        self.record: dict = {"schema": "sepalith.c09.quantize-and-gate-run.v1", "steps": []}

    def run(self, name: str, argv: list[str], *, env=None, log: Path | None = None, outputs=()) -> dict | None:
        if outputs and all(Path(p).exists() for p in outputs):
            print(f"SKIP {name}: outputs exist")
            self.record["steps"].append({"name": name, "skipped": "outputs exist", "outputs": [str(p) for p in outputs]})
            return None
        print(f"{'PLAN' if self.dry_run else 'RUN'} {name}: {' '.join(argv)}", flush=True)
        if self.dry_run:
            return None
        started = time.monotonic()
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                done = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        else:
            done = subprocess.run(argv, env=env, check=False)
        step = {"name": name, "argv": argv, "exit_code": done.returncode, "log": str(log) if log else None,
                "wall_seconds": round(time.monotonic() - started, 1)}
        self.record["steps"].append(step)
        self.save()
        if done.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {done.returncode}" + (f"; see {log}" if log else ""))
        return step

    def save(self):
        if not self.dry_run:
            (self.work / "run.json").write_text(json.dumps(self.record, indent=2) + "\n")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, required=True, help="HF checkpoint (config, tokenizer, safetensors)")
    ap.add_argument("--name", required=True, help="file stem, e.g. arm-a-1071")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR)
    ap.add_argument("--source-gguf", type=Path, help="reuse an existing bf16 GGUF instead of converting")
    ap.add_argument("--existing", action="append", default=[], metavar="LABEL=PATH", help="reuse a GGUF for a label")
    ap.add_argument("--variants", nargs="+", choices=VARIANT_LABELS, default=list(VARIANT_LABELS))
    ap.add_argument("--panels", nargs="+", choices=("dev250", "dev75"), default=["dev250", "dev75"])
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--order", choices=("panel", "shortest"), default="panel")
    ap.add_argument("--imatrix-chunks", type=int, help="default: every calibration chunk")
    ap.add_argument("--kld-chunks", type=int, help="default: every held-out window")
    ap.add_argument("--skip-kld", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gpu-window", action="store_true", help="use the CUDA build (only inside the night window)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    for path in (a.hf_dir, a.work_dir, a.calibration_dir, a.source_gguf or "", *a.existing):
        refuse_sealed(path)
    if a.threads < 1 or (a.threads > 8 and not a.gpu_window):
        raise SystemExit("daytime CPU rule: 1 to 8 threads")
    build = "cuda" if a.gpu_window else "cpu"
    existing = {}
    for item in a.existing:
        label, _, path = item.partition("=")
        if label not in ("bf16", *VARIANT_LABELS) or not Path(path).is_file():
            raise SystemExit(f"bad --existing {item!r}")
        existing[label] = Path(path).resolve()
    if a.source_gguf:
        existing["bf16"] = a.source_gguf.resolve()
    manifest = json.loads((a.calibration_dir / "manifest.json").read_text())
    calibration = a.calibration_dir / "calibration.txt"
    heldout = a.calibration_dir / "kld-heldout.txt"
    if not a.dry_run:
        for path, record in ((calibration, manifest["calibration"]["file"]), (heldout, manifest["kld"]["file"])):
            if file_sha256(path) != record["sha256"]:
                raise SystemExit(f"{path} differs from its manifest")
    config = json.loads((a.hf_dir / "config.json").read_text())
    n_layers = int(config["num_hidden_layers"])
    work = a.work_dir.resolve()
    if not a.dry_run:
        (work / "logs").mkdir(parents=True, exist_ok=True)
    runner = Runner(work, a.dry_run)
    runner.record.update(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), args=vars(a) | {
        "hf_dir": str(a.hf_dir), "work_dir": str(work), "calibration_dir": str(a.calibration_dir),
        "source_gguf": str(a.source_gguf) if a.source_gguf else None}, build=build, llama_cpp_root=str(llama_root()),
        calibration_manifest_sha256=file_sha256(a.calibration_dir / "manifest.json"), final_set_access=False)
    if not a.dry_run:
        runner.record["llama_cpp"] = {tool: llama_version(tool, build) for tool in ("llama-imatrix", "llama-perplexity", "llama-server")}
    env_cpu = {**tool_env("cpu"), "NO_LORA": "1", "SEPALITH_LLAMA_CPP": str(llama_root())}
    export_base = [*NICE, sys.executable, str(EXPORT), "unused", str(a.hf_dir)]

    # 1. bf16 source and Q8_0.
    bf16 = existing.get("bf16", work / f"{a.name}-bf16.gguf")
    ggufs = {"bf16": bf16}
    q8 = existing.get("Q8_0", work / f"{a.name}-Q8_0.gguf")
    if "Q8_0" in a.variants or "bf16" not in existing:
        source_flags = ["--source-gguf", str(bf16)] if "bf16" in existing else ["--keep-source"]
        runner.run("export bf16 + Q8_0", [*export_base, a.name, "--out-dir", str(work), "--tiers", "Q8_0",
                                          "--threads", str(a.threads), *source_flags],
                   env=env_cpu, log=work / "logs" / "export-bf16-q8_0.log",
                   outputs=[p for p in (bf16, q8) if "Q8_0" in a.variants or p == bf16])
    if "Q8_0" in a.variants:
        ggufs["Q8_0"] = q8

    # 2. imatrix.
    specs = variant_specs(n_layers)
    wanted = [label for label in a.variants if label != "Q8_0"]
    imatrix = work / f"{a.name}-imatrix.gguf"
    chunks = a.imatrix_chunks or manifest["calibration"]["chunks"]
    if any(specs[label]["imatrix"] and label not in existing for label in wanted):
        n_ctx = manifest["calibration"]["n_ctx"]
        argv = [*NICE, str(llama_bin("llama-imatrix", build)), "-m", str(bf16), "-f", str(calibration),
                "--parse-special", "--no-escape", "-c", str(n_ctx), "-b", str(n_ctx), "--chunks", str(chunks),
                "--no-ppl", "-o", str(imatrix), "-t", str(a.threads), *gpu_layers(build)]
        step = runner.run("imatrix", argv, env=tool_env(build), log=work / "logs" / "imatrix.log", outputs=[imatrix])
        if not a.dry_run and not imatrix.with_suffix(".gguf.json").exists():
            log = (work / "logs" / "imatrix.log").read_text(errors="replace")
            computed = re.findall(r"computing over (\d+) chunks", log)
            if not computed or int(computed[-1]) != chunks:
                raise SystemExit("imatrix log does not show the requested chunk count")
            check = subprocess.run([str(llama_root() / "venv-convert" / "bin" / "python"), "-c", IMATRIX_CHECK,
                                    str(imatrix), str(bf16), str(llama_root() / "src" / "gguf-py")],
                                   capture_output=True, text=True, check=True)
            coverage = json.loads(check.stdout.strip().splitlines()[-1])
            if coverage["missing"] or coverage["unpaired"] or coverage["entries"] != coverage["expected_matrices"]:
                raise SystemExit(f"imatrix coverage incomplete: {coverage}")
            provenance = {"schema": "sepalith.c09.imatrix.v1", "imatrix_sha256": file_sha256(imatrix),
                          "model_sha256": file_sha256(bf16), "model_path": str(bf16),
                          "calibration": {"path": str(calibration), "sha256": manifest["calibration"]["file"]["sha256"],
                                          "manifest_sha256": file_sha256(a.calibration_dir / "manifest.json")},
                          "chunks": chunks, "n_ctx": n_ctx, "argv": argv, "coverage": coverage,
                          "llama_cpp": llama_version("llama-imatrix", build), "wall_seconds": step and step["wall_seconds"]}
            imatrix.with_suffix(".gguf.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # 3. tiers.
    for label in wanted:
        spec = specs[label]
        output = existing.get(label, work / f"{a.name}-{spec['stem']}-{spec['tier']}.gguf")
        ggufs[label] = output
        if label in existing:
            continue
        flags = ["--source-gguf", str(bf16), "--out-dir", str(work), "--tiers", spec["tier"], "--threads", str(a.threads),
                 "--output-tensor-type", spec["output"], "--token-embedding-type", spec.get("embedding", EMBEDDING)]
        flags += ["--imatrix", str(imatrix)] if spec["imatrix"] else ["--uncalibrated"]
        for override in spec["tensor_types"]:
            flags += ["--tensor-type", override]
        runner.run(f"quantize {label}", [*export_base, f"{a.name}-{spec['stem']}", *flags], env=env_cpu,
                   log=work / "logs" / f"quantize-{label}.log", outputs=[output])

    labels = ["bf16", *[label for label in VARIANT_LABELS if label in ggufs and label != "bf16"]]

    # 4. DEV evaluation.
    results: dict[str, dict[str, str]] = {label: {} for label in labels}
    if not a.skip_eval:
        for panel in a.panels:
            for label in labels:
                output = work / "eval" / f"{label}-{panel}.json"
                results[label][panel] = str(output)
                # gguf_dev_eval.py runs llama-server under nice itself.
                argv = [sys.executable, str(HERE / "gguf_dev_eval.py"), "--gguf", str(ggufs[label]),
                        "--tokenizer-dir", str(a.hf_dir), "--panel", panel, "--output", str(output), "--cap", str(a.cap),
                        "--threads", str(a.threads), "--order", a.order, "--build", build, "--label", label]
                argv += ["--limit", str(a.limit)] if a.limit else []
                argv += ["--gpu-window"] if a.gpu_window else []
                runner.run(f"dev {panel} {label}", argv, log=work / "logs" / f"dev-{panel}-{label}.log", outputs=[output])

    # 5. KLD on the held-out slice.
    kld_files: dict[str, str] = {}
    if not a.skip_kld:
        kld_ctx = manifest["kld"]["n_ctx"]
        kld_chunks = a.kld_chunks or manifest["kld"]["chunks"]
        base = work / "kld" / "bf16-base.kld"
        common = ["-c", str(kld_ctx), "-b", str(kld_ctx), "--chunks", str(kld_chunks), "--no-escape",
                  "-t", str(a.threads), *gpu_layers(build)]
        if not a.dry_run:
            base.parent.mkdir(parents=True, exist_ok=True)
        runner.run("kld base bf16", [*NICE, str(llama_bin("llama-perplexity", build)), "-m", str(bf16), "-f", str(heldout),
                                     "--kl-divergence-base", str(base), *common],
                   env=tool_env(build), log=work / "logs" / "kld-base.log", outputs=[base])
        if not a.dry_run:
            stored = read_kld_base_tokens(base)
            expected = json.loads((a.calibration_dir / "kld-heldout.ids.json").read_text())[:kld_chunks * kld_ctx]
            mismatch = first_mismatch(stored["tokens"], expected)
            runner.record["kld_base_tokens"] = {"n_ctx": stored["n_ctx"], "n_chunk": stored["n_chunk"],
                                                "n_vocab": stored["n_vocab"], "identical_to_slice": mismatch is None,
                                                "first_mismatch": mismatch}
            if mismatch is not None:
                raise SystemExit(f"llama-perplexity tokenized the held-out slice differently (first mismatch {mismatch})")
        for label in labels[1:]:
            log = work / "logs" / f"kld-{label}.log"
            record = work / "kld" / f"{label}.json"
            kld_files[label] = str(record)
            argv = [*NICE, str(llama_bin("llama-perplexity", build)), "-m", str(ggufs[label]),
                    "--kl-divergence-base", str(base), "--kl-divergence", *common]
            if runner.run(f"kld {label}", argv, env=tool_env(build), log=log, outputs=[record]) and not a.dry_run:
                record.write_text(json.dumps({"label": label, "gguf": str(ggufs[label]), "base": str(base),
                                              "argv": argv, "log": str(log), "chunks": kld_chunks, "n_ctx": kld_ctx,
                                              "statistics": parse_kld_log(log.read_text(errors="replace"))}, indent=2) + "\n")

    # 6. Gate.
    if a.dry_run:
        print("PLAN gate: quant_gate.py --plan gate-plan.json --output gate-receipt.json")
        return 0
    runner.record["ggufs"] = {label: {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                              for label, path in ggufs.items()}
    runner.save()
    plan = {"gate_panel": a.panels[0] if a.skip_eval is False else "dev250",
            "reference": {"label": "bf16", "results": results["bf16"]},
            "candidates": [{"label": label, "quant_type": QUANT_TYPE[label], "gguf": runner.record["ggufs"][label],
                            "results": results[label], "kld": kld_files.get(label)} for label in labels[1:]]}
    (work / "gate-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    runner.run("gate", [sys.executable, str(HERE / "quant_gate.py"), "--plan", str(work / "gate-plan.json"),
                        "--output", str(work / "gate-receipt.json")])
    runner.record["gate_receipt"] = {"path": str(work / "gate-receipt.json"), "sha256": file_sha256(work / "gate-receipt.json")}
    runner.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
