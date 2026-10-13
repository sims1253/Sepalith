#!/usr/bin/env python3
"""Merge a trained LoRA and export GGUF for llama.cpp serving + eval.

Usage: export_gguf.py <model_id> <lora_dir> <out_stem>
Run inside .venv-sft when a LoRA has to be merged; NO_LORA=1 and
--source-gguf runs only need the standard library. The HF model is converted
to a bf16 GGUF (``--outtype f16`` is still available), which is the
quantization source and the full-precision reference. Defaults to
<out_stem>-Q8_0.gguf in experiments/models (``--out-dir`` changes that).
Use --tiers for additional formats; Q4_K_M requires --imatrix or
--uncalibrated and every IQ type requires --imatrix.

llama.cpp location: SEPALITH_LLAMA_CPP names the pinned llama.cpp install
(default ~/.local/opt/llama.cpp-b10453, built from b10453 = PIN below), with
``src/convert_hf_to_gguf.py``, ``build-cpu/bin/llama-quantize`` and the
converter interpreter ``venv-convert/bin/python``. LLAMA_QUANT, LLAMA_CONVERT
and LLAMA_CONVERT_PYTHON override single pieces (for example an architecture
fork; see docs/research/2026-09-01-local-builds.md). Only an explicit
LLAMA_CONVERT skips the converter pin check. MERGE_VIA_PEFT=1 merges via
transformers+PEFT instead of unsloth (remote-code archs like spark2_5 — run
that under .venv-spark).

NO_LORA=1 (PFT1, 2026-09-06): <lora_dir> is already a FULL HF model dir (no
adapter_config.json — e.g. train_sft.py FULL_FT final_model). Skips every
merge path and converts that dir directly (base-model arg is unused).

Quantizer pass-through: --imatrix, repeated --tensor-type PATTERN=TYPE,
--output-tensor-type, --token-embedding-type and --dry-run. Non-Q8_0 tiers
default to q8_0 output and token-embedding tensors. --dry-run validates the
inputs and prints the planned commands and receipts without merging,
converting or writing files; with --source-gguf it also runs
``llama-quantize --dry-run`` to report each tier's size.

Every GGUF gets a JSON receipt (<file>.json) with every input/output path,
sha256 and flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ROOT = Path(__file__).resolve().parents[2]
PIN = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"  # llama.cpp tag b10453
PIN_TAG = "b10453"
DEFAULT_LLAMA_CPP = Path.home() / ".local" / "opt" / f"llama.cpp-{PIN_TAG}"
RECEIPT_SCHEMA = "sepalith.gguf-export/v2"
TIERS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS")
CALIBRATED_TIERS = ("Q4_K_M",)  # need --imatrix or an explicit --uncalibrated control
# ggml_type_name() values at PIN (ggml/src/ggml.c); llama-quantize matches them
# case-insensitively for --output-tensor-type, --token-embedding-type and --tensor-type.
GGML_TYPES = (
    "f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64",
    "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "mxfp4", "nvfp4",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
    "tq1_0", "tq2_0",
)
# Files the converter reads from an HF model dir; training state (optimizer.pt etc.) is not an input.
HF_INPUT_PATTERNS = ("*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
                     "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*",
                     "vocab*", "merges.txt", "*.model")
# "version: 0.1.0-dev (build 10453, commit 3cb7ffb1a)" at PIN; "build: N (commit)" in older trees.
BUILD_LINE = re.compile(r"build:? (\d+)(?:, commit | \()([0-9a-f]+)")


def ggml_type(value: str) -> str:
    """argparse type: a ggml type name (case-insensitive), returned in canonical spelling."""
    for name in GGML_TYPES:
        if name.lower() == value.lower():
            return name
    raise argparse.ArgumentTypeError(f"invalid ggml type {value!r}; expected one of {', '.join(GGML_TYPES)}")


def tensor_type(value: str) -> str:
    """argparse type: PATTERN=TYPE as llama-quantize --tensor-type expects it."""
    pattern, sep, kind = value.partition("=")
    if not sep or not pattern or not kind:
        raise argparse.ArgumentTypeError(f"--tensor-type must be PATTERN=TYPE, got {value!r}")
    return f"{pattern}={ggml_type(kind)}"


def needs_imatrix(kind: str | None) -> bool:
    return kind is not None and kind.lower().startswith("iq")


def hf_inputs(model_dir: Path) -> dict[str, dict[str, object]]:
    files = sorted({p for pattern in HF_INPUT_PATTERNS for p in model_dir.glob(pattern) if p.is_file()})
    return {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in files}


def git_head(directory: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_echoed(command: list[str]) -> dict[str, object] | None:
    """Run a tool, mirror its output and return llama.cpp's reported build (number, commit)."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace")
    build = None
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        match = BUILD_LINE.search(line)
        if match and build is None:
            build = dict(number=int(match.group(1)), commit=match.group(2))
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
    return build


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("lora")
    ap.add_argument("stem")
    ap.add_argument("--tiers", nargs="+", choices=TIERS, default=["Q8_0"])
    ap.add_argument("--outtype", choices=["bf16", "f16"], default="bf16",
                    help="full-precision GGUF type written by the converter (default bf16)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "models",
                    help="directory for the GGUFs and receipts (default experiments/models)")
    ap.add_argument("--imatrix", type=Path)
    ap.add_argument("--uncalibrated", action="store_true", help="Explicit control arm without imatrix")
    ap.add_argument("--output-tensor-type", "--output-type", dest="output_type", type=ggml_type,
                    help="output.weight type (default q8_0 for non-Q8_0 tiers)")
    ap.add_argument("--token-embedding-type", "--embedding-type", dest="embedding_type", type=ggml_type,
                    help="token_embd.weight type (default q8_0 for non-Q8_0 tiers)")
    ap.add_argument("--tensor-type", dest="tensor_types", action="append", type=tensor_type, default=[],
                    metavar="PATTERN=TYPE", help="per-tensor override passed to llama-quantize; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: no merge, conversion or output files; llama-quantize --dry-run sizes "
                         "when --source-gguf is given")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--keep-source", "--keep-f16", dest="keep_source", action="store_true",
                    help="keep the converted full-precision GGUF (the quality reference)")
    ap.add_argument("--source-gguf", "--f16", dest="source_gguf", type=Path,
                    help="Reuse an existing full precision GGUF; skip merge/conversion")
    args = ap.parse_args(argv)
    LORA, STEM = args.lora, args.stem  # args.model is the base id; the merge paths read it from the adapter
    if args.threads < 1:
        ap.error("threads must be positive")
    if Path(STEM).name != STEM or STEM in (".", ".."):
        ap.error("stem must be a filename, not a path")
    if args.imatrix and args.uncalibrated:
        ap.error("--imatrix and --uncalibrated are mutually exclusive")
    if args.imatrix and not args.imatrix.is_file():
        ap.error("imatrix file does not exist")
    tiers = list(dict.fromkeys(args.tiers))
    if any(t in CALIBRATED_TIERS for t in tiers) and not (args.imatrix or args.uncalibrated):
        ap.error("Q4 tiers require --imatrix or explicit --uncalibrated control")
    override_types = [args.output_type, args.embedding_type, *(t.partition("=")[2] for t in args.tensor_types)]
    if any(needs_imatrix(t) for t in [*tiers, *override_types]) and not args.imatrix:
        ap.error("IQ types require --imatrix (--uncalibrated is not accepted for IQ)")
    if args.source_gguf and not args.source_gguf.is_file():
        ap.error("--source-gguf file does not exist")

    MERGED = ROOT / "experiments" / "models" / f"merged_{STEM}"
    MODELS = args.out_dir.resolve()
    llama_root = Path(os.environ.get("SEPALITH_LLAMA_CPP", str(DEFAULT_LLAMA_CPP)))
    QUANT = os.environ.get("LLAMA_QUANT", str(llama_root / "build-cpu" / "bin" / "llama-quantize"))
    CONVERT = os.environ.get("LLAMA_CONVERT", str(llama_root / "src" / "convert_hf_to_gguf.py"))
    default_python = llama_root / "venv-convert" / "bin" / "python"
    CONVERT_PYTHON = os.environ.get("LLAMA_CONVERT_PYTHON",
                                    str(default_python) if default_python.is_file() else sys.executable)
    if not Path(QUANT).is_file():
        ap.error(f"quantizer not found: {QUANT}; build llama.cpp {PIN_TAG} under SEPALITH_LLAMA_CPP "
                 f"({llama_root}) or set LLAMA_QUANT")
    converter_commit = None
    if not args.source_gguf:
        if not Path(CONVERT).is_file():
            ap.error(f"converter not found: {CONVERT}; install llama.cpp at {PIN} under SEPALITH_LLAMA_CPP, "
                     "or set LLAMA_CONVERT")
        converter_commit = git_head(Path(CONVERT).parent)
        if not os.environ.get("LLAMA_CONVERT") and converter_commit != PIN:
            ap.error(f"default converter must be pinned to {PIN}; found {converter_commit}")

    reused = args.source_gguf is not None
    if reused:
        pass
    elif os.environ.get("NO_LORA"):
        # PFT1 full-FT export: the trained dir IS the model — convert in place.
        assert (Path(LORA) / "config.json").exists(), f"{LORA} is not an HF model dir"
        assert not (Path(LORA) / "adapter_config.json").exists(), \
            "NO_LORA=1 but an adapter_config.json is present — wrong mode"
        MERGED = Path(LORA)
    elif args.dry_run:
        pass  # a dry run plans the merge below but never loads the training stack
    elif os.environ.get("MERGE_VIA_PEFT"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = json.load(open(Path(LORA) / "adapter_config.json"))["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, LORA).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model.save_pretrained(str(MERGED))
        tokenizer.save_pretrained(str(MERGED))
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LORA,  # unsloth loads base + adapter from the saved dir
            max_seq_length=2048, dtype=None, load_in_4bit=False)
        model.save_pretrained_merged(str(MERGED), tokenizer, save_method="merged_16bit")

    if not args.dry_run:
        MODELS.mkdir(parents=True, exist_ok=True)
    source = args.source_gguf.resolve() if reused else MODELS / f"{STEM}-{args.outtype}.gguf"
    conversion = None
    if not reused:
        config = MERGED / "config.json"
        archs = json.loads(config.read_text()).get("architectures", []) if config.is_file() else []
        nextn = ["--no-nextn"] if any("qwen" in a.lower() for a in archs) else []
        partial = source.with_suffix(".gguf.partial")
        command = [CONVERT_PYTHON, CONVERT, str(MERGED), "--outfile", str(partial),
                   "--outtype", args.outtype, *nextn]
        conversion = dict(hf_model_dir=str(MERGED.resolve()), architectures=archs, outtype=args.outtype,
                          nextn_stripped=bool(nextn), converter=str(Path(CONVERT).resolve()),
                          converter_commit=converter_commit, converter_python=CONVERT_PYTHON,
                          command=[*command[:4], str(source), *command[5:]])
        if args.dry_run:
            print("PLAN convert: " + " ".join([*command[:4], str(source), *command[5:]]))
        else:
            conversion["hf_inputs"] = hf_inputs(MERGED)
            started = time.monotonic()
            try:
                subprocess.run(command, check=True)
                partial.replace(source)
            finally:
                partial.unlink(missing_ok=True)
            conversion["elapsed_seconds"] = round(time.monotonic() - started, 1)
    source_info = None
    if source.is_file():
        source_info = dict(path=str(source), bytes=source.stat().st_size, sha256=sha256(source),
                           outtype=args.outtype if not reused else None, reused=reused)
    if conversion and source_info and not args.dry_run:
        source_receipt = dict(schema=RECEIPT_SCHEMA, kind="source", source=source_info, conversion=conversion,
                              created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        source.with_suffix(".gguf.json").write_text(json.dumps(source_receipt, indent=2) + "\n")
    if args.imatrix:
        provenance = args.imatrix.with_suffix(".gguf.json")
        if provenance.is_file():
            recorded = json.loads(provenance.read_text())
            if source_info and recorded.get("model_sha256") and recorded["model_sha256"] != source_info["sha256"]:
                ap.error("imatrix provenance does not match this source GGUF")
            if recorded.get("imatrix_sha256") and recorded["imatrix_sha256"] != sha256(args.imatrix):
                ap.error("imatrix file differs from its provenance receipt")
    imatrix_info = (dict(path=str(args.imatrix.resolve()), sha256=sha256(args.imatrix))
                    if args.imatrix else None)
    quantizer_info: dict[str, object] = dict(path=str(Path(QUANT).resolve()), sha256=sha256(Path(QUANT)), build=None)
    for tier in tiers:
        output = MODELS / f"{STEM}-{tier}.gguf"
        temporary = output.with_suffix(".gguf.partial")
        output_type = args.output_type or (None if tier == "Q8_0" else "q8_0")
        embedding_type = args.embedding_type or (None if tier == "Q8_0" else "q8_0")
        flags = []
        if output_type:
            flags += ["--output-tensor-type", output_type]
        if embedding_type:
            flags += ["--token-embedding-type", embedding_type]
        for override in args.tensor_types:
            flags += ["--tensor-type", override]
        tier_imatrix = imatrix_info if tier != "Q8_0" else None
        if tier_imatrix:
            flags += ["--imatrix", tier_imatrix["path"]]
        if args.dry_run:
            command = [QUANT, "--dry-run", *flags, str(source), tier, str(args.threads)]
        else:
            command = [QUANT, *flags, str(source), str(temporary), tier, str(args.threads)]
        receipt = dict(schema=RECEIPT_SCHEMA, kind="quant", tier=tier, dry_run=args.dry_run,
                       output=dict(path=str(output), bytes=None, sha256=None),
                       source=source_info or dict(path=str(source), bytes=None, sha256=None),
                       conversion=conversion, quantizer=quantizer_info,
                       quantize_command=[*command[:-3], str(output), *command[-2:]] if not args.dry_run else command,
                       imatrix=tier_imatrix,
                       flags=dict(outtype=args.outtype if not reused else None, output_tensor_type=output_type,
                                  token_embedding_type=embedding_type, tensor_types=list(args.tensor_types),
                                  uncalibrated=args.uncalibrated if tier != "Q8_0" else False,
                                  threads=args.threads, keep_source=args.keep_source, dry_run=args.dry_run))
        if args.dry_run:
            print("PLAN quantize: " + " ".join(command))
            if source.is_file():
                quantizer_info["build"] = run_echoed(command) or quantizer_info["build"]
            print(json.dumps(receipt, indent=2))
            continue
        started = time.monotonic()
        try:
            quantizer_info["build"] = run_echoed(command) or quantizer_info["build"]
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        receipt["quantize_seconds"] = round(time.monotonic() - started, 1)
        receipt["output"] = dict(path=str(output), bytes=output.stat().st_size, sha256=sha256(output))
        receipt["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # v1 flat keys, kept for existing readers.
        receipt.update(converter=conversion["converter"] if conversion else None,
                       bytes=receipt["output"]["bytes"], sha256=receipt["output"]["sha256"],
                       output_type=output_type, embedding_type=embedding_type,
                       imatrix_sha256=tier_imatrix["sha256"] if tier_imatrix else None)
        output.with_suffix(".gguf.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"OK -> {output}")
    if not args.keep_source and not reused and not args.dry_run:
        source.unlink()
        source.with_suffix(".gguf.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
