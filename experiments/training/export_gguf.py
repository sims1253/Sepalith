#!/usr/bin/env python3
"""Merge a trained LoRA and export GGUF for llama.cpp serving + eval.

Usage: export_gguf.py <model_id> <lora_dir> <out_stem>
Run inside .venv-sft. Defaults to <out_stem>-Q8_0.gguf in experiments/models.
Use --tiers for additional formats; Q4 requires --imatrix or --uncalibrated.

Alt-build / remote-code overrides (see docs/research/2026-09-01-local-builds.md
for what's on disk): LLAMA_QUANT, LLAMA_CONVERT point at a different llama.cpp
build; MERGE_VIA_PEFT=1 merges via transformers+PEFT instead of unsloth
(remote-code archs like spark2_5 — run that under .venv-spark).

NO_LORA=1 (PFT1, 2026-09-06): <lora_dir> is already a FULL HF model dir (no
adapter_config.json — e.g. train_sft.py FULL_FT final_model). Skips every
merge path and converts that dir directly (base-model arg is unused).
"""
import json, os, subprocess, sys
from pathlib import Path

import argparse
import hashlib

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ROOT = Path(__file__).resolve().parents[2]
PIN = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("lora")
    ap.add_argument("stem")
    ap.add_argument("--tiers", nargs="+", choices=["Q8_0", "Q6_K", "Q4_K_M", "IQ4_XS"], default=["Q8_0"])
    ap.add_argument("--imatrix", type=Path)
    ap.add_argument("--uncalibrated", action="store_true", help="Explicit control arm without imatrix")
    ap.add_argument("--output-type", choices=["Q8_0", "Q6_K"], default="Q8_0")
    ap.add_argument("--embedding-type", choices=["Q8_0", "Q6_K"], default="Q8_0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--keep-f16", action="store_true")
    ap.add_argument("--f16", type=Path, help="Reuse an existing full precision GGUF; skip merge/conversion")
    args = ap.parse_args(argv)
    MODEL, LORA, STEM = args.model, args.lora, args.stem
    if args.threads < 1:
        ap.error("threads must be positive")
    if Path(STEM).name != STEM or STEM in (".", ".."):
        ap.error("stem must be a filename, not a path")
    if args.imatrix and args.uncalibrated:
        ap.error("--imatrix and --uncalibrated are mutually exclusive")
    if args.imatrix and not args.imatrix.is_file():
        ap.error("imatrix file does not exist")
    if any(t in ("Q4_K_M", "IQ4_XS") for t in args.tiers) and not (args.imatrix or args.uncalibrated):
        ap.error("Q4 tiers require --imatrix or explicit --uncalibrated control")
    MERGED = ROOT / "experiments" / "models" / f"merged_{STEM}"
    MODELS = ROOT / "experiments" / "models"
    MODELS.mkdir(parents=True, exist_ok=True)
    QUANT = os.environ.get("LLAMA_QUANT", str(ROOT / "experiments/bin/llama/llama-b10453/llama-quantize"))
    CONVERT = os.environ.get("LLAMA_CONVERT", str(ROOT / "experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py"))
    if not Path(QUANT).is_file():
        ap.error(f"quantizer not found: {QUANT}")
    if not args.f16 and not Path(CONVERT).is_file():
        ap.error(f"converter not found: {CONVERT}; install llama.cpp at {PIN}, or set LLAMA_CONVERT")
    if not args.f16 and not os.environ.get("LLAMA_CONVERT"):
        actual = subprocess.check_output(["git", "-C", str(Path(CONVERT).parent), "rev-parse", "HEAD"], text=True).strip()
        if actual != PIN:
            ap.error(f"default converter must be pinned to {PIN}; found {actual}")

    if args.f16:
        if not args.f16.is_file():
            ap.error("--f16 file does not exist")
    elif os.environ.get("NO_LORA"):
        # PFT1 full-FT export: the trained dir IS the model — convert in place.
        assert (Path(LORA) / "config.json").exists(), f"{LORA} is not an HF model dir"
        assert not (Path(LORA) / "adapter_config.json").exists(), \
            "NO_LORA=1 but an adapter_config.json is present — wrong mode"
        MERGED = Path(LORA)
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

    f16 = args.f16 or MODELS / f"{STEM}-f16.gguf"
    if not args.f16:
        archs = json.loads((MERGED / "config.json").read_text()).get("architectures", [])
        nextn = ["--no-nextn"] if any("qwen" in a.lower() for a in archs) else []
        subprocess.run([sys.executable, CONVERT, str(MERGED), "--outfile", str(f16),
                        "--outtype", "f16", *nextn], check=True)
    if args.imatrix:
        provenance = args.imatrix.with_suffix(".gguf.json")
        if provenance.is_file():
            recorded = json.loads(provenance.read_text())
            if recorded.get("model_sha256") and recorded["model_sha256"] != sha256(f16):
                ap.error("imatrix provenance does not match this source GGUF")
            if recorded.get("imatrix_sha256") and recorded["imatrix_sha256"] != sha256(args.imatrix):
                ap.error("imatrix file differs from its provenance receipt")
    for tier in dict.fromkeys(args.tiers):
        output = MODELS / f"{STEM}-{tier}.gguf"
        temporary = output.with_suffix(".gguf.partial")
        flags = []
        if tier != "Q8_0":
            flags = ["--output-tensor-type", args.output_type.lower(),
                     "--token-embedding-type", args.embedding_type.lower()]
            if args.imatrix:
                flags += ["--imatrix", str(args.imatrix.resolve())]
        try:
            subprocess.run([QUANT, *flags, str(f16), str(temporary), tier, str(args.threads)], check=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = dict(tier=tier, quantizer=str(Path(QUANT).resolve()),
                       converter=CONVERT if not args.f16 else None,
                       source=str(f16.resolve()), bytes=output.stat().st_size,
                       sha256=sha256(output), output_type=args.output_type if tier != "Q8_0" else None,
                       embedding_type=args.embedding_type if tier != "Q8_0" else None,
                       imatrix_sha256=sha256(args.imatrix) if args.imatrix and tier != "Q8_0" else None)
        output.with_suffix(".gguf.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"OK -> {output}")
    if not args.keep_f16 and not args.f16:
        f16.unlink()


if __name__ == "__main__":
    main()
