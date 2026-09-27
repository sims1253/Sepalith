#!/usr/bin/env python3
"""CPU-only byte/source audit of the installed packed-attention implementation."""
from __future__ import annotations
import hashlib, importlib.metadata, json
from pathlib import Path

FILES = {
    "llama": (Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/unsloth/models/llama.py"), "8a79d94ab5f7f76c241d49da9bb1af9a90a9491977d3d08bbf356334dd007f2d"),
    "packing": (Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/unsloth/utils/packing.py"), "f33d388745e6c5701a900ab443ac205d2beefba84831457970dd048cf7160938"),
    "attention_dispatch": (Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/unsloth/utils/attention_dispatch.py"), "9f0b160da7e3105586c252a9f6384218ce00736ecca2b0a72b741412934b5a31"),
    "transformers_flash": (Path("/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/transformers/modeling_flash_attention_utils.py"), "e0430b0c33be1717465122dd442c01436c05e6401727eaa05d60f7ca32f4253c"),
}
REQUIRED = {
    "llama": ("get_packed_info_from_kwargs(kwargs, Q.device)", "select_attention_backend(use_varlen)", "mask_packed_sequence_boundaries"),
    "packing": ("def get_packed_info_from_kwargs", "packed_seq_lengths"),
    "attention_dispatch": ("BlockDiagonalCausalMask", "flash_attn_varlen_func", "build_sdpa_packed_attention_mask"),
}

def main():
    evidence={}
    for key,(path,want) in FILES.items():
        raw=path.read_bytes(); got=hashlib.sha256(raw).hexdigest()
        if got!=want: raise ValueError(f"installed {key} source hash differs")
        text=raw.decode()
        missing=[s for s in REQUIRED.get(key,()) if s not in text]
        if missing: raise ValueError(f"installed {key} lacks required packed seam: {missing}")
        evidence[key]={"path":str(path),"sha256":got,"bytes":len(raw)}
    versions={name:importlib.metadata.version(name) for name in ("unsloth","transformers","torch","xformers","cut-cross-entropy")}
    try: importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError: flash=False
    else: flash=True
    print(json.dumps({"status":"passed_cpu_source_audit","files":evidence,"versions":versions,
                      "flash_attn_distribution_installed":flash,
                      "runtime_backend_requires_root_cuda_probe":True},indent=2,sort_keys=True))

if __name__ == "__main__": main()
