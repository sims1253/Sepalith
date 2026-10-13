"""Locating and describing the pinned llama.cpp b10453 install (C09 prep, PR #51)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import struct
import subprocess

DEFAULT_LLAMA_CPP = Path.home() / ".local" / "opt" / "llama.cpp-b10453"
BUILDS = ("cpu", "cuda")
SEALED_MARKERS = ("DAT-07-final", "sealed-final")
NICE = ("nice", "-n", "10")


def refuse_sealed(path: Path | str) -> None:
    """Fail before touching anything that names the sealed final set."""
    if any(marker in str(path) for marker in SEALED_MARKERS):
        raise SystemExit(f"refusing a sealed final-set path: {path}")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def llama_root() -> Path:
    return Path(os.environ.get("SEPALITH_LLAMA_CPP", str(DEFAULT_LLAMA_CPP)))


def llama_bin(name: str, build: str = "cpu") -> Path:
    if build not in BUILDS:
        raise ValueError(f"unknown llama.cpp build {build!r}")
    path = llama_root() / f"build-{build}" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; build llama.cpp with scripts/packaging/build_llama_cpp.sh")
    return path


def tool_env(build: str) -> dict[str, str]:
    """CPU runs hide every CUDA device; CUDA runs keep the caller's lease."""
    env = dict(os.environ)
    if build == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def llama_version(name: str, build: str = "cpu") -> dict[str, object]:
    path = llama_bin(name, build)
    done = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    text = (done.stdout + done.stderr).strip()
    match = re.search(r"build (\d+), commit ([0-9a-f]+)", text)
    return {"binary": str(path), "sha256": file_sha256(path), "build": build,
            "version": next((line for line in text.splitlines() if line.startswith("version:")), text[-200:]),
            "build_number": int(match.group(1)) if match else None, "commit": match.group(2) if match else None}


def gpu_layers(build: str) -> list[str]:
    return ["-ngl", "999"] if build == "cuda" else ["-ngl", "0"]


_KLD_PATTERNS = {
    "mean_kld": r"Mean\s+KLD:\s+([-\d.eE+]+)\s+±\s+([-\d.eE+]+)",
    "median_kld": r"Median\s+KLD:\s+([-\d.eE+]+)",
    "p99_kld": r"99\.0%\s+KLD:\s+([-\d.eE+]+)",
    "max_kld": r"Maximum KLD:\s+([-\d.eE+]+)",
    "same_top_p_pct": r"Same top p:\s+([-\d.eE+]+)\s+±\s+([-\d.eE+]+)\s*%",
    "rms_dp_pct": r"RMS Δp\s*:\s+([-\d.eE+]+)\s+±\s+([-\d.eE+]+)\s*%",
    "mean_ppl_q": r"Mean PPL\(Q\)\s*:\s+([-\d.eE+]+)\s+±\s+([-\d.eE+]+)",
    "mean_ppl_base": r"Mean PPL\(base\)\s*:\s+([-\d.eE+]+)\s+±\s+([-\d.eE+]+)",
}


def parse_kld_log(text: str) -> dict[str, float]:
    """Read llama-perplexity --kl-divergence summary statistics (b10453 wording)."""
    found: dict[str, float] = {}
    for name, pattern in _KLD_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            found[name] = float(match.group(1))
            if match.lastindex and match.lastindex > 1:
                found[name + "_uncertainty"] = float(match.group(2))
    if "mean_kld" not in found or "same_top_p_pct" not in found:
        raise ValueError("llama-perplexity output lacks the KLD summary")
    return found


def read_kld_base_tokens(path: Path | str) -> dict[str, object]:
    """Header and token stream of a ``--kl-divergence-base`` file.

    Layout (perplexity.cpp at b10453): ``_logits_``, uint32 n_ctx, int32
    n_vocab, int32 n_chunk, then n_chunk*n_ctx int32 tokens as tokenized
    (before the per-chunk BOS overwrite), then the log-probabilities.
    """
    with open(path, "rb") as stream:
        if stream.read(8) != b"_logits_":
            raise ValueError(f"{path} is not a llama-perplexity base file")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", stream.read(12))
        count = n_ctx * n_chunk
        raw = stream.read(4 * count)
        if len(raw) != 4 * count:
            raise ValueError("base file token stream is truncated")
        tokens = list(struct.unpack(f"<{count}i", raw))
    return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk, "tokens": tokens}
