"""POC-DIFF: masked-diffusion NSE twin vs the AR-FIM twin.

Plan: docs/research/poc-diff-twin-plan-2026-08-26.md. Shared constants for
the package; every module is runnable standalone via
`uv run python -m experiments.training.poc_diff.<mod>` from the repo root.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- vocabulary extension (frozen in the plan, Task 2) ---
BASE_VOCAB = 130_560          # MiniCPM5 tokenizer, as trained in poc_twin
MASK_ID = 130_560             # [MASK] embedding row added for the MD twin
EMPTY_ID = 130_561            # [EMPTY] class token for no-op spans
VOCAB_MD = 130_562            # resized embedding table / tied head

# --- PSM row anatomy (verified against astfim_v1/fixed, 2026-08-26) ---
# text   = prefix_part + span_text + "\n<|end|>"
# prompt = prefix_part + "\n<|end|>\n"      (span removed, terminator line)
# target = span_text + "\n<|end|>"
END_MARKER = "\n<|end|>"          # terminates the span inside text/target
PROMPT_TERMINATOR = "\n<|end|>\n"  # terminates the prompt field

# --- pre-registered caps (frozen before any run) ---
MAX_SPAN_TOK = 256    # longer spans dropped from BOTH arms' train+eval
PROMPT_TOK_MAX = 640  # ladder pattern; keeps prompt+span inside 1024 ctx
CTX_TOK = 1024

# --- paths ---
TMP = "/tmp/poc_diff"                                   # tmpfs staging
SRC_TRAIN = "/mnt/h/sepalith/datasets/astfim_v1/fixed/train.jsonl"
SRC_EVAL = "/mnt/h/sepalith/datasets/astfim_v1/fixed/eval.jsonl"
OUT_RUNS = "/mnt/h/sepalith/runs/poc_diff"              # final artifacts


def span_region(span_ids):
    """Span token region as the model sees it: the tokens themselves, or
    the single [EMPTY] class token when the span is empty (no-op)."""
    return list(span_ids) if span_ids else [EMPTY_ID]
