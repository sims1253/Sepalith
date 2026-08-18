#!/usr/bin/env python3
"""First datatrove job: exact + MinHash dedup over the CRAN R-corpus.

Reads the per-package shards, keeps areas {R, tests, vignettes, src} (no man/),
runs exact-seq dedup then minhash (normalized) dedup, writes surviving docs to
/mnt/h/sepalith/datasets/corpus_dedup/ + a stats report.
"""
import glob, json, os
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    SequenceExactDedupFilter, MinHashDedupFilter, MinHashDedupSignature,
    MinHashDedupCluster, MinHashDedupConsolidate)

IN_GLOB = "/mnt/h/sepalith/datasets/packages/*.jsonl"
OUT = Path("/mnt/h/sepalith/datasets/corpus_dedup")
WORK = Path("/mnt/h/sepalith/.datatrove-cache")

def keep_only(doc):
    if doc.metadata.get("area") not in ("R", "tests", "vignettes", "src"):
        return False, "area-excluded"
    return True, False

# v1 pragmatic approach: sequence-exact dedup then a single-pass MinHash filter
pipe_exact = [
    JsonlReader(glob.IN_GLOB if False else IN_GLOB),
]
# simpler explicit pipeline using the standalone filters:
executor = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(IN_GLOB),
        datatrove_pipeline_filter(),
        SequenceExactDedupFilter(output_folder=str(WORK / "exact")),
        MinHashDedupSignature(output_folder=str(WORK / "signatures")),
        MinHashDedupConsolidate(output_folder=str(WORK / "consolidated")),
        JsonlWriter(str(OUT), output_filename="corpus_dedup.jsonl"),
    ],
    tasks=8, workers=8, logging_dir=str(WORK / "logs"),
)
