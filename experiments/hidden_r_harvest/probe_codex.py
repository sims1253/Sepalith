"""Probe CodeX shards: fence info-string histogram + cheap R-prefilter hit rate."""
import glob
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

shards = sorted(glob.glob(
    "/home/m0hawk/.cache/huggingface/hub/datasets--Modotte--CodeX-7M-Non-Thinking/snapshots/*/data/*.parquet"))
# sample a few shards spread across the dataset
sample = [shards[0], shards[56], shards[113], shards[170], shards[226]][: int(sys.argv[1]) if len(sys.argv) > 1 else 3]

FENCE_RE = re.compile(r"^```+\s*([A-Za-z0-9+#._-]*)", re.M)
R_TOKEN_RE = re.compile(
    r"\blibrary\(|\brequire\(|%>%|%in%|\bdata\.frame\(|\bstopifnot\(|\bread\.csv\(|\bwrite\.csv\("
    r"|\bsaveRDS\(|\breadRDS\(|\bRcpp::|\[\[Rcpp::export\]\]|<-"
)

fence_langs = Counter()
n_rows = 0
n_fenced = 0
prefilter_hits = 0
r_fence_rows = 0
hit_examples = []

for s in sample:
    pf = pq.ParquetFile(s)
    for batch in pf.iter_batches(batch_size=2000):
        ins = batch.column("input").to_pylist()
        outs = batch.column("output").to_pylist()
        for i, o in zip(ins, outs):
            n_rows += 1
            text = (i or "") + "\n" + (o or "")
            infos = FENCE_RE.findall(text)
            langs_here = [x for x in infos if x]
            if langs_here:
                n_fenced += 1
                fence_langs.update(langs_here)
                if any(x.lower() in ("r", "rlang", "rscript", "splus", "rcode") for x in langs_here):
                    r_fence_rows += 1
            if R_TOKEN_RE.search(text):
                prefilter_hits += 1
                if len(hit_examples) < 400:
                    hit_examples.append((i or "")[:100])

print(f"rows={n_rows} fenced={n_fenced} ({n_fenced/n_rows:.1%}) prefilter_hits={prefilter_hits} "
      f"({prefilter_hits/n_rows:.2%}) r_fence_rows={r_fence_rows} ({r_fence_rows/n_rows:.3%})")
print("\nTop 40 fence langs:")
for lang, c in fence_langs.most_common(40):
    print(f"  {lang!r}: {c}")
