import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import snapshot_download

repo = sys.argv[1]
t0 = time.time()
p = snapshot_download(
    repo_id=repo, repo_type="dataset",
    allow_patterns=["data/*.parquet"],
    max_workers=8,
)
print(f"DONE {repo} -> {p} in {time.time()-t0:.0f}s", flush=True)
