# Cloud CPT parent merge binder v2

This fresh binder fixes the frozen v1 provenance defect. Both `--registry` and `--merge-source` are required. Their resolved paths and exact byte hashes enter the review envelope before its canonical binding review hash is computed. A copied or root-owned 1902 registry therefore cannot be mislabeled with the old 1585 registry digest.

Example after root writes a verified 1902 registry:

```
PYTHONNOUSERSITE=1 taskset -c 0,2 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python prepare_binding_v2.py \
  --registry /ABSOLUTE/root-owned-candidates-1902.json \
  --merge-source /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cloud-cpt-parent-merge-preparation-v1/source/merge_cloud_cpt_parent.py \
  --candidate 1902 --review-output /ABSOLUTE/root-binding-review-1902.json
```

After root admission, rerun with a fresh review output plus `--root-admission` and `--binding-output`. This packet does not authorize or run a merge.
