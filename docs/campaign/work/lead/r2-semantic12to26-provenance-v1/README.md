# DAT-10 semantic 12–26 provenance join

This packet independently checks the frozen semantic records from queue
shards 12–26. It verifies every pinned provenance ledger and candidate packet,
exact queued-ID closure, source-hash equality, train-group/source-line joins,
and the recorded positive license/split/protocol gates. It is review-only and
does not reread raw source files, render targets, or admit training data.

Run with the campaign's low-priority CPU envelope:

```text
cd /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=30s 1200 \
  taskset -c 0,2 nice -n 10 ionice -c 3 \
  /usr/bin/python3 -B docs/campaign/work/lead/r2-semantic12to26-provenance-v1/check.py
```

The input is the frozen queue output under `E:/mnt/e/...`; the independent
result belongs in the campaign receipt named by the task. Root's 6–11
reconstruction and the 27–40 replay preparation remain separate.
