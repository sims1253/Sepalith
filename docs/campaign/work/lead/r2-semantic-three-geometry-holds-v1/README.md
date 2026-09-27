# Three geometry holds

This packet narrowly re-evaluates three shard-0011 TRAIN rows held by the older full-document geometry helper. The pinned sources contain both LF and CRLF line endings. The old helper selected one delimiter for the entire file, so its line-block and blank-gap checks used the wrong line coordinates.

`repair_geometry.py` uses the already reviewed semantic occurrence method and span. It maps normalized LF byte boundaries back to exact raw source byte boundaries, requires one target occurrence and immediate attachment to the asserted definition, validates the original target-free candidate window, and proves byte-exact full-source reapplication while preserving every non-target raw byte. It does not infer an occurrence from target text or loosen identity checks.

The outputs remain review-only and require root admission. Run `bash run.sh` once at a fresh output path. Run tests with `TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp python3 -m unittest discover -s tests -v`.
