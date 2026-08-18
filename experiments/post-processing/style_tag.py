#!/usr/bin/env python3
"""Style tagging (tidyverse vs base R) per the TOSEM finding: mixed-style
training hurt, style-separated helped. Tags every edit-format dataset row
with a style field so mixtures can stratify/analyze.

Heuristic classifier on the code content (prefix + region_old + region_new):
- tidyverse: pipe operators (%>%, |>), dplyr/tidyr/purrr/ggplot2 calls,
  tidyselect syms (across, where, cur_group), _join verbs
- base: <- with [ / [[ / aggregate / tapply / apply( / sapply( / lapply( /
  data.frame( without pipes
"""
import json, re, sys
from pathlib import Path

TIDY = re.compile(r"%>%|\|>|dplyr::|tidyr::|purrr::|ggplot|mutate\(|summarise\(|summarize\(|filter\(|select\(|group_by\(|across\(|left_join\(|pivot_longer\(|pivot_wider\(|readr::|tibble\(")
BASE = re.compile(r"\bapply\(|\bsapply\(|\blapply\(|\btapply\(|\baggregate\(|\bmerge\(|\bsubset\(|\bwith\(|\bwithin\(|\bdata\.frame\(|\bstrsplit\(|\bgrepl\(|\bregexpr\(|do\.call\(")

def style_of(text: str) -> str:
    t, b = len(TIDY.findall(text)), len(BASE.findall(text))
    if t >= 2 and t > b * 2: return "tidyverse"
    if b >= 2 and b > t * 2: return "base"
    if t > b: return "tidyverse-lean"
    if b > t: return "base-lean"
    return "neutral"

def tag_file(path, text_fields=("prefix", "region_old", "region_new")):
    tmp = str(path) + ".tmp"
    counts = {}
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            try:
                r = json.loads(line)
            except Exception:
                fo.write(line); continue
            if r.get("lang") == "python" or str(r.get("path", "")).endswith(".py"):
                r["style"] = "python"
            else:
                blob = "\n".join(str(r.get(f, "")) for f in text_fields if r.get(f))
                r["style"] = style_of(blob if isinstance(blob, str) else "")
            counts[r["style"]] = counts.get(r["style"], 0) + 1
            fo.write(json.dumps(r) + "\n")
    Path(tmp).replace(path)
    return counts

if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "/mnt/h/sepalith/datasets/edit_pairs_v1/examples.jsonl",
        "/mnt/h/sepalith/datasets/edit_pairs_v1/eval.jsonl",
        "/mnt/h/sepalith/datasets/scenarios_v1/rename_propagation.jsonl",
        "/mnt/h/sepalith/datasets/scenarios_v1/pipe_rewrite.jsonl",
        "/mnt/h/sepalith/datasets/scenarios_v1/format_propagation.jsonl",
        "/mnt/h/sepalith/datasets/scenarios_v1/doc_sync.jsonl",
        "/mnt/h/sepalith/datasets/scenarios_v1/na_rm_propagation.jsonl",
        "/mnt/h/sepalith/datasets/synthetic_analyst_v1/analyst_scripts.jsonl",
    ]
    for t in targets:
        c = tag_file(Path(t))
        print(Path(t).name, json.dumps(c, sort_keys=True), flush=True)
