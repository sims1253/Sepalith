#!/usr/bin/env python3
"""SFT v1 dataset: render finish-block records in the Zeta-2 SPM edit format.

Mapping: a finish-block record becomes an edit-prediction example where the
editable region is empty (signature kind: cursor right after '{') or contains
the partial body (mid_body kind). Target = region completion + >>>>>>> UPDATED.
Hold out 5% of PACKAGES (not rows) for eval. Output train/eval JSONL on the NAS.
"""
import json, random, re, sys
from pathlib import Path

SRC = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic/finish_block_sample.jsonl")
OUT = Path("/mnt/h/sepalith/datasets/sft_v1")
MAX_CHARS = 6000        # prompt+target char budget (~1.4k tokens)
MIN_TARGET_CHARS = 30

def render(rec):
    prefix_lines = rec["prefix"].splitlines()
    # region content the user has so far: empty for signature kind; head for mid_body
    if rec["kind"] == "signature":
        region = ["<|user_cursor|>"]
    else:
        head = rec["prefix"].split("{\n", 1)[-1] if "{\n" in rec["prefix"] else ""
        region = [l + ("\n" + "<|user_cursor|>" if i == len(head.splitlines()) - 1 else "")
                  for i, l in enumerate(head.splitlines())] or ["<|user_cursor|>"]
    prompt_lines = (["<[fim-suffix]>"] +
                    [f"<[fim-prefix]><filename>{rec['package']}/{rec['path']}"] +
                    prefix_lines + ["<<<<<<< CURRENT"] + region +
                    ["=======", "<[fim-middle]>"])
    target = rec["target"].rstrip() + "\n>>>>>>> UPDATED"
    return "\n".join(prompt_lines), target

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(SRC)]
    # exact-target dedup
    seen, uniq = set(), []
    for r in recs:
        h = " ".join(r["target"].split())
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    # package-level split
    pkgs = sorted({r["package"] for r in uniq})
    rng = random.Random(11)
    rng.shuffle(pkgs)
    eval_pkgs = set(pkgs[: max(1, len(pkgs) // 20)])
    n_train = n_eval = n_drop = 0
    with open(OUT / "train.jsonl", "w") as tr, open(OUT / "eval.jsonl", "w") as ev:
        for r in uniq:
            prompt, target = render(r)
            if len(prompt) + len(target) > MAX_CHARS or len(target) < MIN_TARGET_CHARS:
                n_drop += 1
                continue
            row = dict(text=prompt + target, prompt=prompt, target=target,
                       kind=r["kind"], gated=r["gated"], package=r["package"])
            (ev if r["package"] in eval_pkgs else tr).write(json.dumps(row) + "\n")
            n_eval += r["package"] in eval_pkgs
            n_train += r["package"] not in eval_pkgs
    stats = dict(source=len(recs), after_dedup=len(uniq), train=n_train, eval=n_eval,
                 dropped_len=n_drop, eval_packages=len(eval_pkgs),
                 gated_frac=round(sum(r["gated"] for r in uniq) / len(uniq), 3))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
