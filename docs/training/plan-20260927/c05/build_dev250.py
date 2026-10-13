"""C05 DEV250, phase 2: render, allocate and write the panel.

Reads phase-1 packets (build_dev250_packets.py), runs each through
sepalith.training.rl.campaign_token_audit.prepare_candidate (PRM05 source
window + zeta2-prm03-v1 render, the path DEV75's candidates took), drops
unusable candidates, allocates the quotas under the 3-cases-per-group cap
with a max-flow, and writes dev250-v1.jsonl plus manifest.json.

    PYTHONPATH=packages/sepalith/src python build_dev250.py PHASE1_DIR OUT_DIR
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median

from sepalith.campaign_protocol import (
    RENDERER_ID, TOKENIZER_CONFIG_SHA256, TOKENIZER_JSON_SHA256,
    PromptContext, build_training_row,
)
from sepalith.training.rl.campaign_token_audit import prepare_candidate, rendered_collisions

TOKENIZER = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0")
DEV75 = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/"
             "lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl")
SPLIT = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json")

GROUP_CAP = 3
GENERATION_CAP = 1024   # RULES.md max_new_tokens; longer references could never match
MAX_SEQUENCE = 4096
# Quotas: status file C05.json, card_quotas_250 and the 2026-10-03 deviation.
QUOTAS = {"finish_authored": 55, "finish_cut": 56, "no_op": 50, "pipe_rewrite": 28,
          "format_propagation": 24, "rename_propagation": 23, "roxygen_drafting": 11,
          "na_rm_propagation": 3}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slot(packet) -> str:
    if packet["family"] == "finish_block":
        return "finish_authored" if packet["meta"]["source"] == "authored" else "finish_cut"
    return packet["family"]


def max_flow_allocation(candidates, quotas, group_cap):
    """Integer max-flow source -> slot -> group -> sink (Edmonds-Karp with
    sorted adjacency, so the result is deterministic)."""
    cap = defaultdict(int)
    adj = defaultdict(set)

    def edge(u, v, c):
        cap[(u, v)] += c
        adj[u].add(v)
        adj[v].add(u)

    supply = Counter((slot(c["packet"]), c["packet"]["row_ref"]["group_id"]) for c in candidates)
    for s, q in quotas.items():
        edge("S", ("slot", s), q)
    for (s, g), n in sorted(supply.items()):
        edge(("slot", s), ("group", g), n)
    for g in sorted({g for _, g in supply}):
        edge(("group", g), "T", group_cap)
    order = {n: i for i, n in enumerate(sorted(adj, key=repr))}
    flow = defaultdict(int)
    while True:
        parent, queue = {"S": None}, deque(["S"])
        while queue and "T" not in parent:
            u = queue.popleft()
            for v in sorted(adj[u], key=order.get):
                if v not in parent and cap[(u, v)] - flow[(u, v)] > 0:
                    parent[v] = u
                    queue.append(v)
        if "T" not in parent:
            break
        path, v = [], "T"
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        push = min(cap[e] - flow[e] for e in path)
        for u, v in path:
            flow[(u, v)] += push
            flow[(v, u)] -= push
    return {(s[1], g[1]): flow[(s, g)] for (s, g) in list(flow)
            if isinstance(s, tuple) and s[0] == "slot" and isinstance(g, tuple)
            and g[0] == "group" and flow[(s, g)] > 0}


def case_record(c):
    packet, cand, row = c["packet"], c["candidate"], c["candidate"]["row"]
    ref = packet["row_ref"]
    return {
        "id": ref["row_id"],
        "family": packet["family"],
        "operation": packet["result"]["operation"],
        "package_id": row["package_id"],
        "group_id": ref["group_id"],
        "split": "dev",
        "admitted_for_training": False,
        "candidate_status": cand["status"],
        "context": cand["context"],
        "region_new": list(packet["result"]["target_body"]),
        "target_body_text": "\n".join(packet["result"]["target_body"]),
        "prompt_sha256": cand["prompt_sha256"],
        "target_sha256": cand["target_sha256"],
        "target_body_token_count": row["target_body_token_count"],
        "target_terminal_token_count": row["target_terminal_token_count"],
        "selection": cand["selection"],
        "source_ref": ref,
        "source_provenance": cand["source_provenance"],
        "strata": {"family": packet["family"], "slot": slot(packet), "group_id": ref["group_id"],
                   **packet["meta"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase1", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    for name, expected in (("tokenizer.json", TOKENIZER_JSON_SHA256),
                           ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256)):
        assert sha256((TOKENIZER / name).read_bytes()) == expected, name
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True,
                                              trust_remote_code=False)

    dev75 = [json.loads(line) for line in DEV75.open()]
    dev75_prompts = {c["prompt_sha256"] for c in dev75}
    dev75_ids = {c["id"] for c in dev75}
    dev75_groups = {c["group_id"] for c in dev75}

    packets_path = args.phase1 / "packets.jsonl"
    packets = [json.loads(line) for line in packets_path.open()]
    usable, excluded = [], Counter()
    for packet in packets:
        try:
            cand = prepare_candidate(packet, tokenizer)
        except (ValueError, KeyError, TypeError) as e:
            excluded[f"{packet['family']}:{e}"] += 1
            continue
        lengths = cand["lengths"]
        if lengths["response_with_terminal_eos"] > GENERATION_CAP:
            excluded[f"{packet['family']}:reference_exceeds_generation_cap"] += 1
            continue
        if lengths["sequence"] > MAX_SEQUENCE:
            excluded[f"{packet['family']}:sequence_over_{MAX_SEQUENCE}"] += 1
            continue
        if cand["prompt_sha256"] in dev75_prompts or packet["row_ref"]["row_id"] in dev75_ids:
            excluded[f"{packet['family']}:same_as_dev75"] += 1
            continue
        usable.append({"packet": packet, "candidate": cand})

    # One case per rendered prompt: later duplicates are dropped.
    seen, unique = set(), []
    for c in usable:
        if c["candidate"]["prompt_sha256"] in seen:
            excluded[f"{c['packet']['family']}:duplicate_prompt"] += 1
            continue
        seen.add(c["candidate"]["prompt_sha256"])
        unique.append(c)

    alloc = max_flow_allocation(unique, QUOTAS, GROUP_CAP)
    take = Counter()
    chosen = []
    for c in unique:
        key = (slot(c["packet"]), c["packet"]["row_ref"]["group_id"])
        if take[key] < alloc.get(key, 0):
            take[key] += 1
            chosen.append(c)
    by_slot = Counter(slot(c["packet"]) for c in chosen)
    shortfall = {s: q - by_slot[s] for s, q in QUOTAS.items() if by_slot[s] < q}

    cases = sorted((case_record(c) for c in chosen), key=lambda r: (r["family"], r["id"]))
    # Re-render through the evaluator's own path as a check.
    for case in cases:
        ctx = PromptContext.from_mapping(case["context"])
        row = build_training_row(ctx, operation=case["operation"], region_new=case["region_new"],
                                 tokenizer=tokenizer, row_id=case["id"], family=case["family"],
                                 package_id=case["package_id"], split="dev")
        assert sha256(row["prompt_text"].encode()) == case["prompt_sha256"], case["id"]
        assert sha256(row["target_text"].encode()) == case["target_sha256"], case["id"]
    panel = args.out / "dev250-v1.jsonl"
    panel.write_text("".join(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n" for c in cases))

    split = json.loads(SPLIT.read_text())
    split_of = {g["group_id"]: g["split"] for g in split["groups"]}
    groups = Counter(c["group_id"] for c in cases)
    target_tokens = [c["target_body_token_count"] for c in cases]
    by_family_tokens = defaultdict(list)
    for c in cases:
        by_family_tokens[c["family"]].append(c["target_body_token_count"])

    def dist(values):
        values = sorted(values)
        q = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
        return {"n": len(values), "min": values[0], "p25": q(0.25), "median": median(values),
                "p75": q(0.75), "p95": q(0.95), "max": values[-1]}

    manifest = {
        "panel": "DEV250 v1",
        "card": "C05",
        "renderer_id": RENDERER_ID,
        "file": {"path": str(panel), "sha256": sha256(panel.read_bytes()), "cases": len(cases)},
        "case_ids_sha256": sha256("\n".join(c["id"] for c in cases).encode()),
        "quotas": QUOTAS,
        "shortfall": shortfall,
        "counts": {"by_family": dict(Counter(c["family"] for c in cases)),
                   "by_slot": dict(by_slot),
                   "by_operation": dict(Counter(c["operation"] for c in cases)),
                   "authored_by_backend": dict(Counter(c["strata"].get("backend") for c in cases
                                                       if c["strata"]["slot"] == "finish_authored"))},
        "groups": {"distinct": len(groups), "max_cases_per_group": max(groups.values()),
                   "cases_per_group_hist": dict(Counter(groups.values())),
                   "split_of_groups": dict(Counter(split_of.get(g, "c05_authored_dev_only")
                                                   for g in groups)),
                   "shared_with_dev75": sorted(set(groups) & dev75_groups)},
        "disjointness": {
            "case_ids_shared_with_dev75": len({c["id"] for c in cases} & dev75_ids),
            "prompts_shared_with_dev75": len({c["prompt_sha256"] for c in cases} & dev75_prompts),
            "groups_in_train_split": sum(1 for g in groups if split_of.get(g) == "train_group"),
            "rendered_prompt_collisions": len(rendered_collisions([c["candidate"] for c in chosen])),
        },
        "target_body_tokens": {"all": dist(target_tokens),
                               "by_family": {f: dist(v) for f, v in sorted(by_family_tokens.items())}},
        "inputs": {
            "phase1_packets": {"path": str(packets_path), "sha256": sha256(packets_path.read_bytes()),
                               "packets": len(packets)},
            "dev75": {"path": str(DEV75), "sha256": sha256(DEV75.read_bytes())},
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        },
        "filters": {"generation_cap": GENERATION_CAP, "max_sequence": MAX_SEQUENCE,
                    "group_cap": GROUP_CAP, "one_case_per_rendered_prompt": True},
        "excluded_candidates": dict(excluded),
        "candidates_after_filters": len(unique),
        "final_files_opened": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("file", "shortfall", "counts", "groups",
                                               "disjointness")}, indent=2))


if __name__ == "__main__":
    main()
