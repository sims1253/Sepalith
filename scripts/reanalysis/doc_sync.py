#!/usr/bin/env python3
"""Prepare anonymous, deduplicated doc_sync packets from frozen raw responses."""

import argparse
import hashlib
import json
from pathlib import Path

from common import Evidence, dump, keyed
from doc_prompts import additions


def prepare(e):
    renderer = e.module("/experiments/eval/run_eval.py")
    native = e.module("/experiments/eval/landscape_zeta_render.py")
    prompt_additions = additions(e, renderer)
    holdout = keyed(
        [
            dict(r, id=hashlib.sha1(r["prompt"].encode()).hexdigest()[:12])
            for r in e.rows("/sft_v3/eval.jsonl")
            if r.get("family") == "doc_sync"
        ]
    )
    cases = {}
    for r in e.rows("/scenarios_v1/doc_sync.jsonl"):
        prompt = renderer.render_zeta2(dict(r, suffix=[]))
        rid = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if rid in holdout:
            if rid in cases:
                raise ValueError("Duplicate scenario ID")
            if (
                holdout[rid]["target"]
                != "\n".join(r["region_new"]).rstrip() + "\n>>>>>>> UPDATED"
            ):
                raise ValueError("Target changed")
            cases[rid] = dict(r, prompt=prompt)
    if len(cases) != 15:
        raise ValueError(f"Expected 15 cases, got {len(cases)}")
    packets, mapping, arms = {}, [], {}
    for f in e.records:
        if "/results_scenarios_" not in f["path"] or not f["path"].endswith(".jsonl"):
            continue
        rows = [r for r in e.rows(f["path"]) if r.get("family") == "doc_sync"]
        if not rows:
            continue
        arm = Path(f["path"]).stem.removeprefix("results_scenarios_")
        keyed(rows)
        if {r["id"] for r in rows} != set(cases):
            arms[arm] = {"available": len(rows), "complete": False}
        else:
            arms[arm] = {"available": len(rows), "complete": True}
        for r in rows:
            if r["id"] not in cases:
                raise ValueError("Unknown doc_sync row")
            case = cases[r["id"]]
            if any(case[k] != r[k] for k in ("package", "path")):
                raise ValueError("Scenario provenance mismatch")
            request_prompt = case["prompt"]
            if arm == "glm53_3shot":
                request_prompt += "\n\n" + prompt_additions["three_shot"]
            elif arm in {"glm53_zeroshot", "gemma4e2b_zeroshot"}:
                request_prompt += "\n\n" + prompt_additions["FMT_INSTRUCTION"]
                if r.get("oneshot"):
                    request_prompt += prompt_additions["ONESHOT_EXAMPLE"]
            elif arm == "zeta21_native":
                request_prompt = native.build_v0318_prompt(
                    case["prefix"],
                    case["region_old"],
                    case["cursor_idx"],
                    case["path"],
                    native.diff_body_from_event_diff(case["event_diff"]),
                )[0]
            raw = r.get("raw")
            evidence_status = (
                "missing_raw"
                if raw is None
                else "truncated_raw"
                if arm == "zeta21_native" and len(raw) >= 400
                else "truncated_raw"
                if arm in {"gemma4e2b_zeroshot", "glm53_zeroshot", "glm53_3shot"}
                and len(raw) >= 600
                else "complete"
            )
            pred = renderer.parse_pred("zeta2", raw) if raw is not None else []
            if arm == "zeta21_native" and evidence_status == "complete":
                _, _, editable, _ = native.build_v0318_prompt(
                    case["prefix"],
                    case["region_old"],
                    case["cursor_idx"],
                    case["path"],
                    native.diff_body_from_event_diff(case["event_diff"]),
                )
                ok, updated = native.apply_marker_span_v0318(editable, raw)
                pred = (
                    native.extract_region_lines(
                        editable, updated, case["region_old"], len(case["prefix"])
                    )
                    if ok
                    else []
                )
                pred = [l.replace("<|user_cursor|>", "").rstrip() for l in (pred or [])]
                while pred and not pred[0]:
                    pred.pop(0)
                while pred and not pred[-1]:
                    pred.pop()
            if evidence_status == "complete" and (
                "\n".join(pred)[:400] != r["pred"] or len(pred) != r["n_pred_lines"]
            ):
                raise ValueError(f"Parsed response mismatch: {arm} {r['id']}")
            digest = hashlib.sha256(
                json.dumps([r["id"], raw], ensure_ascii=False).encode()
            ).hexdigest()
            pid = digest[:16]
            prompt_variants = set(packets.get(pid, {}).get("request_prompts", []))
            prompt_variants.add(request_prompt)
            packets[pid] = {
                "packet": pid,
                "case": r["id"],
                "prompt": case["prompt"],
                "request_prompts": sorted(prompt_variants),
                "old": case["region_old"],
                "target": case["region_new"],
                "raw": raw,
                "prediction": pred,
            }
            mapping.append(
                {
                    "arm": arm,
                    "case": r["id"],
                    "packet": pid,
                    "exact": r["exact"],
                    "valid_pass": r["valid_pass"],
                    "fail_kind": r.get("fail_kind"),
                    "valid_reason": r.get("valid_reason"),
                    "request_prompt_sha256": hashlib.sha256(
                        request_prompt.encode()
                    ).hexdigest(),
                    "evidence_status": evidence_status,
                    "source_sha256": f["sha256"],
                }
            )
    return cases, [packets[k] for k in sorted(packets)], mapping, arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", type=Path, required=True, help="PRIVATE output directory")
    a = p.parse_args()
    cases, packets, mapping, arms = prepare(Evidence(a.inventory, a.snapshot))
    a.out.mkdir(parents=True, exist_ok=True)
    dump(a.out / "cases.json", cases)
    dump(a.out / "packets.json", packets)
    dump(a.out / "unblind.json", mapping)
    dump(a.out / "arms.json", arms)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "unique_packets": len(packets),
                "outputs": len(mapping),
                "arms": len(arms),
            }
        )
    )


if __name__ == "__main__":
    main()
