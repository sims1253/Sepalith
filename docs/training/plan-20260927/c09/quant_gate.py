#!/usr/bin/env python3
"""C09: gate every GGUF tier against the bf16 GGUF and write a JSON receipt.

    uv run python docs/training/plan-20260927/c09/quant_gate.py --plan gate-plan.json --output gate-receipt.json

``gate-plan.json`` (``quantize_and_gate.py`` writes it)::

    {"gate_panel": "dev250",
     "reference": {"label": "bf16", "results": {"dev250": "<result.json>", "dev75": "<result.json>"}},
     "candidates": [{"label": "Q4_K_M", "quant_type": "Q4_K_M", "gguf": {"path": ..., "bytes": ..., "sha256": ...},
                     "results": {"dev250": "<result.json>", "dev75": "<result.json>"}, "kld": "<kld.json>"}, ...]}

``gate_panel`` other than ``dev250`` (smoke runs) computes the same paired
statistics on that panel, but no tier can then be gate-eligible.

Result files are ``gguf_dev_eval.py`` outputs (PC result JSONs also load: the
same ``summary`` and ``cases`` layout).  ``kld`` files are the orchestrator's
parsed ``llama-perplexity --kl-divergence`` records.  Missing pieces make a
check ``not_measured`` and the tier ``incomplete``.  Thresholds and the
statistics are in ``sepalith.quant.gate``.  The gate is binding only when
both sides are complete, gate-eligible DEV250 runs (cap 1,024, all 250
cases) from the same llama.cpp build and backend; otherwise each tier is
``not_gate_eligible`` (``checks_verdict`` keeps the raw outcome) and the
numbers are plumbing only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages" / "sepalith" / "src"))
from sepalith.quant import gate
from sepalith.quant.gguf_dev import GATE_CAP, PANELS
from sepalith.quant.llama_tools import file_sha256

SCHEMA = "sepalith.c09.quant-gate.v1"


def load_result(path: str | None):
    if not path:
        return None
    result = json.loads(Path(path).read_text())
    cases = Path(result["cases"]["path"])
    if file_sha256(cases) != result["cases"]["sha256"]:
        raise SystemExit(f"cases file hash differs: {cases}")
    body = json.loads(cases.read_text())
    if body.get("status") != "complete":
        raise SystemExit(f"incomplete cases file: {cases}")
    return {"path": path, "result": result, "results": body["results"]}


def runtime_identity(loaded) -> dict:
    runtime = loaded["result"]["summary"].get("generation_runtime") or {}
    llama = runtime.get("llama_cpp") or {}
    return {"engine": runtime.get("engine", "pc"), "build": runtime.get("build"), "commit": llama.get("commit"),
            "cap": loaded["result"]["summary"]["generation_max_new_tokens"]}


def eligibility(ref, cand) -> list[str]:
    problems = []
    for side, loaded in (("reference", ref), ("candidate", cand)):
        if loaded is None:
            problems.append(f"{side} DEV250 result missing")
            continue
        result = loaded["result"]
        if result["panel"]["sha256"] != PANELS["dev250"][1]:
            problems.append(f"{side} is not DEV250")
        if result.get("gate_eligible") is False or len(loaded["results"]) != PANELS["dev250"][2]:
            problems.append(f"{side} is a partial or smoke run")
        if result["summary"]["generation_max_new_tokens"] != GATE_CAP:
            problems.append(f"{side} cap is not {GATE_CAP}")
    if ref and cand and runtime_identity(ref) != runtime_identity(cand):
        problems.append("reference and candidate ran on different llama.cpp builds or backends")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    plan = json.loads(a.plan.read_text())
    gate_panel = plan.get("gate_panel", "dev250")
    ref250 = load_result(plan["reference"]["results"].get(gate_panel))
    ref75 = load_result(plan["reference"]["results"].get("dev75"))
    rows = []
    for candidate in plan["candidates"]:
        tier = gate.tier_of(candidate["quant_type"])
        cand250 = load_result(candidate["results"].get(gate_panel))
        cand75 = load_result(candidate["results"].get("dev75"))
        metrics = gate.paired_metrics(ref250["results"], cand250["results"]) if ref250 and cand250 else None
        kld = json.loads(Path(candidate["kld"]).read_text())["statistics"] if candidate.get("kld") else None
        checks = gate.gate_checks(tier, metrics, kld)
        problems = eligibility(ref250, cand250)
        checks_verdict = gate.verdict(checks)
        tier_verdict = "not_gate_eligible" if problems else checks_verdict
        parity = None
        if ref75 and cand75:
            parity = gate.dev75_parity(ref75["results"], cand75["results"])
            parity["outcome"] = (gate.PASS if parity["within_tolerance"] else gate.FAIL) if tier == "q8" else gate.RECORDED
        rows.append({"label": candidate["label"], "quant_type": candidate["quant_type"], "tier": tier,
                     "gguf": candidate.get("gguf"), "verdict": tier_verdict, "checks_verdict": checks_verdict,
                     "eligibility_problems": problems,
                     "checks": checks, "ship_net_loss_bound": gate.ship_check(metrics),
                     "shippable": tier_verdict == gate.PASS and gate.ship_check(metrics)["outcome"] == gate.PASS,
                     "gate_panel": gate_panel, "paired": metrics, "dev75_parity": parity, "kld": kld,
                     "inputs": {"results": candidate["results"], "kld": candidate.get("kld")}})
    by_label = {row["label"]: row for row in rows}

    def contrast(arm, base):
        if arm not in by_label or base not in by_label:
            return None
        a_row, b_row = by_label[arm], by_label[base]
        value = lambda row, name: row["checks"][name]["value"]  # noqa: E731
        return {"arm": arm, "base": base,
                "exact_delta_pp": [value(a_row, "exact_delta_pp"), value(b_row, "exact_delta_pp")],
                "mean_kld": [value(a_row, "mean_kld"), value(b_row, "mean_kld")],
                "top1_agreement_pct": [value(a_row, "top1_agreement_pct"), value(b_row, "top1_agreement_pct")],
                "bytes": [(a_row["gguf"] or {}).get("bytes"), (b_row["gguf"] or {}).get("bytes")]}

    shippable = sorted((row for row in rows if row["shippable"]), key=lambda row: (row["gguf"] or {}).get("bytes") or 0)
    receipt = {
        "schema": SCHEMA, "card": "C09", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": {"path": str(a.plan), "sha256": file_sha256(a.plan)},
        "gate_panel": gate_panel,
        "reference": {"label": plan["reference"].get("label"), "results": plan["reference"]["results"],
                      "runtime": runtime_identity(ref250) if ref250 else None},
        "thresholds": gate.THRESHOLDS, "ship_net_loss_upper_bound_pp": gate.SHIP_NET_LOSS_UPPER_BOUND_PP,
        "rules": {"mcnemar": "exact two-sided binomial on discordant pairs; fails only if p <= 0.05 and lost > gained",
                  "nofp_valid_denominator": "percentage points of all panel cases",
                  "net_loss_bound": "Bonett-Price adjusted Wald, one-sided 95%",
                  "dev75_parity": "card C09: Q8_0 DEV75 valid/cap/edit/nofp within 2 of bf16; recorded for other tiers"},
        "tiers": rows,
        "contrasts": {"q4_k_m_plus_vs_q4_k_m": contrast("Q4_K_M+", "Q4_K_M"),
                      "q4_k_m_attn_out_q8_0_vs_q4_k_m": contrast("Q4_K_M-attn-out-q8_0", "Q4_K_M"),
                      "q4_k_m_head_q6_k_vs_q8_0": contrast("Q4_K_M-head-q6_k", "Q4_K_M"),
                      "q4_k_m_imatrix_vs_uncalibrated": contrast("Q4_K_M", "Q4_K_M-uncalibrated")},
        "smallest_shippable": shippable[0]["label"] if shippable else None,
        "decision_note": ("Addendum: ship Q8_0 plus the smallest tier that passes; keep Q4_K_M+ only if it beats "
                          "Q4_K_M on the gate. Thresholds are proposals for the user's sign-off."),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({row["label"]: {"verdict": row["verdict"], "shippable": row["shippable"]} for row in rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
