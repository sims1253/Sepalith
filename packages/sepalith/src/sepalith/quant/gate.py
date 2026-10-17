"""The C09/C17 quant gate: each GGUF tier against the bf16 GGUF reference.

Thresholds are the proposal in the 2026-10-12 quantization addendum
(``docs/training/plan-20260927/research/wiki-and-trl-review-20261011.md`` on
the research branch), approved by the user for C09:

| Metric | Q8_0 | Q6_K, Q5_K_M | Q4_K_M |
| --- | --- | --- | --- |
| DEV250 exact-match change | >= -0.4 pp | >= -1.0 pp | >= -2.0 pp |
| Paired McNemar on exact match | p > 0.05 | p > 0.05 | p > 0.05 |
| ``nofp`` | no worse | at most +1 pp | at most +1 pp |
| ``valid`` (paired) | no significant net loss | same | same |
| Whole-output token identity vs bf16 | >= 97% | >= 90% | recorded |
| Mean KLD on held-out TRAIN text | <= 0.002 | <= 0.01 | <= 0.03 |
| Top token agrees with bf16 | >= 99.5% | >= 98.5% | >= 97% |

and every shipped tier needs a one-sided 95% upper bound of at most 3 pp on
net exact-match loss.

Choices this module makes where the table is silent (recorded in every receipt):

- McNemar is the exact two-sided binomial test on the discordant pairs.  It
  fails a tier only when p <= 0.05 *and* the quant loses more cases than it
  gains; a significant improvement is reported, not failed.
- ``nofp`` changes are percentage points of the whole panel (250 cases on
  DEV250) and count no-op cases with a suggested edit.  "no worse" means a
  change of 0.
- ``valid`` (protocol-valid outputs) uses the same paired rule as McNemar
  on exact match: it fails only when the exact two-sided test on the
  discordant pairs has p <= 0.05 *and* the quant loses more valid outputs
  than it gains.  The drop in pp is recorded, not gated.  User decision
  2026-10-17 (C09): every lost valid case in the first gate run was a
  cap-1,024 run-on on a case bf16 also got wrong, and PC bf16 vs GGUF bf16
  already differ by 5 valid cases, so a fixed pp threshold sat below the
  noise floor.
- The net-loss bound is the Bonett-Price adjusted Wald interval for a paired
  difference of proportions (one-sided, z = 1.645), which stays informative
  when there are no discordant pairs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

Z_ONE_SIDED_95 = 1.6448536269514722
SHIP_NET_LOSS_UPPER_BOUND_PP = 3.0
MCNEMAR_ALPHA = 0.05

# Token identity and top-1 agreement are recorded, not gated (user decision 2026-10-12):
# near-ties flip them without changing edits, which the DEV250 exact-match checks cover.
THRESHOLDS: dict[str, dict[str, float | None]] = {
    "q8": {"exact_delta_pp_min": -0.4, "nofp_rise_pp_max": 0.0, "valid_drop_pp_max": None,
           "token_identity_min_pct": None, "mean_kld_max": 0.002, "top1_agreement_min_pct": None},
    "q6_q5": {"exact_delta_pp_min": -1.0, "nofp_rise_pp_max": 1.0, "valid_drop_pp_max": None,
              "token_identity_min_pct": None, "mean_kld_max": 0.01, "top1_agreement_min_pct": None},
    "q4": {"exact_delta_pp_min": -2.0, "nofp_rise_pp_max": 1.0, "valid_drop_pp_max": None,
           "token_identity_min_pct": None, "mean_kld_max": 0.03, "top1_agreement_min_pct": None},
}
TIER_OF_TYPE = {"Q8_0": "q8", "Q6_K": "q6_q5", "Q5_K_M": "q6_q5", "Q4_K_M": "q4", "IQ4_XS": "q4"}

PASS, FAIL, NOT_MEASURED, RECORDED = "pass", "fail", "not_measured", "recorded"


def tier_of(quant_type: str) -> str:
    if quant_type not in TIER_OF_TYPE:
        raise ValueError(f"no gate tier for quant type {quant_type!r}")
    return TIER_OF_TYPE[quant_type]


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed exactly in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    top = max(log_terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in log_terms))


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    if lost < 0 or gained < 0:
        raise ValueError("discordant counts are nonnegative")
    n = lost + gained
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binomial_cdf(min(lost, gained), n))


def net_loss_upper_bound_pp(lost: int, gained: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided upper bound (pp) on P(ref right, quant wrong) - P(ref wrong, quant right).

    Bonett and Price (2012) adjusted Wald: add one to each discordant cell and
    two to n.
    """
    if n < 1 or lost < 0 or gained < 0 or lost + gained > n:
        raise ValueError("invalid paired counts")
    m = n + 2
    p_lost, p_gained = (lost + 1) / m, (gained + 1) / m
    difference = p_lost - p_gained
    variance = max(0.0, (p_lost + p_gained - difference ** 2) / m)
    return 100.0 * (difference + z * math.sqrt(variance))


def load_cases(results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index = {str(r["id"]): r for r in results}
    if len(index) != len(results) or not index:
        raise ValueError("empty or duplicate case ids")
    return index


def paired_metrics(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exact-match, validity, no-op false positive and token-identity comparison per case."""
    ref, cand = load_cases(reference), load_cases(candidate)
    if ref.keys() != cand.keys():
        raise ValueError("case ids differ between reference and candidate")
    n = len(ref)
    ids = sorted(ref)

    def count(index: Mapping[str, Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], Any]) -> int:
        return sum(1 for case_id in ids if predicate(index[case_id]))

    def nofp(row: Mapping[str, Any]) -> bool:
        return bool(row["expected_noop"]) and bool(row["suggestion"])

    lost = sum(1 for i in ids if ref[i]["exact_region"] and not cand[i]["exact_region"])
    valid_lost = sum(1 for i in ids if ref[i]["protocol_valid"] and not cand[i]["protocol_valid"])
    valid_gained = sum(1 for i in ids if cand[i]["protocol_valid"] and not ref[i]["protocol_valid"])
    gained = sum(1 for i in ids if cand[i]["exact_region"] and not ref[i]["exact_region"])
    identical = sum(1 for i in ids if list(ref[i]["generated_ids"]) == list(cand[i]["generated_ids"]))
    exact_ref, exact_cand = count(ref, lambda r: r["exact_region"]), count(cand, lambda r: r["exact_region"])
    valid_ref, valid_cand = count(ref, lambda r: r["protocol_valid"]), count(cand, lambda r: r["protocol_valid"])
    nofp_ref, nofp_cand = count(ref, nofp), count(cand, nofp)
    noop_cases = count(ref, lambda r: r["expected_noop"])
    return {
        "cases": n, "noop_cases": noop_cases,
        "exact": {"reference": exact_ref, "candidate": exact_cand, "delta_pp": 100.0 * (exact_cand - exact_ref) / n,
                  "lost": lost, "gained": gained, "mcnemar_p_two_sided": mcnemar_exact(lost, gained),
                  "net_loss_pp": 100.0 * (lost - gained) / n,
                  "net_loss_upper_bound_pp": net_loss_upper_bound_pp(lost, gained, n)},
        "valid": {"reference": valid_ref, "candidate": valid_cand, "drop_pp": 100.0 * (valid_ref - valid_cand) / n,
                  "lost": valid_lost, "gained": valid_gained,
                  "mcnemar_p_two_sided": mcnemar_exact(valid_lost, valid_gained)},
        "nofp": {"reference": nofp_ref, "candidate": nofp_cand, "rise_pp": 100.0 * (nofp_cand - nofp_ref) / n,
                 "rise_pp_of_noop_cases": 100.0 * (nofp_cand - nofp_ref) / noop_cases if noop_cases else None},
        "cap": {"reference": count(ref, lambda r: r["cap_hit"]), "candidate": count(cand, lambda r: r["cap_hit"])},
        "score": {"reference": count(ref, lambda r: (not r["expected_noop"] and r["exact_region"]) or (r["expected_noop"] and r["predicted_noop"])),
                  "candidate": count(cand, lambda r: (not r["expected_noop"] and r["exact_region"]) or (r["expected_noop"] and r["predicted_noop"]))},
        "token_identity": {"identical": identical, "pct": 100.0 * identical / n},
        "changed_case_ids": [i for i in ids if list(ref[i]["generated_ids"]) != list(cand[i]["generated_ids"])],
        "lost_case_ids": [i for i in ids if ref[i]["exact_region"] and not cand[i]["exact_region"]],
        "gained_case_ids": [i for i in ids if cand[i]["exact_region"] and not ref[i]["exact_region"]],
    }


def _check(value: float | None, threshold: float | None, *, minimum: bool) -> str:
    if threshold is None:
        return RECORDED
    if value is None:
        return NOT_MEASURED
    return PASS if (value >= threshold if minimum else value <= threshold) else FAIL


def gate_checks(tier: str, metrics: Mapping[str, Any] | None, kld: Mapping[str, float] | None) -> dict[str, dict[str, Any]]:
    """Per-check outcomes for one candidate; missing measurements are ``not_measured``."""
    limits = THRESHOLDS[tier]
    exact = metrics["exact"] if metrics else None
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, value: float | None, threshold: float | None, minimum: bool, rule: str) -> None:
        checks[name] = {"value": value, "threshold": threshold, "rule": rule,
                        "outcome": _check(value, threshold, minimum=minimum)}

    add("exact_delta_pp", exact["delta_pp"] if exact else None, limits["exact_delta_pp_min"], True, ">=")
    if exact:
        p, losing = exact["mcnemar_p_two_sided"], exact["lost"] > exact["gained"]
        outcome = FAIL if (p <= MCNEMAR_ALPHA and losing) else PASS
        checks["mcnemar"] = {"value": p, "threshold": MCNEMAR_ALPHA, "rule": "fail if p <= 0.05 and lost > gained",
                             "lost": exact["lost"], "gained": exact["gained"], "outcome": outcome}
    else:
        checks["mcnemar"] = {"value": None, "threshold": MCNEMAR_ALPHA, "rule": "fail if p <= 0.05 and lost > gained",
                             "outcome": NOT_MEASURED}
    add("nofp_rise_pp", metrics["nofp"]["rise_pp"] if metrics else None, limits["nofp_rise_pp_max"], False, "<=")
    add("valid_drop_pp", metrics["valid"]["drop_pp"] if metrics else None, limits["valid_drop_pp_max"], False, "<=")
    if metrics:
        valid = metrics["valid"]
        p, losing = valid["mcnemar_p_two_sided"], valid["lost"] > valid["gained"]
        checks["valid_mcnemar"] = {"value": p, "threshold": MCNEMAR_ALPHA, "rule": "fail if p <= 0.05 and lost > gained",
                                   "lost": valid["lost"], "gained": valid["gained"],
                                   "outcome": FAIL if (p <= MCNEMAR_ALPHA and losing) else PASS}
    else:
        checks["valid_mcnemar"] = {"value": None, "threshold": MCNEMAR_ALPHA, "rule": "fail if p <= 0.05 and lost > gained",
                                   "outcome": NOT_MEASURED}
    add("token_identity_pct", metrics["token_identity"]["pct"] if metrics else None,
        limits["token_identity_min_pct"], True, ">=")
    add("mean_kld", kld.get("mean_kld") if kld else None, limits["mean_kld_max"], False, "<=")
    add("top1_agreement_pct", kld.get("same_top_p_pct") if kld else None, limits["top1_agreement_min_pct"], True, ">=")
    return checks


def verdict(checks: Mapping[str, Mapping[str, Any]]) -> str:
    outcomes = {check["outcome"] for check in checks.values()}
    if FAIL in outcomes:
        return FAIL
    if NOT_MEASURED in outcomes:
        return "incomplete"
    return PASS


def ship_check(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    bound = metrics["exact"]["net_loss_upper_bound_pp"] if metrics else None
    return {"value": bound, "threshold": SHIP_NET_LOSS_UPPER_BOUND_PP, "rule": "<=",
            "outcome": _check(bound, SHIP_NET_LOSS_UPPER_BOUND_PP, minimum=False)}


def dev75_parity(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
                 tolerance: int = 2) -> dict[str, Any]:
    """Card C09 export check: DEV75 R1 counts of the quant within ``tolerance`` of bf16."""
    def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {"valid": sum(bool(r["protocol_valid"]) for r in rows), "cap": sum(bool(r["cap_hit"]) for r in rows),
                "edit": sum(bool(not r["expected_noop"] and r["exact_region"]) for r in rows),
                "nofp": sum(bool(r["expected_noop"] and r["suggestion"]) for r in rows)}
    if load_cases(reference).keys() != load_cases(candidate).keys():
        raise ValueError("DEV75 case ids differ")
    ref, cand = counts(reference), counts(candidate)
    differences = {name: cand[name] - ref[name] for name in ref}
    return {"reference": ref, "candidate": cand, "difference": differences, "tolerance": tolerance,
            "within_tolerance": all(abs(v) <= tolerance for v in differences.values())}
