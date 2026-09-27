#!/usr/bin/env python3
"""Read-only, allowlisted Anyscale credit and usage reconciliation."""
from __future__ import annotations
import argparse, datetime as dt, json
from pathlib import Path

CPT_RESERVATION_JOB_IDS = (
    "prodjob_5nza57fm5lfcjfx379ehifckxj",
    "prodjob_sas7lgcknzhw7hkfc7e1ha1v5l",
    "prodjob_ham6ny19ri2c8ek4m6uaxlw2gi",
    "prodjob_neqwkumat4i3rii5yl8af241ks",
)
RESERVATION_USD = 28.0
CAMPAIGN_CEILING_USD = 60.0
SPENT_AT_RESERVATION_USD = 5.905478882


def _rows(value, fields):
    return [{k: (str(row[k]) if k == "date" else row[k]) for k in fields if k in row}
            for row in value.to_dict().get("results", [])]


def collect(client, start: dt.date, end: dt.date):
    api = client._internal_api_client
    from openapi_client.models.aggregated_usage_query import AggregatedUsageQuery
    query = AggregatedUsageQuery(start_date=start, end_date=end, group_by_date=True, asc=True)
    by_type = _rows(api.fetch_usage_group_by_instance_type_api_v2_aggregated_instance_usage_instance_type_post(
        query, count=100, _request_timeout=20), ("date", "instance_type", "anyscale_credits", "dollar_value"))
    by_cluster = _rows(api.fetch_usage_group_by_cluster_api_v2_aggregated_instance_usage_cluster_post(
        query, count=100, _request_timeout=20), ("date", "job_id", "job_name", "anyscale_credits", "dollar_value"))
    credit = api.get_credits_v2_api_v2_organization_billing_credits_v2_get(_request_timeout=20).to_dict()
    relevant = [x for x in by_cluster if x.get("job_id") in CPT_RESERVATION_JOB_IDS]
    observed_charge = sum(float(x.get("dollar_value") or 0) for x in relevant)
    ledger_delta = float(credit["amount_spent_usd"]) - SPENT_AT_RESERVATION_USD
    return {
        "schema": "sepalith.anyscale-long-context-cost-evidence.v1",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "query": {"start": str(start), "end": str(end), "end_semantics": "provider API date range"},
        "credits": {k: credit.get(k) for k in ("current_balance_usd", "amount_spent_usd", "total_granted_usd")},
        "usage_by_instance_type": by_type,
        "reservation_jobs": relevant,
        "reconciliation": {
            "reservation_envelope_usd": RESERVATION_USD,
            "spent_at_reservation_usd": SPENT_AT_RESERVATION_USD,
            "current_cumulative_spent_usd": float(credit["amount_spent_usd"]),
            "ledger_delta_since_reservation_usd": ledger_delta,
            "listed_reservation_job_dollar_value_sum": observed_charge,
            "listed_rows_rounding_gap_usd": ledger_delta - observed_charge,
            "unused_reservation_by_ledger_delta_usd": RESERVATION_USD - ledger_delta,
            "current_balance_already_net_of_actual_charges": True,
            "subtract_full_reservation_from_current_balance": False,
            "campaign_headroom_if_unused_reservation_remains_held_usd": CAMPAIGN_CEILING_USD - (float(credit["amount_spent_usd"]) + RESERVATION_USD - ledger_delta),
            "campaign_headroom_if_root_releases_unused_reservation_usd": CAMPAIGN_CEILING_USD - float(credit["amount_spent_usd"]),
        },
        "allowed_calls": ["credits_GET", "usage_by_instance_type_POST_read_only", "usage_by_cluster_POST_read_only"],
        "mutations": 0,
        "credentials_or_user_fields_serialized": False,
    }


def main():
    p=argparse.ArgumentParser();p.add_argument("--start",type=dt.date.fromisoformat,required=True);p.add_argument("--end",type=dt.date.fromisoformat,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise SystemExit("fresh output required")
    import anyscale
    value=collect(anyscale.Anyscale()._anyscale_client,a.start,a.end)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"reservation_jobs":len(value["reservation_jobs"]),"output":str(a.output)}))
if __name__ == "__main__": main()
