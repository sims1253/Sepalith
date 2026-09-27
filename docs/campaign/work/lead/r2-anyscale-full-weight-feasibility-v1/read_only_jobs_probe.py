#!/usr/bin/env python3
"""Sanitized read-only snapshot of recent Anyscale production-job states."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def collect(client):
    response = client.list_jobs(count=100, sort_field="CREATED_AT", sort_order="DESC")
    rows = []
    for job in response.results:
        state = getattr(job, "state", None)
        current = getattr(state, "current_state", None)
        rows.append(
            {
                "job_id": getattr(job, "id", None),
                "job_name": getattr(job, "name", None),
                "state": str(current) if current is not None else None,
                "status_updated_at": str(getattr(job, "status_updated_at", None)),
            }
        )
    terminal = {"SUCCESS", "SUCCEEDED", "FAILED", "OUT_OF_RETRIES", "TERMINATED", "CANCELED", "CANCELLED"}
    running = [row for row in rows if (row["state"] or "").split(".")[-1] not in terminal]
    return {
        "schema": "sepalith.anyscale-production-job-status.v1",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only_call": "list_jobs_GET",
        "returned_count": len(rows),
        "nonterminal_count": len(running),
        "nonterminal_jobs": running,
        "state_counts": {
            state: sum(1 for row in rows if row["state"] == state)
            for state in sorted({row["state"] for row in rows if row["state"]})
        },
        "credentials_or_user_fields_serialized": False,
        "mutations": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    import anyscale

    value = collect(anyscale.Anyscale()._anyscale_client)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "nonterminal_count": value["nonterminal_count"]}))


if __name__ == "__main__":
    main()
