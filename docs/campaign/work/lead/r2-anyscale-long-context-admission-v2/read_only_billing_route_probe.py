#!/usr/bin/env python3
"""Probe newly identified read-only billing routes without serializing secrets.

The Anyscale SDK exposes account billing URLs and organization contract routes,
but these are separate from the previously checked credit, usage, and GPU-fleet
surfaces.  This probe records only route status, response shapes, and redacted
URL descriptors.  Signed billing/dashboard URLs and response bodies never leave
the process.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping


SDK_VERSION = "0.26.108"
TARGETS = ("g6e.2xlarge", "p5.4xlarge")
DASHBOARD_TYPES = ("invoices", "credits", "usage")

# Every account API call in this file is GET-only.  The external dashboard
# dereference is also HTTP GET and is disabled unless --dereference is passed.
ALLOWED_GET_ROUTES = {
    "user_info_GET": "/api/v2/userinfo",
    "organization_contract_info_GET": "/api/v2/billing_scripts/{organization_id}",
    "billing_versions_by_organization_GET": (
        "/api/v2/organization_billing/billing_versions?organization_id={organization_id}"
    ),
    "manage_billing_url_GET": "/api/v2/organization_billing/manage_billing_url",
    "metronome_customer_info_GET": (
        "/api/v2/metronome_customer_info/{organization_id}"
    ),
    "metronome_usage_alerts_GET": "/api/v2/organization_billing/alerts",
    "metronome_dashboard_url_GET": (
        "/api/v2/organization_billing/metronome_embedded_dashboard_url/{dashboard_type}"
    ),
}

SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "credential",
    "customer",
    "organization",
    "account",
    "user",
    "email",
    "api_key",
    "apikey",
)


def _error(exc: BaseException) -> Dict[str, Any]:
    """Return status metadata without response text, IDs, or URLs."""

    return {
        "status": "error",
        "exception_type": type(exc).__name__,
        "http_status": getattr(exc, "status", None),
    }


def _url_descriptor(value: str) -> Dict[str, Any]:
    """Describe a signed URL without emitting its path, query, or token."""

    parsed = urllib.parse.urlsplit(value)
    return {
        "present": True,
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "path_present": bool(parsed.path),
        "query_present": bool(parsed.query),
        "fragment_present": bool(parsed.fragment),
    }


def _response_value(value: Any) -> Any:
    """Convert an SDK value to a JSON-compatible object in memory."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _shape(value: Any, key: str = "", depth: int = 0) -> Any:
    """Keep billing/status fields while suppressing identity-bearing values."""

    if depth > 5:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        key_lower = key.lower()
        if any(part in key_lower for part in SENSITIVE_KEY_PARTS):
            return {"type": "str", "redacted": True, "length": len(value)}
        # Contract and billing schemas can contain useful rate/term fields.  A
        # value is retained only when the field name itself describes a safe
        # billing/status quantity; all other strings become type/length metadata.
        safe_parts = (
            "rate",
            "price",
            "cost",
            "amount",
            "unit",
            "hour",
            "billing",
            "credit",
            "plan",
            "status",
            "type",
            "version",
            "effective",
            "interval",
            "minimum",
            "commitment",
            "discount",
            "currency",
            "region",
            "instance",
            "gpu",
            "contract",
            "term",
            "quota",
            "limit",
            "available",
            "eligible",
            "launch",
            "resource",
            "provider",
            "shape",
            "catalog",
        )
        if any(part in key_lower for part in safe_parts):
            return value[:240]
        return {"type": "str", "length": len(value)}
    if isinstance(value, Mapping):
        return {
            str(k): _shape(v, str(k), depth + 1)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "list",
            "length": len(value),
            "items": [_shape(item, key, depth + 1) for item in list(value)[:20]],
        }
    return {"type": type(value).__name__}


def _call(label: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        return {"status": "returned", "shape": _shape(_response_value(fn()))}
    except Exception as exc:  # noqa: BLE001 - status evidence must survive API denial
        return _error(exc)


def _signed_url_call(label: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        value = _response_value(fn())
        if not isinstance(value, str):
            return {"status": "unexpected_non_string", "type": type(value).__name__}
        result: Dict[str, Any] = {
            "status": "returned",
            "signed_url": _url_descriptor(value),
        }
        if label == "manage_billing_url" or label.startswith("metronome_dashboard_"):
            result["portal_or_dashboard_route"] = True
        return result
    except Exception as exc:  # noqa: BLE001 - see _call
        return _error(exc)


def _http_descriptor(url: str) -> Dict[str, Any]:
    """GET a signed portal URL and retain only transport/body metadata."""

    request = urllib.request.Request(
        url, headers={"User-Agent": "sepalith-pre04-readonly/1"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        body = response.read(4 * 1024 * 1024 + 1)
        truncated = len(body) > 4 * 1024 * 1024
        body_for_scan = body[: 4 * 1024 * 1024]
        text = body_for_scan.decode("utf-8", "replace")
        final_url = urllib.parse.urlsplit(response.geturl())
        lower = text.lower()
        return {
            "status": "returned",
            "status_code": int(getattr(response, "status", 0) or 0),
            "final_host": final_url.netloc,
            "content_type": response.headers.get("Content-Type", "").split(";", 1)[0],
            "bytes_read": len(body),
            "body_truncated": truncated,
            "body_sha256": hashlib.sha256(body_for_scan).hexdigest(),
            "title_present": bool(re.search(r"<title[^>]*>.*?</title>", text, re.I | re.S)),
            "keyword_counts": {
                key: lower.count(key)
                for key in (
                    "g6e.2xlarge",
                    "p5.4xlarge",
                    "l40s",
                    "h100",
                    "price",
                    "rate",
                    "invoice",
                    "subscription",
                    "billing",
                    "usage",
                )
            },
        }


def collect(client: Any, *, dereference: bool = False) -> Dict[str, Any]:
    """Collect account-visible route evidence using GET-only calls."""

    api = client._internal_api_client
    user = api.get_user_info_api_v2_userinfo_get(_request_timeout=20).result
    organizations = getattr(user, "organizations", None) or []
    if not organizations:
        raise RuntimeError("no organization available to query read-only routes")
    organization_id = organizations[0].id

    result: Dict[str, Any] = {
        "schema": "sepalith.anyscale-pre04-long-context-admission-evidence.v2",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sdk_version": SDK_VERSION,
        "targets": list(TARGETS),
        "allowed_get_routes": ALLOWED_GET_ROUTES,
        "organization_present": True,
        "organization_count": len(organizations),
        "calls": {
            "user_info_GET": {
                "status": "returned",
                "organization_count": len(organizations),
            }
        },
        "external_dereference": {"enabled": dereference, "mutations": 0},
        "credentials_or_user_fields_serialized": False,
        "paid_allocation": False,
    }

    result["calls"]["organization_contract_info_GET"] = _call(
        "organization_contract_info",
        lambda: api.get_organization_contract_info_api_v2_billing_scripts_organization_id_get(
            organization_id, _request_timeout=20
        ),
    )
    result["calls"]["billing_versions_by_organization_GET"] = _call(
        "billing_versions",
        lambda: api.get_billing_versions_by_organization_api_v2_organization_billing_billing_versions_get(
            organization_id, _request_timeout=20
        ),
    )
    result["calls"]["manage_billing_url_GET"] = _signed_url_call(
        "manage_billing_url",
        lambda: api.get_manage_billing_url_api_v2_organization_billing_manage_billing_url_get(
            _request_timeout=20
        ),
    )
    result["calls"]["metronome_customer_info_GET"] = _call(
        "metronome_customer_info",
        lambda: api.get_metronome_customer_info_api_v2_metronome_customer_info_organization_id_get(
            organization_id, _request_timeout=20
        ),
    )
    result["calls"]["metronome_usage_alerts_GET"] = _call(
        "metronome_usage_alerts",
        lambda: api.get_organization_metronome_usage_alerts_api_v2_organization_billing_alerts_get(
            _request_timeout=20
        ),
    )

    for dashboard_type in DASHBOARD_TYPES:
        label = f"metronome_dashboard_{dashboard_type}_GET"
        value = _signed_url_call(
            f"metronome_dashboard_{dashboard_type}",
            lambda dashboard_type=dashboard_type: api.get_metronome_embedded_usage_dashboard_api_v2_organization_billing_metronome_embedded_dashboard_url_dashboard_type_get(
                dashboard_type, _request_timeout=20
            ),
        )
        if dereference and value.get("status") == "returned":
            # Re-fetch only inside this process so the signed URL itself is
            # never persisted.  If the API result was malformed, skip safely.
            try:
                signed_url = api.get_metronome_embedded_usage_dashboard_api_v2_organization_billing_metronome_embedded_dashboard_url_dashboard_type_get(
                    dashboard_type, _request_timeout=20
                )
                signed_url = _response_value(signed_url)
                if isinstance(signed_url, str):
                    value["http"] = _http_descriptor(signed_url)
            except Exception as exc:  # noqa: BLE001 - retain transport status only
                value["http"] = _error(exc)
        result["calls"][label] = value

    if dereference:
        portal = result["calls"].get("manage_billing_url_GET", {})
        if portal.get("status") == "returned":
            try:
                signed_url = api.get_manage_billing_url_api_v2_organization_billing_manage_billing_url_get(
                    _request_timeout=20
                )
                signed_url = _response_value(signed_url)
                if isinstance(signed_url, str):
                    portal["http"] = _http_descriptor(signed_url)
            except Exception as exc:  # noqa: BLE001 - retain transport status only
                portal["http"] = _error(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dereference",
        action="store_true",
        help="GET returned signed billing/dashboard URLs; retain only metadata",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    import anyscale

    value = collect(anyscale.Anyscale()._anyscale_client, dereference=args.dereference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "dereference": args.dereference}))


if __name__ == "__main__":
    main()
