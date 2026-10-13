import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("probe", HERE / "targeted_account_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Value:
    def __init__(self, value): self.value = value
    def to_dict(self): return self.value
    @property
    def results(self): return self.value


class Cloud:
    id = "secret-cloud-id"
    name = "Hosted"
    provider = "AWS"
    region = "us-east-2"
    state = "ACTIVE"
    status = "ready"


class API:
    def get_credits_v2_api_v2_organization_billing_credits_v2_get(self, **_):
        return Value({"current_balance_usd": 86.0, "amount_spent_usd": 14.0,
          "total_granted_usd": 100.0, "in_use_credits": [{"credit_name": "private",
          "effective_date_start": "2026-01-01", "effective_date_end": "2126-01-01",
          "total_granted_usd": 100.0, "total_balance_usd": 86.0,
          "amount_consumed_usd": 14.0}]})
    def get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(self, *_a, **_k):
        return Value({"result": {"instance_types": []}})
    def get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(self, *_a, **_k):
        return Value({"result": {"groups": [], "node_rollup": {"nodes_total": 0}}})


class Client:
    _internal_api_client = API()
    def list_clouds(self, **_): return Value([Cloud()])
    def get_cloud(self, **_): return type("Full", (), {"is_aioa": True})()


def test_output_is_allowlisted_and_semantically_narrow():
    got = probe.collect(Client())
    text = repr(got)
    assert "secret-cloud-id" not in text and "private" not in text
    assert got["mutations"] == 0
    assert got["interpretation"]["fleet_snapshot_is_entitlement_catalog"] is False
    assert set(got["clouds"][0]["target_fleet_snapshots"]) == set(probe.TARGETS)


def test_source_has_no_mutating_sdk_calls():
    tree = ast.parse((HERE / "targeted_account_probe.py").read_text())
    calls = [n.func.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(any(v in name for v in ("create", "update", "delete", "launch", "submit", "terminate")) for name in calls)


def test_error_never_serializes_message():
    value = probe._error(RuntimeError("https://x/?token=SECRET bearer SECRET"))
    assert value == {"type": "RuntimeError", "status": None}


def test_feasibility_keeps_conditional_price_distinct_from_quote():
    value = json.loads((HERE / "feasibility.json").read_text())
    assert value["decision"] == "not_admissible_from_read_only_evidence"
    assert all(x["exact_all_in_anyscale_rate"] is None for x in value["technical_candidates"])
    math = value["conditional_a100_math_not_a_quote"]
    assert abs(math["hours_from_remaining_campaign_ceiling"] - 32 / 4.9591) < 1e-12
    assert "Only valid if" in math["condition"]


def test_official_evidence_has_primary_sources_and_no_capacity_claim():
    value = json.loads((HERE / "official-evidence.json").read_text())
    urls = [x["url"] for x in value["sources"]]
    assert all(u.startswith(("https://docs.anyscale.com/", "https://www.anyscale.com/", "https://docs.aws.amazon.com/", "https://aws.amazon.com/")) for u in urls)
    assert any("does not establish" in x["finding"] for x in value["sources"])
