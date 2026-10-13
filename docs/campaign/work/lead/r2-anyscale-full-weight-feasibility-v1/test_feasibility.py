import json
from pathlib import Path

HERE = Path(__file__).parent


def test_budget_and_fit_claims_are_exact_and_quote_safe():
    x = json.loads((HERE / "feasibility.json").read_text())
    account = x["account"]
    assert abs(account["campaign_ceiling_usd"] - account["current_cumulative_spent_usd"] - account["remaining_campaign_headroom_usd"]) < 1e-9
    measured = x["measured_full_weight_16k_requirement"]
    assert measured["accumulation16_two_update_peak_reserved_bytes"] > 24_000_000_000
    by_type = {row["instance_type"]: row for row in x["options"]}
    assert by_type["g5.2xlarge"]["technical_fit"] is False
    for target in ("g6e.2xlarge", "p5.4xlarge"):
        assert by_type[target]["exact_all_in_rate"] is None
        assert by_type[target]["fundable_hours"] is None
    assert x["account_api_result"]["new_nonallocating_target_quote_route"] is None
    assert x["decision"] == "funding_not_yet_proven_for_full_16k_training"
