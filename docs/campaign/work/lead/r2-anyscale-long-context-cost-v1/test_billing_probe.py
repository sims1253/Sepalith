import ast, importlib.util, json, sys, types
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location("probe",HERE/"billing_probe.py");probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Value:
 def __init__(self,x):self.x=x
 def to_dict(self):return self.x
class Q:
 def __init__(self,**kw):self.kw=kw
class API:
 def fetch_usage_group_by_instance_type_api_v2_aggregated_instance_usage_instance_type_post(self,*a,**k):return Value({"results":[{"date":"2026-09-14","instance_type":"g5.2xlarge","anyscale_credits":2,"dollar_value":1.8,"user_email":"secret"}]})
 def fetch_usage_group_by_cluster_api_v2_aggregated_instance_usage_cluster_post(self,*a,**k):return Value({"results":[{"date":"2026-09-14","job_id":probe.CPT_RESERVATION_JOB_IDS[0],"job_name":"owned","anyscale_credits":2,"dollar_value":1.8,"user_email":"secret"}]})
 def get_credits_v2_api_v2_organization_billing_credits_v2_get(self,**k):return Value({"current_balance_usd":92.294521118,"amount_spent_usd":7.705478882,"total_granted_usd":100,"private":"secret"})
class Client:_internal_api_client=API()

def install_model():
 pkg=types.ModuleType("openapi_client");models=types.ModuleType("openapi_client.models");mod=types.ModuleType("openapi_client.models.aggregated_usage_query");mod.AggregatedUsageQuery=Q
 sys.modules.update({"openapi_client":pkg,"openapi_client.models":models,"openapi_client.models.aggregated_usage_query":mod})
def test_reconciliation_does_not_double_count():
 import datetime;install_model();x=probe.collect(Client(),datetime.date(2026,9,13),datetime.date(2026,9,15))
 assert abs(x["reconciliation"]["ledger_delta_since_reservation_usd"]-1.8)<1e-12
 assert abs(x["reconciliation"]["unused_reservation_by_ledger_delta_usd"]-26.2)<1e-12
 assert x["reconciliation"]["subtract_full_reservation_from_current_balance"] is False
 assert "secret" not in repr(x)
def test_only_known_reservation_jobs_counted():
 import datetime;install_model();x=probe.collect(Client(),datetime.date(2026,9,13),datetime.date(2026,9,15))
 assert len(x["reservation_jobs"])==1
def test_source_has_no_cloud_mutation_calls():
 tree=ast.parse((HERE/"billing_probe.py").read_text());names=[n.func.attr.lower() for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 assert not any(any(v in n for v in ("create","update","delete","launch","submit","terminate")) for n in names)
def test_cost_review_uses_net_balance_and_no_invented_rate():
 x=json.loads((HERE/"cost-review.json").read_text());r=x["reservation_reconciliation"]
 assert abs(r["actual_ledger_delta_since_admission_usd"]-(r["current_cumulative_spent_usd"]-r["paid_admission_cumulative_spent_usd"]))<1e-12
 assert abs(r["unused_envelope_usd"]-(r["original_incremental_envelope_usd"]-r["actual_ledger_delta_since_admission_usd"]))<1e-12
 assert x["pricing"]["single_l40s48_hosted_all_in_usd_or_ac_per_hour"] is None
 assert x["pricing"]["single_h10080_hosted_all_in_usd_or_ac_per_hour"] is None
 assert x["admission"]["exact_fundable_hours"] is None
