import importlib.util,pathlib,tempfile,unittest
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('account_probe_v2',HERE/'account_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class ProbeTests(unittest.TestCase):
 def test_call_surface_is_read_only(self):
  self.assertEqual(set(probe.ALLOWED_CALLS),{'list_clouds','get_cloud','list_resource_quotas','additional_instance_types_GET','gpu_fleet_instance_types_GET','list_machine_pools_GET','default_compute_GET','plan_status_GET','active_billing_version_GET','cloud_resources_GET','search_cluster_computes'})
  forbidden={'create','update','delete','launch','submit','attach','put','patch'}
  for name in probe.ALLOWED_CALLS:self.assertTrue(forbidden.isdisjoint(name.lower().split('_')))
 def test_errors_are_redacted(self):
  class E(Exception):status=400
  value=probe.safe_error(E('secret token=abc; Anyscale-hosted clouds can not be fetched'))
  self.assertEqual(value,{'type':'E','status':400,'provider_detail':'hosted_cloud_resources_unavailable'})
  self.assertNotIn('secret',str(value))
if __name__=='__main__':unittest.main()
