import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from read_only_probe import collect


class FakeResponse:
    def __init__(self, value): self.value = value
    def to_dict(self): return self.value


class FakeClient:
    def __init__(self):
        self.calls = []
        self._internal_api_client = self

    def list_clouds(self, count):
        self.calls.append("list_clouds")
        return NS(results=[NS(id="cld_test", name="Anyscale Cloud", provider="AWS", region="us-east-2", state="ACTIVE", status="ready", compute_stack="VM")])

    def list_resource_quotas(self, cloud_id, max_items):
        self.calls.append("list_resource_quotas")
        return []

    def get_cloud_additional_instance_types_api_v2_clouds_cloud_id_additional_instance_types_get(self, cloud_id):
        self.calls.append("get_additional")
        return FakeResponse({"results": []})

    def get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(self, cloud_id):
        self.calls.append("get_gpu_fleet")
        return FakeResponse({"result": {"instance_types": [{"instance_type": "g5.2xlarge"}]}})

    def search_cluster_computes(self, query):
        self.calls.append("search_cluster_computes")
        config = NS(head_node_type=NS(instance_type="g5.2xlarge"), worker_node_types=[])
        return NS(results=[NS(id="cpt_test", name="existing", config=config)])


class ReadOnlyProbeTest(unittest.TestCase):
    def test_uses_only_read_only_inventory_calls(self):
        client = FakeClient()
        got = collect(client)
        self.assertEqual(client.calls, [
            "list_clouds", "list_resource_quotas", "get_additional",
            "get_gpu_fleet", "search_cluster_computes",
        ])
        self.assertEqual(got["clouds"][0]["observed_gpu_fleet_instance_types"]["result"]["instance_types"][0]["instance_type"], "g5.2xlarge")


if __name__ == "__main__": unittest.main()
