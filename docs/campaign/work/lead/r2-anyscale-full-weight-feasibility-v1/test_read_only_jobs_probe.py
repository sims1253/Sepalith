import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("probe", HERE / "read_only_jobs_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_collect_is_sanitized_and_counts_nonterminal():
    jobs = [
        SimpleNamespace(id="prodjob_a", name="a", state=SimpleNamespace(current_state="RUNNING"), status_updated_at="x"),
        SimpleNamespace(id="prodjob_b", name="b", state=SimpleNamespace(current_state="SUCCESS"), status_updated_at="y"),
        SimpleNamespace(id="prodjob_c", name="c", state=SimpleNamespace(current_state="OUT_OF_RETRIES"), status_updated_at="z"),
    ]
    class Client:
        def list_jobs(self, **kwargs):
            assert kwargs["count"] == 100
            return SimpleNamespace(results=jobs)
    got = probe.collect(Client())
    assert got["nonterminal_count"] == 1
    assert got["nonterminal_jobs"][0]["job_id"] == "prodjob_a"
    assert got["mutations"] == 0
