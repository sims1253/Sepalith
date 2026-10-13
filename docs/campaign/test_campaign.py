#!/usr/bin/env python3
"""Self-contained tests for the campaign state machine and local API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign  # noqa: E402


def make_manifest() -> dict:
    def task(tid: str, kind: str, deps: list[str]) -> dict:
        return {
            "id": tid,
            "phase": "p1",
            "title": f"Task {tid}",
            "summary": f"Summary for {tid}",
            "kind": kind,
            "resource": "CPU",
            "role": "engineer",
            "effort": "1h",
            "dependsOn": deps,
            "steps": [f"Do {tid}"],
            "acceptance": [f"Accept {tid}"],
            "outputs": [f"{tid}.json"],
            "sources": ["docs/72-HOUR-MODEL-PLAN.md"],
            "delegate": "agent",
            "stopRule": "Stop when acceptance is met.",
        }

    return {
        "schemaVersion": 1,
        "campaignId": "test-campaign",
        "title": "Test Campaign",
        "deadline": "2026-09-15T09:00:00+02:00",
        "phases": [{"id": "p1", "title": "Phase 1", "window": "now", "goal": "Test"}],
        "tasks": [
            task("A", "required", []),
            task("B", "conditional", ["A"]),
            task("C", "optional", ["B"]),
        ],
    }


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        (self.directory / "tasks.json").write_text(json.dumps(make_manifest()), encoding="utf-8")
        (self.directory / "board.html").write_text("<h1>board</h1>", encoding="utf-8")
        self.store = campaign.CampaignStore(self.directory)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_manifest_dag_and_dependency_gate(self) -> None:
        self.assertEqual(len(self.store.load_manifest()["tasks"]), 3)
        initial = self.store.read_state()
        self.assertEqual(initial["revision"], 0)
        with self.assertRaises(campaign.CampaignError):
            self.store.update("B", {"status": "in_progress"})

        state = self.store.update("A", {"status": "done", "note": "checked", "receipt": "a.log"})
        self.assertEqual(state["revision"], 1)
        state = self.store.update("B", {"status": "in_progress"})
        self.assertEqual(state["revision"], 2)
        self.assertEqual([task["id"] for task in self.store.ready_tasks()], [])
        state = self.store.update("B", {"status": "done", "note": "checked", "receipt": "b.log"})
        self.assertEqual(state["revision"], 3)
        self.assertEqual([task["id"] for task in self.store.ready_tasks()], ["C"])

    def test_done_requires_note_and_receipt_and_conditional_defer_does_not_open_gate(self) -> None:
        with self.assertRaises(campaign.CampaignError):
            self.store.update("A", {"status": "done"})
        self.store.update("A", {"status": "done", "note": "checked", "receipt": "a.log"})
        state = self.store.update("B", {"status": "deferred", "note": "provider unavailable"})
        self.assertEqual(state["tasks"]["B"]["status"], "deferred")
        with self.assertRaises(campaign.CampaignError):
            self.store.update("C", {"status": "in_progress"})
        with self.assertRaises(campaign.CampaignError):
            self.store.update("A", {"status": "deferred", "note": "skip"})

    def test_revision_race_does_not_overwrite_newer_state(self) -> None:
        state = self.store.update(
            "A", {"status": "done", "note": "first", "receipt": "a.log"}, expected_revision=0
        )
        self.assertEqual(state["revision"], 1)
        with self.assertRaises(campaign.ConflictError):
            self.store.update(
                "A", {"status": "done", "note": "stale", "receipt": "old.log"}, expected_revision=0
            )
        self.assertEqual(self.store.read_state()["tasks"]["A"]["note"], "first")

    def test_import_rejects_unknown_id_and_invalid_done(self) -> None:
        base = self.store.read_state()
        unknown = copy.deepcopy(base)
        unknown["tasks"]["UNKNOWN"] = copy.deepcopy(unknown["tasks"]["A"])
        with self.assertRaises(campaign.CampaignError):
            self.store.import_state(unknown, expected_revision=0)

        invalid_done = copy.deepcopy(base)
        invalid_done["tasks"]["A"]["status"] = "done"
        with self.assertRaises(campaign.CampaignError):
            self.store.import_state(invalid_done, expected_revision=0)

        malformed = copy.deepcopy(base)
        malformed["unexpected"] = True
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_state(self.store.load_manifest(), malformed)

    def test_import_cannot_overwrite_existing_task_progress(self) -> None:
        self.store.update("A", {"status": "in_progress", "owner": "agent-a"})
        offline = self.store.read_state()
        offline["tasks"]["A"]["owner"] = "agent-b"
        with self.assertRaises(campaign.CampaignError):
            self.store.import_state(offline, expected_revision=1)
        self.assertEqual(self.store.read_state()["tasks"]["A"]["owner"], "agent-a")

    def test_import_browser_branch_with_its_own_revision(self) -> None:
        exported = self.store.read_state()
        exported['revision'] = 5
        exported['tasks']['A'].update(status='done', note='Verified offline', receipt='offline.md')
        source = self.directory / 'browser-progress.json'
        source.write_text(json.dumps(exported))
        imported = self.store.import_file(source)
        self.assertEqual(imported['revision'], 1)
        self.assertEqual(imported['tasks']['A']['status'], 'done')
        # A later file still cannot erase the already accepted work.
        exported['tasks']['A']['note'] = 'An incompatible branch'
        source.write_text(json.dumps(exported))
        with self.assertRaises(campaign.CampaignError):
            self.store.import_file(source)

    def test_rollback_prerequisite_is_rejected_when_descendant_started(self) -> None:
        self.store.update("A", {"status": "done", "note": "checked", "receipt": "a.log"})
        self.store.update("B", {"status": "in_progress"})
        with self.assertRaises(campaign.CampaignError):
            self.store.update("A", {"status": "todo"})

        imported = self.store.read_state()
        imported["tasks"]["A"]["status"] = "todo"
        imported["tasks"]["B"]["status"] = "todo"
        with self.assertRaises(campaign.CampaignError):
            self.store.import_state(imported, expected_revision=2)

    def test_brief_contains_control_constraints(self) -> None:
        brief = self.store.brief("A")
        self.assertIn("CUDA launch is lead-only", brief)
        self.assertIn("never launches a job", brief)
        self.assertIn("## Acceptance", brief)

    def test_http_api_conflict_origin_and_safe_static(self) -> None:
        server = campaign.CampaignHTTPServer(("127.0.0.1", 0), self.directory)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base_url + "/api/campaign", timeout=3) as response:
                payload = json.loads(response.read())
            self.assertIn("manifest", payload)
            self.assertEqual(payload["state"]["revision"], 0)

            body = json.dumps(
                {
                    "revision": 0,
                    "taskId": "A",
                    "changes": {"status": "done", "note": "api", "receipt": "api.log"},
                }
            ).encode()
            with urlopen(
                Request(base_url + "/api/state", data=body, headers={"Content-Type": "application/json"}),
                timeout=3,
            ) as response:
                state = json.loads(response.read())
            self.assertEqual(state["revision"], 1)

            with self.assertRaises(HTTPError) as conflict:
                urlopen(
                    Request(base_url + "/api/state", data=body, headers={"Content-Type": "application/json"}),
                    timeout=3,
                )
            self.assertEqual(conflict.exception.code, 409)
            self.assertIn("error", json.loads(conflict.exception.read()))

            evil_headers = {"Content-Type": "application/json", "Origin": "https://evil.example"}
            with self.assertRaises(HTTPError) as forbidden:
                urlopen(Request(base_url + "/api/state", data=body, headers=evil_headers), timeout=3)
            self.assertEqual(forbidden.exception.code, 403)

            with urlopen(base_url + "/", timeout=3) as response:
                self.assertEqual(response.read(), b"<h1>board</h1>")
            with self.assertRaises(HTTPError) as hidden:
                urlopen(base_url + "/state.json", timeout=3)
            self.assertEqual(hidden.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
