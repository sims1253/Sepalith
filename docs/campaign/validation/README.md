# Board validation, September 12, 2026

The final manifest has 84 tasks across eight phases. Its dependency graph is
acyclic, every dependency names a real task, and required tasks do not depend
on conditional or optional branches. All document links and task source pointers
were checked locally; receipt outputs are future artifacts, not existing inputs.

Nine standard-library tests passed with:

```sh
python3 -m unittest discover -s docs/campaign -p 'test_*.py' -q
python3 docs/campaign/campaign.py validate
```

The tests cover dependency gates, completion receipts, deferral rules, revision
conflicts, malformed imports, offline-branch imports, conflicting ownership,
prerequisite rollback and the local HTTP API.

Headless Chrome through the installed Python Playwright package checked desktop
(1440 × 1000) and mobile (390 × 844). Search, readiness filtering, guide opening,
receipt enforcement, save/reload, phase navigation and the mobile detail panel
passed. Mutating UI checks used a temporary campaign directory on port 8767.
The final board on port 8766 was checked read-only and retains revision 0 with
all 84 tasks pending. No experiment progress was created by these tests.

The published page has 84 native expandable tasks and no script tags. The
fetched published bytes match `postplan.html` by SHA-256. Publication metadata,
response policy and test results are in [publish.json](publish.json).

Screenshots:

- [Local desktop](desktop.png)
- [Local mobile](mobile.png)
- [Mobile task brief](mobile-brief.png)
- [Published desktop](postplan-desktop.png)
- [Published mobile](published-mobile.png)

The T3 preview automation host was unavailable in this environment, so local
headless Chrome supplied the browser checks. The PostPlan copy remains a
readable snapshot; status synchronization belongs to the local server/CLI.
